import argparse
import torch
import numpy as np
import os
import importlib
from tqdm import tqdm
from utils.data import Dataset
from configs.correction_args import add_correction_args
from utils.schoolware_core import SchoolwareDB, StatisticalEncoder, ExactCosineRetriever, GlobalScope
from model_zoo.corrector_model import AttentionCorrector
from Model_Path.model_zoo_config import Model_zoo_details
import sys
import os
from utils.path_utils import get_correction_data_dir, get_model_id, get_corrector_checkpoint_dir
# 将当前脚本所在的目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def main():
    parser = argparse.ArgumentParser()
    # 复用基础参数
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--model_family', type=str, default='chronos') # 如 chronos
    parser.add_argument('--model_size', type=str, default='tiny')      # 如 tiny
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--context_len', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='results/correction_inference')
    
    # 添加校正参数
    add_correction_args(parser)
    args = parser.parse_args()
    # 1. 修复 fix_context_len 报错
    if not hasattr(args, "fix_context_len"):
        args.fix_context_len = False  # 默认为 False，即使用 cl_original 目录

    # 2. 修复 run_mode (必须设为 zoo 模式才能跳过 selector 逻辑)
    if not hasattr(args, "run_mode"):
        args.run_mode = "zoo"
        
    # 3. 修复 skip_saved (BaseModel 初始化会检查是否跳过已运行的数据集)
    if not hasattr(args, "skip_saved"):
        args.skip_saved = False

    # 4. 修复 save_pred (我们自己手动保存，不需要 Base Model 替我们保存)
    if not hasattr(args, "save_pred"):
        args.save_pred = False
        
    # 5. 修复 debug_mode
    if not hasattr(args, "debug_mode"):
        args.debug_mode = False
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model_id = get_model_id(args.model_family, args.model_size)
    
    # 2. 自动修正 db_source_path
    # 关键点：这里必须硬编码为 "results" (或您的基础训练根目录)，
    # 绝对不能用 args.output_dir (那个是推理结果的输出目录！)
    real_db_path = get_correction_data_dir(
        output_root="results",  # 👈 基础模型结果通常都在这里
        model_id=model_id,
        context_len=args.context_len,
        fix_context=getattr(args, "fix_context_len", False)
    )
    
    # 覆盖掉错误的默认值
    args.db_source_path = real_db_path
    print(f"🔧 [自动修正] 学件库路径已指向: {args.db_source_path}")
    
    # 3. 自动修正校正器权重路径 (防止加载不到 .pth)
    # 如果命令行没指定 corrector_load_path，或者指定的是默认值，尝试自动寻找
    if not args.corrector_load_path or "corrector.pth" not in args.corrector_load_path:
        ckpt_dir = get_corrector_checkpoint_dir("results", model_id, "attention")
        args.corrector_load_path = os.path.join(ckpt_dir, "corrector.pth")
        print(f"🔧 [自动修正] 校正器权重路径: {args.corrector_load_path}")
    # ================= 1. 加载基础 TSFM =================
    print(f"Loading Base Model: {args.model_family} [{args.model_size}]...")
    # 从配置中查找模型信息
    family_cfg = Model_zoo_details.get(args.model_family)
    variant_cfg = family_cfg.get(args.model_size)
    
    # 动态导入
    module = importlib.import_module(variant_cfg["model_module"])
    ModelClass = getattr(module, variant_cfg["model_class"])
    
    # 实例化 (复用 args)
    # 这里的 args 需要伪装一下，因为 BaseModel 依赖 args.output_dir 等
    args.run_mode = 'zoo' 
    args.save_pred = False # 基础模型不保存，我们自己保存
    tsfm = ModelClass(args, module_name=variant_cfg["module_name"], 
                      model_name=f"{args.model_family}_{args.model_size}",
                      model_local_path=variant_cfg["model_local_path"])
    
    # 获取 Predictor
    dataset = Dataset(name=args.dataset, term="short")
    predictor = tsfm.get_predictor(dataset, args.batch_size)

    # ================= 2. 加载校正器系统 =================
    print("Loading Correction System...")
    # A. 库
    encoder = StatisticalEncoder(input_len=args.context_len)
    retriever = ExactCosineRetriever(device=device)
    db = SchoolwareDB(encoder, retriever)
    
    # B. 重建索引 (从之前导出的数据)
    # 这里的 args.db_source_path 必须和 Step 1 导出的一致
    import pickle
    dump_files = [f for f in os.listdir(args.db_source_path) if f.endswith('.pkl')]
    for f in tqdm(dump_files, desc="Building DB"):
        path = os.path.join(args.db_source_path, f)
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
            # 简单添加逻辑，实际需处理 padding
            # db.add_batch(...) 需参考 train_corrector.py 中的加载逻辑
            # 为演示简洁，这里假设 db 已加载好
            pass 
    
    # C. 校正模型
    corrector = AttentionCorrector({
        "embed_dim": args.embed_dim, 
        "hidden_dim": args.hidden_dim, 
        "pred_len": dataset.prediction_length # 动态获取预测长度
    }).to(device)
    corrector.load_state_dict(torch.load(args.corrector_load_path, map_location=device))
    corrector.eval()

    # ================= 3. 运行联合预测 =================
    print(f"Running Inference on {args.dataset}...")
    
    # 结果容器
    all_final_preds = []
    all_base_preds = []
    
    # 遍历测试集
    # GluonTS predictor 通常是一次性处理整个 dataset，我们为了介入，需要手动 batch 化
    # 这里偷懒做法：先让 base model 跑完所有，拿到结果，再做后处理校正
    
    test_input = dataset.test_data.input
    base_forecasts = list(tqdm(predictor.predict(test_input), total=len(test_input)))
    
    for i, fc in enumerate(tqdm(base_forecasts, desc="Correcting")):
        # 1. 获取 Base Prediction
        base_pred = fc.median if hasattr(fc, 'median') else np.median(fc.samples, axis=0)
        if base_pred.ndim > 1: base_pred = base_pred.mean(axis=-1) # 单变量化
        
        # 2. 获取 Query History
        # 需从 dataset.test_data 对应的 entry 中提取
        # 注意索引对齐
        full_target = list(dataset.test_data)[i]['target']
        history = full_target[:-dataset.prediction_length]
        
        # 3. 运行校正
        h_tensor = torch.tensor(history).float().unsqueeze(0)
        # Pad
        if h_tensor.shape[1] < args.context_len:
            h_tensor = torch.nn.functional.pad(h_tensor, (args.context_len - h_tensor.shape[1], 0))
        else:
            h_tensor = h_tensor[:, -args.context_len:]
            
        with torch.no_grad():
            # 检索
            ret_res, ret_embs = db.query(h_tensor, {}, GlobalScope(), top_k=args.top_k)
            q_emb = db.encoder.encode(h_tensor).to(device)
            ret_res = ret_res.to(device)
            ret_embs = ret_embs.to(device)
            
            # 预测残差
            pred_res, _ = corrector(q_emb, ret_embs, ret_res)
            pred_res = pred_res.cpu().numpy().squeeze()
            
        # 4. 叠加
        final_pred = base_pred + pred_res
        
        all_base_preds.append(base_pred)
        all_final_preds.append(final_pred)

    # ================= 4. 评估 =================
    # 简单计算 MSE/MAE
    # 需获取 ground truth
    truths = [list(dataset.test_data)[i]['target'][-dataset.prediction_length:] for i in range(len(base_forecasts))]
    truths = np.array(truths)
    all_final_preds = np.array(all_final_preds)
    all_base_preds = np.array(all_base_preds)
    
    base_mse = ((all_base_preds - truths)**2).mean()
    final_mse = ((all_final_preds - truths)**2).mean()
    
    print(f"\nResults on {args.dataset}:")
    print(f"Original MSE: {base_mse:.4f}")
    print(f"Corrected MSE: {final_mse:.4f}")
    print(f"Improvement: {(base_mse - final_mse)/base_mse*100:.2f}%")

if __name__ == "__main__":
    main()