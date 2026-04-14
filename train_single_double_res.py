import os
import sys
import argparse
import logging
import torch
import json
import itertools
import numpy as np
import hashlib
import gc

# [新增] 强制显存清理
torch.cuda.empty_cache()
gc.collect()

# [关键修复] 防止 Windows 下 PyTorch 与 FAISS/NumPy 的 OpenMP 冲突
# 建议放在最前面
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 路径 Hack
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.grid_config import ENCODER_CONFIGS, CORRECTOR_CONFIGS
from configs.single_grid_config import SEARCH_GRIDS, BASE_CONFIG
# 导入数据集配置
from Datasets.processed_datasets.dataset_config import DATASET_GROUPS

from corrector.trainer import CorrectionTrainer
from corrector.corrector_model import (
    DeepTransformerCorrector, AttentionCorrector, DualLatentCrossAttnCorrector, IntraInterRouterCorrector, LinearWeightedCorrector, SemanticRouterCorrector, 
    SimilarityWeightedCorrector, WeightedBaselineCorrector, LearnableWeightedCorrector,
    MLPCorrector, ZeroCorrector, StandardTransformerCorrector, LightWeightMetaCorrector,
    DualSourceFusionCorrector, DualSourceSetMLPCorrector, DualSourceGatedMLPCorrector,
    DualSourceResMLPCorrector, MeanRetrievalCorrector, GlobalBiasCorrector, 
    LocalResARCorrector,RandomFrozenEncoderSetMLPCorrector
)
try:
    from Model_Path.model_zoo_config import Model_zoo_details
except ImportError:
    print("❌ 错误: 无法导入 Model_Path.model_zoo_config")
    exit(1)
def setup_logger(log_file):
    logger = logging.getLogger("SingleTSFMGrid")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    sh = logging.StreamHandler(); sh.setFormatter(formatter); logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8'); fh.setFormatter(formatter); logger.addHandler(fh)
    return logger

def get_corrector_model(config):
    """模型工厂"""
    arch = config.get("corrector_arch")
    MODEL_MAP = {
        "standard_transformer": StandardTransformerCorrector,
        "similarity_weighted": SimilarityWeightedCorrector,
        "linear": LinearWeightedCorrector, 
        "mlp": MLPCorrector,
        "BSA1": WeightedBaselineCorrector, 
        "BSA2": LightWeightMetaCorrector,
        "learnable_weighted": LearnableWeightedCorrector, 
        "zero": ZeroCorrector,
        "deep_transformer": DeepTransformerCorrector,
        "attention": AttentionCorrector,
        "dual_source_fusion": DualSourceFusionCorrector,
        "dual_set_mlp": DualSourceSetMLPCorrector,
        "dual_gated_mlp": DualSourceGatedMLPCorrector,
        "dual_res_mlp": DualSourceResMLPCorrector,
        "mean_retrieval": MeanRetrievalCorrector,
        "global_bias": GlobalBiasCorrector,
        "local_ar": LocalResARCorrector,
        "dual_frozen_set_mlp": RandomFrozenEncoderSetMLPCorrector,
        "semantic_router" : SemanticRouterCorrector,
        "intra_inter_router": IntraInterRouterCorrector,
        "dual_latent_cross_attn": DualLatentCrossAttnCorrector,
    }
    
    if arch not in MODEL_MAP:
        if arch == "std_tf": return StandardTransformerCorrector(config)
        raise ValueError(f"Unknown corrector architecture: {arch}")
    
    return MODEL_MAP[arch](config)


def get_real_tsfm_name(data_root, target_tsfm):
    tsfm_dir = os.path.join(data_root, target_tsfm)
    if not os.path.exists(tsfm_dir):
        parent = os.listdir(data_root)
        candidates = [d for d in parent if d.lower() == target_tsfm.lower()]
        if candidates:
            return candidates[0]
        else:
            raise FileNotFoundError(f"❌ 找不到 TSFM 目录: {tsfm_dir}")
    return target_tsfm

def main():
    parser = argparse.ArgumentParser(description="Single TSFM Grid Search")
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--corrector", type=str, required=True)
    parser.add_argument("--target_tsfm", type=str, required=True, help="Target TSFM folder name")
    parser.add_argument("--grid_mode", type=str, default="standard", help="Key in SEARCH_GRIDS")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    
    # [关键修复 1] 添加 Beta 参数
    parser.add_argument("--retrieval_alpha", type=float, default=1.0, help="Weight for Sequence Similarity")
    parser.add_argument("--retrieval_beta", type=float, default=1.0, help="Mix Ratio for Retrieval Strategy")
    
    # 分组参数
    parser.add_argument("--train_group", type=str, default=BASE_CONFIG.get("train_group", "lotsa_train_clean"), 
                        help="Key in DATASET_GROUPS OR a single dataset name")
    parser.add_argument("--test_group", type=str, default=BASE_CONFIG.get("test_group", "ge_test_all"), 
                        help="Key in DATASET_GROUPS OR a single dataset name")
    
    args = parser.parse_args()

    real_tsfm_name = get_real_tsfm_name(args.data_root, args.target_tsfm)
    
    train_list = DATASET_GROUPS.get(args.train_group, [args.train_group])
    test_list = DATASET_GROUPS.get(args.test_group, [args.test_group])

    print(f"📚 数据集加载配置:")
    print(f"   - Train Group: '{args.train_group}' -> {len(train_list)} datasets")
    print(f"   - Test Group:  '{args.test_group}'  -> {len(test_list)} datasets")

    if args.encoder not in ENCODER_CONFIGS: raise ValueError(f"Unknown Encoder: {args.encoder}")
    if args.corrector not in CORRECTOR_CONFIGS: raise ValueError(f"Unknown Corrector: {args.corrector}")
    if args.grid_mode not in SEARCH_GRIDS: raise ValueError(f"Unknown Grid Mode: {args.grid_mode}")

    db_config = ENCODER_CONFIGS[args.encoder]
    db_config['data_dir'] = args.data_root
    db_config['context_len'] = 512
    base_corrector_cfg = CORRECTOR_CONFIGS[args.corrector]
    grid_params = SEARCH_GRIDS[args.grid_mode]

    keys, values = zip(*grid_params.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🚀 启动搜索 | TSFM: {real_tsfm_name} | Encoder: {args.encoder} | Corrector: {args.corrector}")
    print(f"📋 模式: {args.grid_mode} | 实验数: {len(param_combinations)}")

    shared_db_cache = None 
    shared_ds_cache = None
    last_ds_signature = {} 

    for idx, params in enumerate(param_combinations):
        current_train_cfg = BASE_CONFIG.copy()
        
        # [关键修复 2] 将 Alpha 和 Beta 写入配置
        current_train_cfg['retrieval_alpha'] = args.retrieval_alpha 
        current_train_cfg['retrieval_beta'] = args.retrieval_beta
        
        current_train_cfg.update(params) # 如果 Grid 中有定义，会覆盖命令行参数
        
        current_train_cfg['train_datasets_list'] = train_list
        current_train_cfg['test_datasets_list'] = test_list
        current_train_cfg['target_tsfm_filter'] = real_tsfm_name 
        
        # [关键修复 3] 更新哈希生成逻辑，包含 Beta
        param_str_full = "_".join([f"{k}{v}".replace(".", "p") for k, v in params.items()])
        param_str_full += f"_a{current_train_cfg['retrieval_alpha']}_b{current_train_cfg['retrieval_beta']}"
        param_hash = hashlib.md5(param_str_full.encode()).hexdigest()[:8]
        exp_name = f"run_{idx:03d}_{param_hash}"
        
        exp_dir = os.path.join(args.output_root, real_tsfm_name, args.encoder, args.corrector, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        current_train_cfg['output_dir'] = exp_dir
        current_train_cfg['logger'] = setup_logger(os.path.join(exp_dir, "train.log"))
        logger = current_train_cfg['logger']
        
        logger.info(f"👉 [Run {idx+1}/{len(param_combinations)}] ID: {exp_name}")
        logger.info(f"📋 Params: {params}")
        logger.info(f"🛠️ Alpha: {current_train_cfg['retrieval_alpha']} | Beta: {current_train_cfg['retrieval_beta']}")
        
        # [关键修复 4] 更新缓存敏感键，加入 retrieval_beta
        # 这一步至关重要：如果 beta 变了，必须重新构建 Dataset，不能复用 Cache
        ds_sensitive_keys = ['top_k', 'diversity_max_per_dataset', 'pseudo_ratio', 'retrieval_alpha', 'retrieval_beta']
        current_ds_signature = {k: current_train_cfg.get(k) for k in ds_sensitive_keys}
        
        preloaded_data = None
        if shared_ds_cache is not None and current_ds_signature == last_ds_signature:
            logger.info("♻️ [Cache] 复用完整 Dataset")
            preloaded_data = shared_ds_cache
        elif shared_db_cache is not None:
            logger.info("♻️ [Cache] 复用 DB")
            preloaded_data = shared_db_cache
        
        current_model_cfg = base_corrector_cfg.copy()
        current_model_cfg['embed_dim'] = db_config.get('embed_dim', 128)
        if args.encoder == 'hybrid_math': current_model_cfg['embed_dim'] = 798
        if args.encoder == 'advanced_hybrid_math': current_model_cfg['embed_dim'] = 112
            
        current_model_cfg['top_k'] = current_train_cfg['top_k']
        current_model_cfg.update(params)
        
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        try:
            model = get_corrector_model(current_model_cfg)
        except Exception as e:
            logger.error(f"❌ 模型初始化失败 (参数错误?): {e}")
            continue
        
        trainer = CorrectionTrainer(
            model=model,
            model_config=current_model_cfg,
            db_config=db_config,
            train_config=current_train_cfg, 
            preloaded_data=preloaded_data
        )
        
        try:
            trainer.run()
            
            # =========================================================
            # [关键修复 5] 保存全量超参数 (包含 Beta)
            # =========================================================
            final_saved_config = current_model_cfg.copy()
            
            final_saved_config.update({
                "experiment_id": exp_name,
                "target_tsfm": real_tsfm_name,
                "encoder_type": args.encoder,
                
                "retrieval_alpha": current_train_cfg.get("retrieval_alpha"),
                "retrieval_beta": current_train_cfg.get("retrieval_beta"), # Save Beta
                "retrieval_scope": current_train_cfg.get("retrieval_scope"),
                
                "train_group": args.train_group,
                "test_group": args.test_group,
                "seed": args.seed,
                
                "batch_size": current_train_cfg.get("batch_size"),
                "learning_rate": current_train_cfg.get("learning_rate"),
                "weight_decay": current_train_cfg.get("weight_decay"),
                "optimizer": current_train_cfg.get("optimizer"),
                "scheduler": current_train_cfg.get("scheduler"),
                "epochs": current_train_cfg.get("epochs")
            })

            json_path = os.path.join(exp_dir, "hyperparams.json")
            with open(json_path, "w") as f:
                safe_cfg = {k: (v.item() if isinstance(v, torch.Tensor) else v) 
                           for k, v in final_saved_config.items()}
                json.dump(safe_cfg, f, indent=4)
                
            logger.info(f"💾 已保存全量超参数至: {json_path}")
            # =========================================================
            
            if shared_db_cache is None:
                shared_db_cache = (trainer.db, trainer.train_samples, trainer.test_samples_dict)
            
            shared_ds_cache = (
                trainer.db, trainer.train_samples, trainer.test_samples_dict,
                trainer.train_ds, trainer.val_loaders
            )
            last_ds_signature = current_ds_signature
            logger.info("✅ 本轮实验完成")
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()