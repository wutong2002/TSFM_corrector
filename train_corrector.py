import os
import sys
import argparse
import logging
import torch
import numpy as np
import random

# 路径 Hack
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.grid_config import ENCODER_CONFIGS, CORRECTOR_CONFIGS, TRAIN_CONFIGS
from configs.correction_args import add_correction_args
from corrector.trainer import CorrectionTrainer
from corrector.corrector_model import (
    DeepTransformerCorrector, AttentionCorrector, LinearWeightedCorrector, 
    SimilarityWeightedCorrector, WeightedBaselineCorrector, LearnableWeightedCorrector,
    MLPCorrector, ZeroCorrector, StandardTransformerCorrector, LightWeightMetaCorrector
)

# 尝试导入数据集分组配置
try:
    from processed_datasets.dataset_config import DATASET_GROUPS
except ImportError:
    try:
        from Datasets.processed_datasets.dataset_config import DATASET_GROUPS
    except ImportError:
        print("⚠️ 未找到 dataset_config.py，将使用原始组名")
        DATASET_GROUPS = {}

def setup_logger(log_file):
    logger = logging.getLogger("Corrector")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    sh = logging.StreamHandler(); sh.setFormatter(formatter); logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter); logger.addHandler(fh)
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
        "attention": AttentionCorrector
    }
    
    if arch not in MODEL_MAP:
        raise ValueError(f"Unknown corrector architecture: {arch}")
    
    return MODEL_MAP[arch](config)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Residual Corrector (Optimized)")
    
    # 1. 先添加公共参数 (包含 train_group/test_group)
    add_correction_args(parser)
    
    # 2. 添加脚本特有参数
    parser.add_argument("--encoder_config", type=str, required=True, choices=ENCODER_CONFIGS.keys())
    parser.add_argument("--corrector_configs", type=str, nargs='+', required=True, 
                        help="List of correctors to run sequentially")
    
    parser.add_argument("--train_config", type=str, default="standard", choices=TRAIN_CONFIGS.keys())
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="results/grid_search")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2025)
    
    # 注意：不要重复定义 --train_group 等，因为 add_correction_args 已经添加了
    
    args = parser.parse_args()
    set_seed(args.seed)

    # 3. 准备配置
    if args.encoder_config not in ENCODER_CONFIGS:
        raise ValueError(f"Encoder config '{args.encoder_config}' not found")
    
    enc_cfg = ENCODER_CONFIGS[args.encoder_config]
    db_config = enc_cfg.copy()
    db_config['data_dir'] = args.data_source_path if hasattr(args, 'data_source_path') and args.data_source_path else args.data_root
    if 'context_len' not in db_config: db_config['context_len'] = 96 

    if args.train_config not in TRAIN_CONFIGS:
        raise ValueError(f"Train config '{args.train_config}' not found")
    
    base_train_config = TRAIN_CONFIGS[args.train_config]
    
    # 解析数据集列表
    train_ds_list = DATASET_GROUPS.get(args.train_group, [args.train_group])
    test_ds_list = DATASET_GROUPS.get(args.test_group, [args.test_group])

    # =========================================================================
    # 🛠️ [Fix] 智能参数合并逻辑 (Smart Merge)
    # 规则: 
    # 1. 必选参数直接覆盖
    # 2. 可选参数：仅当命令行传了非默认值时，才覆盖 Config 中的值；否则保留 Config 值
    # =========================================================================
    
    # --- 1. 必选覆盖项 ---
    base_train_config.update({
        "train_datasets_list": train_ds_list,
        "test_datasets_list": test_ds_list,
        "seed": args.seed,
        "debug": args.debug,
        "output_dir": args.output_root
    })

    # --- 2. 条件覆盖项 (Smart Overwrite) ---
    
    # Max Samples (Default in args: -1)
    if args.max_samples_per_dataset != -1:
        base_train_config["max_samples_per_dataset"] = args.max_samples_per_dataset
    elif "max_samples_per_dataset" not in base_train_config:
        base_train_config["max_samples_per_dataset"] = -1

    # Max Test Samples (Default in args: -1)
    if args.max_test_samples_per_dataset != -1:
        base_train_config["max_test_samples_per_dataset"] = args.max_test_samples_per_dataset
    elif "max_test_samples_per_dataset" not in base_train_config:
        base_train_config["max_test_samples_per_dataset"] = -1

    # Pseudo Ratio (Default in args: 0.0)
    if args.pseudo_ratio != 0.0:
        base_train_config["pseudo_ratio"] = args.pseudo_ratio
    elif "pseudo_ratio" not in base_train_config:
        base_train_config["pseudo_ratio"] = 0.0
        
    # Pseudo Strength (Default in args: 0.8)
    if args.pseudo_strength != 0.8:
        base_train_config["pseudo_strength"] = args.pseudo_strength
    elif "pseudo_strength" not in base_train_config:
        base_train_config["pseudo_strength"] = 0.8

    # Retrieval Scope (Default in args: "global")
    # Config 中通常设置为 "cross_dataset"
    if args.retrieval_scope != "global": 
        base_train_config["retrieval_scope"] = args.retrieval_scope
    elif "retrieval_scope" not in base_train_config:
        base_train_config["retrieval_scope"] = "global"

    # Top K (Default in args: 5)
    # Config 中通常设置为 100
    if args.top_k != 5:
        base_train_config["top_k"] = args.top_k
    elif "top_k" not in base_train_config:
        base_train_config["top_k"] = 5

    # =========================================================================

    # =================================================================
    # 🚀 核心优化: 共享数据缓存 (Data Cache)
    # =================================================================
    # 格式: (db, train_samples, test_samples_dict, train_ds, val_loaders)
    shared_data_cache = None 

    print(f"🚀 开始训练循环 | Encoder: {args.encoder_config} | Correctors: {args.corrector_configs}")
    print(f"📌 [Config Check] Max Train Samples: {base_train_config.get('max_samples_per_dataset')} (Expected: from config if not set in args)")
    print(f"📌 [Config Check] Retrieval Scope:   {base_train_config.get('retrieval_scope')}")
    print("="*80)

    for i, corr_name in enumerate(args.corrector_configs):
        if corr_name not in CORRECTOR_CONFIGS:
            print(f"❌ 警告: 未知 Corrector '{corr_name}'，跳过。")
            continue
            
        # 准备配置
        model_config = CORRECTOR_CONFIGS[corr_name].copy()
        model_config['embed_dim'] = db_config.get('output_dim', db_config.get('embed_dim', 64))
        model_config['context_len'] = db_config['context_len']
        model_config['pred_len'] = args.pred_len
        model_config['top_k'] = base_train_config.get('top_k', 5)
        
        # 构造输出路径
        exp_dir = os.path.join(args.output_root, args.encoder_config, corr_name, f"seed_{args.seed}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # 设置 Logger
        logger = setup_logger(os.path.join(exp_dir, "train.log"))
        logger.info(f"🔄 [Start] Running Corrector: {corr_name} ({i+1}/{len(args.corrector_configs)})")
        
        # 实例化模型
        try:
            model = get_corrector_model(model_config)
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            continue
            
        # 更新训练配置
        current_train_cfg = base_train_config.copy()
        current_train_cfg['output_dir'] = exp_dir
        current_train_cfg['logger'] = logger
        
        # 实例化 Trainer (传入缓存!)
        trainer = CorrectionTrainer(
            model=model,
            model_config=model_config,
            db_config=db_config,
            train_config=current_train_cfg,
            preloaded_data=shared_data_cache 
        )
        
        # 运行训练
        try:
            trainer.run()
            
            # === [关键] 第一次运行后，保存构建好的 Dataset 供后续复用 ===
            if shared_data_cache is None and trainer.train_ds is not None:
                logger.info("📦 [Cache] 正在缓存构建好的数据集 (Dataset & Loaders) 以供复用...")
                shared_data_cache = (
                    trainer.db, 
                    trainer.train_samples, 
                    trainer.test_samples_dict,
                    trainer.train_ds,      
                    trainer.val_loaders    
                )
                
        except Exception as e:
            logger.error(f"❌ Training failed for {corr_name}: {e}")
            import traceback
            traceback.print_exc()
            
        print("-" * 80)

    print("✅ 所有任务完成。")

if __name__ == "__main__":
    main()