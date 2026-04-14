# configs/grid_config.py

# =========================================================
# 1. 编码器配置 (Encoder Configurations)
# =========================================================
ENCODER_CONFIGS = {
    # === 深度学习基模型 (DL Models) ===
    "units": {
        "encoder_type": "units",
        "embed_dim": 128,
        "ckpt_path": "checkpoints/units_x128_pretrain_checkpoint.pth",
        "context_len": 512  
    },
    
    "timesfm": {
        "encoder_type": "timesfm",
        "embed_dim": 1280, 
        "ckpt_path": "checkpoints/timesfm-2.5-200m-pytorch",
        "context_len": 512
    },
    
    "moirai": {
        "encoder_type": "moirai",
        # [Fix 1] Moirai Base 的输出维度是 768，Small 才是 384
        "embed_dim": 768,  
        "ckpt_path": "checkpoints/moirai-1.1-R-base",
        "patch_size": 32,
        "context_len": 512
    },

    # === 数学/统计方法 (Math/Stats Methods) ===
    # [Fix 2] 必须添加 context_len，否则 trainer.py 中的 _ingest_to_db 会报错
    
    "fft": {
        "encoder_type": "fft",
        "embed_dim": 128,  
        "output_dim": 128,
        "context_len": 512  # 新增
    },
    
    "stats": {
        "encoder_type": "stats", 
        "embed_dim": 128,       
        "output_dim": 128,
        "context_len": 512  # 新增
    },
    
    "wavelet": {
        "encoder_type": "wavelet", 
        "embed_dim": 128,          
        "output_dim": 128,
        "wavelet_func": "db4",     
        "level": 3,
        "context_len": 512  # 新增
    },
    "hybrid_math": {
        "encoder_type": "hybrid_math",
        "embed_dim": 798,        # [警告] 此数值与 context_len=512 强绑定
        "wavelet_func": "db4",
        "level": 3,
        "context_len": 512       # 必须固定，否则 embed_dim 会失效
    },
    "advanced_hybrid_math":{
        "encoder_type": "advanced_hybrid_math",
        "embed_dim": 112,
        "context_len": 512
    },
    "random_nn_frozen": {
    "encoder_type": "random_nn_frozen",  # 对应 ENCODER_MAP 中的 Key
    "embed_dim": 128,                   # 建议与修正器默认的 embed_dim 保持一致 (128)
    "context_len": 512,                 # 输入序列的最大上下文长度
    "frozen": True                      # 显式标注该编码器为冻结状态
},
}

# =========================================================
# 2. 校正器架构配置 (Corrector Architectures)
# =========================================================
CORRECTOR_CONFIGS = {
    # 1. Standard Transformer (最强的基线)
    

    # 2. Similarity Weighted (无参数/少参数基线)
    "sim_weight": {
        "corrector_arch": "similarity_weighted",
        "temperature": 1.0       
    },
    "std_tf": {
        "corrector_arch": "standard_transformer",
        "hidden_dim": 128,       
        "num_heads": 8,          
        "num_layers": 4,         
        "dim_feedforward": 128,  
        "dropout": 0.3,
        "activation": "gelu"
    },
    # 3. Weighted Baseline (BSA1 - 无训练)
    "weighted_base": {
        "corrector_arch": "BSA1", 
        "lambda_weight": 0.5,     
        "temperature": 1.0
    },

    # 4. Learnable Weighted (可学习权重的加权)
    "learnable_weight": {
        "corrector_arch": "learnable_weighted",
        "temperature_init": 1.0,
        "lambda_init": 0.5
    },

    # 5. LightWeight Meta Corrector (BSA2 - 轻量级元学习)
    "meta_corrector": {
        "corrector_arch": "BSA2", 
        "hidden_dim": 64,         
        "temperature": 1.0
    },
    "dual_fusion_V2": {
    "corrector_arch": "dual_source_fusion",
        "d_model": 256,      # 减半：从 128 -> 64
        "n_heads": 32,       # 保持 head_dim = 16 (64/4=16)
        "n_layers": 1,      # 减半：对于残差修正，往往 1 层就够了
        "dropout": 0.8,     # 恢复正常：模型小了，就不需要扔掉 80% 了
        "use_rank_encoding": True,
        'use_vibe_features': False,
        'use_learnable_gate': True
    },

    "dual_fusion_large": {
        "corrector_arch": "dual_source_fusion",
        "d_model": 128,      # 减半：从 128 -> 64
        "n_heads": 32,       # 保持 head_dim = 16 (64/4=16)
        "n_layers": 1,      # 减半：对于残差修正，往往 1 层就够了
        "dropout": 0.8,     # 恢复正常：模型小了，就不需要扔掉 80% 了
        "use_rank_encoding": True,
        'use_vibe_features': False,
        'use_learnable_gate': True
    },
    "dual_set_mlp_V2": {
    "corrector_arch": "dual_set_mlp", # 路由到 DualSourceSetMLPCorrector
    "hidden_dim": 128,
    "dropout": 0.3,       # 大 Dropout 防止过拟合
    "use_vibe_features": False,  # 开启物理截断
    # "gating_strategy": "learnable" # 开启最新的门控机制
},
    "dual_set_mlp": {
    "corrector_arch": "dual_set_mlp", # 路由到 DualSourceSetMLPCorrector
    "hidden_dim": 256,
    "dropout": 0.6,       # 大 Dropout 防止过拟合
    "use_vibe_features": False,  # 开启物理截断
    # "gating_strategy": "learnable" # 开启最新的门控机制
},
    "dual_frozen_set_mlp": {
    "corrector_arch": "dual_frozen_set_mlp", # 对应 MODEL_MAP 中的 Key
    "hidden_dim": 64,                         # 编码器和预测头的隐藏层维度
    "dropout": 0.1,                            # 仅作用于可学习的预测头
    # "embed_dim": 128,                        # 保持与 Encoder 输出维度一致
    'use_vibe_features': False,
},
    "dual_gated_mlp": {
        "corrector_arch": "dual_gated_mlp", # ⚠️ 对应 DualSourceGatedMLPCorrector
        
        # --- 核心维度 ---
        # "embed_dim": 112,       # [注意] 如果用 advanced_hybrid_math 必须是 112，否则是 128
        "hidden_dim": 40,       # 给门控网络留一点计算容量 (比纯 MLP 的 32 略大)
        
        # --- 正则化 ---
        "dropout": 0.2,         # 门控能主动关闭噪声，所以 Dropout 不用激进到 0.8，0.5 足够稳健
        
        # --- 检索参数 ---
        # "top_k": 40,            # 门控可以过滤掉不靠谱的邻居，所以 K 可以比纯 MLP (20) 稍多一点
        "zcr_threshold": 0.05     # 0.35 是一个经验值，意味着"每10个点里只有不到3.5次符号反转"才允许修正
        # --- 其他 ---
        # "activation": "gelu",
        # "pred_len": 96
    },
    "dual_res_mlp": {
    "corrector_arch": "dual_res_mlp",
    "context_len": 512, # 必须与数据一致
    "hidden_dim": 128, #+32
    "dropout": 0.2
    },
    "mean_retrieval": {
        "corrector_arch": "mean_retrieval",
        # 该模型无参数，主要依赖检索回来的邻居质量
        # 建议使用较大的 top_k 来利用大数定律去噪
        "top_k": 50 
    },

    # === 2. 全局偏差基线 (验证误差的系统性) ===
    "global_bias": {
        "corrector_arch": "global_bias",
        "pred_len": 96,
        # 这是一个非常简单的单参数模型，只有 pred_len 个参数
        # 建议使用较大的学习率，让它迅速收敛到全局平均偏差
        "learning_rate": 1e-2, 
        "weight_decay": 0.0
    },

    # === 3. 局部自回归基线 (验证误差的惯性) ===
    "local_ar": {
        "corrector_arch": "local_ar",
        # [关键] 必须与 dataset 的 context_len (通常是96) 严格一致
        # 因为它是一个 Linear(context_len, pred_len)
        "context_len": 96, 
        "pred_len": 96,
        # 这是一个简单的线性回归，标准学习率即可
        "learning_rate": 1e-3,
        "weight_decay": 0
    },
    "semantic_router": {
        "corrector_arch": "semantic_router", # 对应下方的 MODEL_MAP
        "d_model": 128,        # Q/K 投影层以及微调层的隐藏维度
        "dropout": 0.4         # 仅用于微调层的轻量级 Dropout
        # embed_dim 和 pred_len 通常在工厂函数中会动态注入，无需在此硬编码
    },
    "intra_inter_router": {
        "corrector_arch": "intra_inter_router",
        "d_model": 128,        # 内部投影和隐藏层维度
        "dropout": 0.4         # 轻量级 Dropout
    },
    "dual_latent_cross_attn": {
        "corrector_arch": "dual_latent_cross_attn", # 必须和上面的 MODEL_MAP 键一致
        "d_model": 256,         # 隐空间维度 (建议256或512，容量越大生成能力越强)
        "n_heads": 8,           # 交叉注意力的多头数量
        "dropout": 0.4          # 防止过拟合的正则化
        # embed_dim 和 pred_len 会在代码中根据数据集自动注入，这里无需硬编码
    },

}

# =========================================================
# 3. 训练与数据配置 (Training & Data)
# =========================================================
TRAIN_CONFIGS = {
    # 标准全量训练
    "standard": {
        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 50,
        "early_stop_patience": 150,
        "top_k": 30,
        "diversity_max_per_dataset": 10,  
        "retrieval_scope": "cross_dataset", 
        "optimizer": "adam",
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "max_samples_per_dataset": 200, 
        "max_test_samples_per_dataset": 20,
        "pseudo_ratio": 0,
        "pseudo_strength": 0
    },
    "standard2": {
        "lr": 1e-4,
        "batch_size": 64,
        "epochs": 50,
        "early_stop_patience": 150,
        "top_k": 30,
        "diversity_max_per_dataset": 30,  
        "retrieval_scope": "cross_dataset", 
        "optimizer": "adam",
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "max_samples_per_dataset": 200, 
        "max_test_samples_per_dataset": 20,
        "pseudo_ratio": 0,
        "pseudo_strength": 0
    },
     "standard3": {
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 50,
        "early_stop_patience": 150,
        "top_k": 30,
        "diversity_max_per_dataset": 2,  
        "retrieval_scope": "cross_dataset", 
        "optimizer": "adam",
        "weight_decay": 1e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "max_samples_per_dataset": 200, 
        "max_test_samples_per_dataset": 20,
        "pseudo_ratio": 0,
        "pseudo_strength": 0
    },
    # 快速调试 (跑通流程用)
    "debug": {
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 2,
        "early_stop_patience": 1,
        "top_k": 5,
        "diversity_max_per_dataset": 2,
        "max_samples_per_dataset": 1,
        "max_test_samples_per_dataset": 1,
        "debug": True
    },
    
    # 激进训练
    "aggressive": {
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 50,
        "early_stop_patience": 8,
        "scheduler": "plateau",
        "max_samples_per_dataset": 5000
    }
}