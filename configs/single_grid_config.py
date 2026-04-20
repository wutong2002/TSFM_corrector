# configs/single_grid_config.py

# =========================================================
# 🎯 单模型专项搜索空间 (Single TSFM Grid Search Space)
# =========================================================

# 定义不同的搜索模式，可以在脚本中通过 --mode 选择
SEARCH_GRIDS = {
    # 1. 标准搜索：均衡探索检索参数和训练参数
    "standard": {
        "learning_rate": [1e-4],
        "batch_size": [64],
        "top_k": [50],
        "diversity_max_per_dataset": [2, 50],
        "pseudo_ratio": [0.0],
        "weight_decay": [1e-4]
    },

    # 2. 检索重型搜索：重点探索 TopK 和 伪样本
    "retrieval_heavy": {
        "learning_rate": [1e-4],   # 固定 LR
        "batch_size": [32],        # 固定 BS
        "top_k": [10, 30, 100],
        "diversity_max_per_dataset": [5, 20, 50],
        "pseudo_ratio": [0.0, 0.5],
        "weight_decay": [1e-4, 1e-5]
    },

    # 3. 极速调试 (Debug)
    "debug": {
        "learning_rate": [1e-3],
        "batch_size": [16],
        "top_k": [5],
        "diversity_max_per_dataset": [2],
        "pseudo_ratio": [0.0],
        "weight_decay": [1e-4]
    },
    # 3. [新增] Standard Transformer 架构搜索
    "std_tf_opt": {
        # 固定训练参数
        "learning_rate": [1e-4],
        "batch_size": [128],
        "top_k": [50],
        
        # --- std_tf 专属架构参数 ---
        "n_head": [8],      # 注意力头数
        "n_layer": [8],     # Transformer 层数
        "dropout": [0.3],    # Dropout 比率
        "d_ff": [512],       # 前馈网络维度 (通常是 embed_dim * 4)
        "activation": ["gelu"],    # 激活函数
        # "top_k": [10],
        "diversity_max_per_dataset": [50, 10],
        "pseudo_ratio": [0.0],
        "weight_decay": [1e-4]
    },
    "ablation_search": {
        "learning_rate": [1e-4],
        "batch_size": [128],
        "top_k": [50],
        
        # 1. 振动特征与物理截断消融: 0(关) vs 1(开)
        "use_vibe_features": [0],
        
        # 2. 损失函数消融: 对比常规 MSE/L1 与 您的混合频率损失, "l1", "hybrid"
        "loss_type": ["mse"],
        
        # 3. 误差指纹检索度量消融: 对比 L2 距离与余弦相似度"l2", 
        "err_sim_metric": ["cosine"],
        
        "retrieval_alpha": [0.2], 
        "retrieval_beta": [0.5],
        "hard_quantile_train": [100.0],
        "hard_quantile_test": [[1, 10, 50, 100.0]],
    },
    # 4. [新增] 双重指纹专项搜索 (Dual-Source Search)
    # 重点探索 alpha 值对性能的影响
    "dual_mode_search": {
        # 训练参数
        "learning_rate": [1e-2],
        "batch_size": [256],
        "weight_decay": [0],
        
        # 检索参数
        "top_k": [25],
        "diversity_max_per_dataset": [100], # 适当降低多样性，聚焦高质量邻居
        # 1. 振动特征与物理截断消融: 0(关) vs 1(开)
        "use_vibe_features": [0],
        
        # 2. 损失函数消融: 对比常规 MSE/L1 与 您的混合频率损失, "l1", "hybrid"
        "loss_type": ["mse"],
        
        # 3. 误差指纹检索度量消融: 对比 L2 距离与余弦相似度"l2", 
        "err_sim_metric": ["cosine"],
        # [核心] 探索不同的混合比例
        # 1.0 = 仅序列 (Baseline)
        # 0.5 = 混合双指纹 (New Method)
        # 0.0 = 仅误差 (Pure Error Matching)
        "retrieval_alpha": [1.0], 
        "retrieval_beta": [1],
        "filter_by_freq": [0],  # 检索相同频率的邻居
        "filter_by_domain": [0],  # 检索相同领域
        "pseudo_ratio": [0],  # 不使用伪样本，专注于检索质量
        "pseudo_strength": [0] , # 伪样本强度保持不变
        # "context_len": [96],
        # "pred_len": [512],
        "hard_quantile_train": [100.0],  # 仅训练误差最大的20%数据10.0,30.0,80.0,
        # "hard_quantile_test": [[i for i in range(1,101)]],   # 评估模型在困难样本上的专精能力20.0, 10.0, 50.0, 
        "hard_quantile_test": [[100.0]],   # 评估模型在困难样本上的专精能力20.0, 10.0, 50.0,"static" , , "learnable","none"
        "gating_strategy": ["none", "learnable"],  # 开启最新的门控机制
        # "pseudo_method": ["tsfm"],  # 同时测试使用 TSFM 预测的伪样本和随机生成的伪样本

        # 如果你想同时搜索模型参数，可以解开下面注释
        # "d_model": [128, 256]
    },
}


# 默认的基础训练配置 (会被网格参数覆盖)
BASE_CONFIG = {
    "epochs": 200,
    "early_stop_patience": 20,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "warmup_epochs": 5,
    # [关键] 针对单个 TSFM，我们通常希望加载该目录下尽可能多的数据
    # 如果显存不够，可以适当调小
    "max_samples_per_dataset": 2000,  
    "max_test_samples_per_dataset": -1,
    "train_group": "qb_test_all", #"lotsa_train_clean", ge_test_all, qb_test_all, all, 
    "test_group": "qb_test_all",
    # 检索范围：因为我们只加载了单 TSFM 的数据，所以这里设为 cross_dataset 
    # 实际上等价于在当前已加载的所有数据(即单 TSFM)中检索"cross_dataset"
    # "retrieval_scope": "global" ,
    'shuffle_retrieved_order': False,
    "use_vibe_features": 0,      # 默认关闭振动特征
    "loss_type": "mse",       # 默认不使用高频混合损失
    "err_sim_metric": "cosine",      # 默认使用余弦相似度检索误差指纹
    "allow_data_leakage": 1,     # 数据泄漏
    "group_by_parent_item_id": 1,  # 按原始母序列分组切分，避免多通道同源泄漏
    "v2_last_sequence_only": 1,  # 运行实验时按 source(parent) 仅保留最后/目标序列
    "train_test_split_mode":"seq_per_dataset" ,#"cross_dataset(默认, 子集完全隔离); ""seq_per_dataset(同子集但序列隔离); ""temporal_per_seq(同序列但时间隔离)"
    "retrieval_scope":"exclude_seq" ,#"allow_self(允许查到目标本身); ""exclude_self(禁止查到目标本身); ""exclude_seq(禁止查到目标所在序列的任何窗口); ""cross_dataset(禁止查到目标所在子集)"
    # "gating_strategy": "none", 
    "static_threshold": 0.15,
    "gate_loss_weight": 0.9,
    "show_train_metrics": 1, 
    'train_eval_mode': 'sample',
    'train_eval_samples': 1000,
} 
