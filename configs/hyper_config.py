# configs/hyper_config.py

# =========================================================
# 🔍 超参数搜索空间 (Search Space)
# =========================================================

# 定义不同的搜索配置，可以在脚本中通过 --search_conf 指定
HYPER_SEARCH_SPACE = {
    # 1. 针对数据与检索的搜索 (Data & Retrieval Focused)
    "retrieval_search": {
        "top_k": [10, 50, 100],               # 检索多少个邻居
        "diversity_max_per_dataset": [2, 10, 50], # 每个数据集最多检索多少个
        "pseudo_ratio": [0.0],           # 伪样本比例
        "batch_size": [64],                   # 固定
        "lr": [1e-3, 1e-4]                          # 固定
    },

    # 2. 针对优化器的搜索 (Optimization Focused)
    "optim_search": {
        "lr": [1e-3, 1e-4, 5e-5],
        "weight_decay": [1e-4, 1e-3],
        "batch_size": [32, 64, 128],
        "top_k": [50],                        # 固定
        "diversity_max_per_dataset": [10]     # 固定
    },

    # 3. 全面大搜索 (Full Grid - 慎用，组合数会很多)
    "full_grid": {
        "top_k": [5, 30, 100],
        "diversity_max_per_dataset": [2, 10, 50],
        "batch_size": [32, 64],
        "lr": [1e-3, 1e-4, 5e-5],
        "pseudo_ratio": [0.0],
        "retrieval_scope": ["cross_dataset"] # 也可以搜索检索范围
    }
}

# 基础训练配置 (作为默认值，未在网格中指定的参数将使用此处的默认值)
BASE_TRAIN_CONFIG = {
    "epochs": 30,
    "early_stop_patience": 20,
    "optimizer": "adam",
    "scheduler": "cosine",
    "warmup_epochs": 2,
    # 默认数据限制，加快搜索速度
    "max_samples_per_dataset": 100, 
    "max_test_samples_per_dataset": 20,
}