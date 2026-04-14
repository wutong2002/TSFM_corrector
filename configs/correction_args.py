import argparse
import os

def add_correction_args(parser: argparse.ArgumentParser):
    """
    向现有的 parser 中添加残差校正相关的超参数。
    分为四个 Group，方便查看帮助信息。
    """
    
    # ===============================================================
    # 1. 基础实验与数据参数 (General & Data)
    # ===============================================================
    group_base = parser.add_argument_group("Residual Correction - General")
    
    group_base.add_argument("--correction_output_dir", type=str, default="results/corrector_checkpoints",
                            help="校正器模型保存路径")
    group_base.add_argument("--db_source_path", type=str, default="results/cl_original/correction_data",
                            help="用于构建学件库的 .pkl 数据源目录")
    group_base.add_argument("--corrector_load_path", type=str, default=None,
                            help="推理时加载的校正器权重路径 (.pth)")
    group_base.add_argument("--allow_missing_values", type=int, default=1,
                            help="是否允许使用包含缺失值(NaN)的原始样本? 1=允许(填充), 0=禁止(丢弃)")
    
    # [新增] 训练集平衡与加速参数
    group_base.add_argument("--max_samples_per_dataset", type=int, default=-1,
                            help="[入库约束] 构造训练集时，每个数据集最多保留多少样本 (-1=不限制). 用于平衡数据分布和加速调试.")
    
    # [新增] 测试集样本数限制
    group_base.add_argument("--max_test_samples_per_dataset", type=int, default=-1,
                            help="[入库约束] 构造测试集时，每个数据集最多保留多少样本 (-1=不限制). 用于加速验证和调试.")
    group_base.add_argument("--dataset_properties_path", type=str, 
                            default=os.path.join("Datasets", "processed_datasets", "dataset_properties.json"),
                            help="包含数据集 domain 和 frequency 信息的 JSON 文件路径")
    # 基础训练超参
    group_base.add_argument("--lr", type=float, default=1e-3, help="校正器训练学习率")
    group_base.add_argument("--epochs", type=int, default=20, help="训练轮数")
    group_base.add_argument("--corrector_bs", type=int, default=32, help="校正器训练/推理的 Batch Size")
    group_base.add_argument("--patience", type=int, default=5, help="Early Stopping 耐心轮数 (旧参数兼容)")

    # 数据集分组控制
    group_base.add_argument("--train_group", type=str, default="lotsa_train_clean", 
                            help="训练集组名 (来自 dataset_config.py)")
    group_base.add_argument("--test_group", type=str, default="ge_test", 
                            help="测试集组名 (来自 dataset_config.py)")
    group_base.add_argument("--train_test_split_mode", type=str, default="cross_dataset",
                        choices=["cross_dataset", "seq_per_dataset", "temporal_per_seq"],
                        help="数据划分模式: "
                             "cross_dataset(默认, 子集完全隔离); "
                             "seq_per_dataset(同子集但序列隔离); "
                             "temporal_per_seq(同序列但时间隔离)")
    # ===============================================================
    # 2. 检索与学件库参数 (Retrieval & Schoolware)
    # ===============================================================
    group_retrieval = parser.add_argument_group("Residual Correction - Retrieval")
    
    # [修复] 统一在此处定义 encoder_type，包含所有选项
    group_retrieval.add_argument("--encoder_type", type=str, default="statistical", 
                                 choices=["statistical", "units", "timesfm"],
                                 help="选择用于生成 Key 的编码器类型")

    # [新增] 将 timesfm_ckpt 移至此处
    group_retrieval.add_argument("--timesfm_ckpt", type=str, default="google/timesfm-1.0-200m",
                                 help="TimesFM 模型的 HuggingFace 路径")
    
    group_retrieval.add_argument("--retriever_type", type=str, default="exact_cosine",
                                 choices=["exact_cosine", "faiss_l2", "faiss_ivf"],
                                 help="向量检索算法后端")
                                 
    # 👇 [新增] 误差指纹检索度量机制
    group_retrieval.add_argument("--err_sim_metric", type=str, default="l2",
                                 choices=["l2", "cosine"],
                                 help="误差指纹检索空间的度量机制 (l2=欧氏距离, cosine=余弦相似度)")
    
    group_retrieval.add_argument("--top_k", type=int, default=5, 
                                 help="检索最相似的历史片段数量")
    
    # [修复] 将 diversity 参数整合至此，删除重复定义
    group_retrieval.add_argument("--diversity_max_per_dataset", type=int, default=2,
                                 help="[检索约束] 单次检索K个邻居时，允许来自同一数据集的最大数量 (防止Context单调)")

    group_retrieval.add_argument("--enforce_cross_time_restriction", type=int, default=1,
                                 help="[Causal模式] 是否强制时间因果限制? 1=是, 0=否")
    
    group_retrieval.add_argument("--retrieval_scope", type=str, default="global",
                                 choices=["global", "same_dataset", "causal", "cross_dataset"],
                                 help="检索范围策略")
    # [新增] 检索过滤控制开关
    group_retrieval.add_argument("--filter_by_freq", type=int, default=0, choices=[0, 1],
                                 help="[检索约束] 是否只检索【相同频率】的邻居? 1=是, 0=否")
    
    group_retrieval.add_argument("--filter_by_domain", type=int, default=0, choices=[0, 1],
                                 help="[检索约束] 是否只检索【相同领域】的邻居? 1=是, 0=否")
    group_retrieval.add_argument("--retrieval_scope", type=str, default="cross_dataset",
                             choices=["allow_self", "exclude_self", "exclude_seq", "cross_dataset"],
                             help="检索范围掩码策略: "
                                  "allow_self(允许查到目标本身); "
                                  "exclude_self(禁止查到目标本身); "
                                  "exclude_seq(禁止查到目标所在序列的任何窗口); "
                                  "cross_dataset(禁止查到目标所在子集)")
    # ===============================================================
    # 3. 校正器模型结构参数 (Corrector Model Architecture)
    # ===============================================================
    group_model = parser.add_argument_group("Residual Correction - Model Structure")
    
    group_model.add_argument("--corrector_arch", type=str, default="attention",
                             choices=["attention", "linear", "mlp", "rnn", "zero", "deep_transformer", "standard_transformer","similarity_weighted","BSA1","BSA2","learnable_weighted"],
                             help="校正器网络架构类型")
    # 👇 [新增] 振动特征与物理截断开关
    group_model.add_argument("--use_vibe_features", type=int, default=1, choices=[0, 1],
                             help="是否对模型输入振动特征并开启物理包络截断 (1=开启, 0=关闭)")
    group_model.add_argument("--embed_dim", type=int, default=64,
                             help="Encoder 输出的向量维度")
    
    group_model.add_argument("--hidden_dim", type=int, default=128,
                             help="校正器内部隐层维度")
    
    group_model.add_argument("--num_heads", type=int, default=8,
                             help="[Attention/Transformer] 多头注意力的头数")
    
    group_model.add_argument("--num_layers", type=int, default=8,
                             help="[DeepTransformer] Transformer 堆叠层数")
    
    group_model.add_argument("--dim_feedforward", type=str, default=None,
                             help="[DeepTransformer] FFN 中间层维度")
    
    group_model.add_argument("--dropout", type=float, default=0.3,
                             help="Dropout 比率")
    
    group_model.add_argument("--temperature", type=float, default=1.0,
                             help="[Linear Only] Softmax 温度系数")
    group_model.add_argument("--gating_strategy", type=str, default="none",
                             choices=["none", "learnable", "static", "scan_100"],
                             help="门控拦截策略: "
                                  "none(无门控, 全量修正); "
                                  "learnable(端到端可学习 Critic 门控); "
                                  "static(基于静态残差阈值的硬门控); "
                                  "scan_100(评估时瞬间扫描 1~100 阈值最佳性能)")
    
    group_model.add_argument("--static_threshold", type=float, default=0.15,
                             help="静态门控的触发阈值 (仅在 gating_strategy=static 时生效)")
                             
    group_model.add_argument("--gate_loss_weight", type=float, default=0.5,
                             help="Critic Gate 的 BCE Loss 权重 (仅在 gating_strategy=learnable 时生效)")
    # ===============================================================
    # 4. 训练技巧参数 (Training Techniques)
    # ===============================================================
    group_train = parser.add_argument_group("Training Techniques")
    
    group_train.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"],
                             help="优化器类型")
    
    group_train.add_argument("--momentum", type=float, default=0.9,
                             help="[SGD] 动量系数")
    
    group_train.add_argument("--nesterov", action="store_true",
                             help="[SGD] 是否使用Nesterov加速")
    
    group_train.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"],
                             help="学习率调度器")
    
    group_train.add_argument("--early_stop_patience", type=int, default=10,
                             help="早停耐心值 (Epochs)")
    
    group_train.add_argument("--weight_decay", type=float, default=1e-4,
                             help="权重衰减系数")
    group_train.add_argument("--shuffle_retrieved_order", type=bool, default=False,
                             help="是否在训练时加载样本上打乱辅助序列顺序")
    group_train.add_argument("--pseudo_ratio", type=float, default=0.0,
                                help="伪样本比例 (0.0 表示不使用伪样本)")
    group_train.add_argument("--pseudo_strength", type=float, default=0.8,
                                help="伪样本强度 (仅当 pseudo_ratio > 0 时生效)")
   # 👇 [修改] 困难样本过滤参数
    group_train.add_argument("--hard_quantile_train", type=float, default=100.0,
                             help="训练集保留最困难的前 x% 样本 (如 20 表示仅训练误差最大的20%数据。100=全量)")
    
    group_train.add_argument("--hard_quantile_test", nargs='+', type=float, default=[100.0],
                             help="测试集评估分位数列表 (例如 100.0 50.0 20.0)。列表中【最大值】(覆盖最广的集合) 将用于决定最佳Epoch和早停。")
    # 👇 [新增] 损失函数类型
    group_train.add_argument("--loss_type", type=str, default="hybrid", 
                             choices=["hybrid", "huber", "mse", "l1"],
                             help="训练使用的损失函数类型")
    # 控制训练集验证的模式：'none' (不验证直接输出0), 'sample' (随机采样验证), 'full' (全量验证)
    group_train.add_argument("--train_eval_mode", type=str, default='sample', choices=['none', 'sample', 'full'], 
                             help="训练集验证的模式。")
    group_train.add_argument("--train_eval_samples", type=int, default=1000, 
                             help="当 train_eval_mode='sample' 时，随机采样的样本数量。")
    
    # 👇 [新增] 训练集评估展示开关
    group_debug.add_argument("--show_train_metrics", type=int, default=0, choices=[0, 1], 
                             help="每个 epoch 是否计算并展示训练集上的 sMAPE 增益 (1=展示, 0=关闭)")
    # 注意：删除了原来末尾重复的 group_rag，相关参数已移至 group_retrieval
    
    # ===============================================================
    # 5. 数据检查参数 (Data Checking)
    # ===============================================================
    group_check = parser.add_argument_group("Residual Correction - Data Checking")
    group_check.add_argument("--enable_data_check", type=int, default=0, choices=[0, 1], help="是否启用数据检查 (0=关闭, 1=启用)")
    group_check.add_argument("--check_level", type=str, default="basic", choices=["basic", "full"], help="数据检查级别")
    
    # ===============================================================
    # 6. Debug 参数
    # ===============================================================
    group_debug = parser.add_argument_group("Residual Correction - Debug")
    group_debug.add_argument("--debug", type=int, default=1, choices=[0, 1], help="是否启用调试模式 (0=关闭, 1=启用)")
    # 👇 [新增] 数据泄露模拟开关
    group_debug.add_argument("--allow_data_leakage", type=int, default=0, choices=[0, 1], 
                             help="[模拟实验] 是否允许数据泄露(将测试集同时放入训练检索库)? 1=允许, 0=禁止")
    return parser