import os
import pandas as pd
from datasets import Dataset, Features, Value, Sequence

# ================= 1. 配置区域 =================
# 定义基础工程目录
BASE_DIR = r"D:\Res-context\TSFM_Lib-main\Datasets"

# 原始 .parquet 文件存放的目录
RAW_QB_DIR = os.path.join(BASE_DIR, "raw_data", "Quito_Bench")

# 拆分处理后的数据集输出目录 (与 raw_data 同级)
OUTPUT_ROOT = RAW_QB_DIR

# 保持与你原有预处理脚本完全一致的特征定义
UNIVARIATE_FEATURES = Features({
    "item_id": Value("string"),
    "start": Value("timestamp[s]"),
    "freq": Value("string"),
    "target": Sequence(Value("float32")), 
})

# ================= 2. 核心处理逻辑 =================

def parse_timestamp(start_raw):
    """复用你的时间戳解析逻辑"""
    try:
        if pd.isna(start_raw):
            return int(pd.Timestamp("2000-01-01").timestamp())
        ts = pd.Timestamp(str(start_raw)).timestamp()
        return int(ts)
    except:
        return int(pd.Timestamp("2000-01-01").timestamp())

def process_quitobench_parquet(parquet_path, output_root, dataset_prefix, freq_label):
    """
    处理 QuitoBench 的 parquet 文件，转换为你的 HuggingFace Dataset 单变量格式。
    """
    if not os.path.exists(parquet_path):
        print(f"❌ 找不到原始文件: {parquet_path}")
        return

    print(f"\n📦 正在加载 QuitoBench 数据: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # 确保同一个序列的数据按时间绝对顺序排列
    df = df.sort_values(by=["item_id", "date_time"])
    
    # 按照独立的序列 ID 进行分组
    grouped = df.groupby("item_id")
    
    # 原数据提供的 5 个多变量通道
    channels = ["ind_1", "ind_2", "ind_3", "ind_4", "ind_5"]
    
    # 按照“单一序列模式”将每个通道的数据拆分为独立的数据集
    for ch_idx, col in enumerate(channels):
        # 命名规范对齐你的拆分逻辑 (例如: QB_min_1)
        dataset_name = f"QB_{dataset_prefix}_{ch_idx + 1}"
        save_path = os.path.join(output_root, dataset_name)
        
        # 避免重复处理
        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"⏩ {dataset_name} 已存在于 processed_datasets 中，跳过。")
            continue
            
        print(f"  🔨 正在处理通道 {col} -> 组装 {dataset_name}...")
        rows = []
        
        for item_id, group in grouped:
            # 提取时间戳
            start_raw = group["date_time"].iloc[0]
            start_ts = parse_timestamp(start_raw)
            
            # 提取目标通道序列，并将原生的 NaN 填充为 0.0
            target_vals = group[col].fillna(0.0).tolist()
            
            rows.append({
                "item_id": str(item_id),
                "start": start_ts,
                "freq": freq_label,
                "target": target_vals
            })
            
        # 构建 HuggingFace Dataset 字典
        data_dict = {
            "item_id": [r["item_id"] for r in rows],
            "start": [r["start"] for r in rows],
            "freq": [r["freq"] for r in rows],
            "target": [r["target"] for r in rows]
        }
        
        # 转换为 Dataset 对象并序列化保存到磁盘
        ds = Dataset.from_dict(data_dict, features=UNIVARIATE_FEATURES)
        ds.save_to_disk(save_path)
        print(f"    💾 保存至: {save_path} (序列总数: {len(ds)})")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 1. 处理分钟级数据 (源频: 10分钟 -> freq_label: '10T')
    min_file = os.path.join(RAW_QB_DIR, "test_min-00001-of-00001.parquet")
    process_quitobench_parquet(
        parquet_path=min_file, 
        output_root=OUTPUT_ROOT, 
        dataset_prefix="min", 
        freq_label="10T"
    )
        
    # 2. 处理小时级数据 (源频: 1小时 -> freq_label: '1H')
    hour_file = os.path.join(RAW_QB_DIR, "test_hour-00001-of-00001.parquet")
    process_quitobench_parquet(
        parquet_path=hour_file, 
        output_root=OUTPUT_ROOT, 
        dataset_prefix="hour", 
        freq_label="1H"
    )
        
    print("\n🎉 QuitoBench 数据处理完成！所有处理后的数据集已存入 processed_datasets 目录。")