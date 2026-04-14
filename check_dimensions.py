import numpy as np
from datasets import load_from_disk

# 请替换为您原始 GE 数据集的路径
DATASET_PATH = "Dataset_Path\\bizitobs_l2c\\5T"

def analyze_channel(data, channel_idx):
    """分析某个通道的数据特征"""
    vals = data[channel_idx]
    unique_vals = np.unique(vals)
    
    print(f"  --- [维度 {channel_idx}] 统计 ---")
    print(f"  前10个值: {vals[:10]}")
    print(f"  最小值: {np.min(vals):.4f}")
    print(f"  最大值: {np.max(vals):.4f}")
    print(f"  平均值: {np.mean(vals):.4f}")
    print(f"  唯一值数量: {len(unique_vals)}")
    
    # 判断是否像掩码
    if len(unique_vals) <= 5 and set(unique_vals).issubset({0.0, 1.0}):
        print(f"  👉 结论: 看起来像是【掩码 (Mask)】(只有 0/1)")
        return "MASK"
    else:
        print(f"  👉 结论: 看起来像是【真实数值 (Target)】")
        return "VALUE"

def main():
    if not os.path.exists(DATASET_PATH):
        print("❌ 路径不存在，请检查 path")
        return

    print(f"正在加载: {DATASET_PATH}")
    ds = load_from_disk(DATASET_PATH)
    
    # 随机抽查 3 个样本，避免某个样本正好特殊
    import random
    sample_size = min(len(ds), 3)
    indices = random.sample(range(len(ds)), sample_size)
   
    for i in indices:
        item = ds[i]
        target = np.array(item['target'])
        print(f"\n{'='*20} 样本 ID: {item['item_id']} {'='*20}")
        print(f"Target Shape: {target.shape}")
        
        if target.ndim == 1:
            print("⚠️ 这是一个单变量数据集，没有维度之争。")
            continue
            
        if target.shape[0] != 2:
            print(f"⚠️ 维度不是 2 (是 {target.shape[0]})，情况复杂。")
            continue

        # 分析第 0 维
        type_0 = analyze_channel(target, 0)
        # 分析第 1 维
        type_1 = analyze_channel(target, 1)
        
        # 最终判定
        if type_0 == "VALUE" and type_1 == "MASK":
            print("\n✅ 确认: 维度 0 是目标，维度 1 是掩码。")
        elif type_0 == "MASK" and type_1 == "VALUE":
            print("\n❗ 反转: 维度 1 才是目标！(请修改代码)")
        else:
            print("\n❓ 无法确定: 两个维度看起来都像是数值，或者是多变量预测。")

if __name__ == "__main__":
    import os
    main()