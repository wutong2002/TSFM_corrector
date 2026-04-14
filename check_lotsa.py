import os
import numpy as np
from datasets import load_from_disk
import pandas as pd

# ================= 配置区域 =================
# 请修改为您实际的路径
PATH_GE_ORIGIN = "original_datasets/bizitobs_application"  # 原始 GE 数据路径
PATH_LOTSA_NEW = "processed_datasets/LOTSA_PEMS03"         # 加工后的 LOTSA 路径

def inspect_dataset(path, name):
    print(f"\n{'='*20} 正在检查: {name} {'='*20}")
    
    # 1. 检查路径是否存在
    if not os.path.exists(path):
        print(f"❌ 路径不存在: {path}")
        return None
    
    try:
        # 2. 加载数据集
        ds = load_from_disk(path)
        print(f"✅ 加载成功! 样本总数: {len(ds)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    # 3. 获取第一个样本进行深度解剖
    sample = ds[0]
    info = {}
    
    # --- 检查 Start ---
    start_val = sample.get('start')
    info['start_type'] = type(start_val).__name__
    print(f"  [Start] 类型: {info['start_type']} | 值示例: {start_val}")
    
    # --- 检查 Freq ---
    freq_val = sample.get('freq')
    info['freq_val'] = freq_val
    print(f"  [Freq]  值: {freq_val}")
    
    # --- 检查 Target (关键!) ---
    target = np.array(sample.get('target', []))
    info['target_shape'] = target.shape
    info['target_dtype'] = target.dtype
    
    # 检查是否包含 NaN 或 Inf
    has_nan = np.isnan(target).any()
    print(f"  [Target] Shape: {target.shape} (期望: (2, T)) | Dtype: {target.dtype} | Has NaN: {has_nan}")
    if target.shape[0] == 2:
        print(f"      -> Channel 0 (Data) Mean: {np.mean(target[0]):.4f}")
        print(f"      -> Channel 1 (Aux)  Mean: {np.mean(target[1]):.4f} (LOTSA应为0)")
    else:
        print(f"      ❌ 警告: Target 通道数不是 2!")

    # --- 检查 Features (关键!) ---
    feat = np.array(sample.get('past_feat_dynamic_real', []))
    info['feat_shape'] = feat.shape
    print(f"  [Feat]   Shape: {feat.shape} (期望: (35, T))")
    
    if feat.shape[0] == 35:
        print("      ✅ 特征维度正确 (35)")
    else:
        print(f"      ❌ 警告: 特征维度不匹配!")
        
    # --- 检查 Item ID ---
    info['item_id_type'] = type(sample.get('item_id')).__name__
    
    return info

def compare_results(ge_info, lotsa_info):
    print(f"\n{'='*20} 最终对比结果 {'='*20}")
    
    if not ge_info or not lotsa_info:
        print("❌ 无法对比，因为其中一个数据集加载失败。")
        return

    success = True
    
    # 1. 对比 Target 维度 (Channel 数)
    ge_channels = ge_info['target_shape'][0]
    lotsa_channels = lotsa_info['target_shape'][0]
    
    if ge_channels == lotsa_channels:
        print(f"✅ Target 通道一致: {ge_channels}")
    else:
        print(f"❌ Target 通道不一致! GE={ge_channels}, LOTSA={lotsa_channels}")
        success = False
        
    # 2. 对比 Feature 维度 (Channel 数)
    # 注意: 如果 GE 为空或不同，这里会报错，我们需要健壮性
    ge_feat_channels = ge_info['feat_shape'][0] if len(ge_info['feat_shape']) > 0 else 0
    lotsa_feat_channels = lotsa_info['feat_shape'][0] if len(lotsa_info['feat_shape']) > 0 else 0
    
    if ge_feat_channels == lotsa_feat_channels:
        print(f"✅ Feature 通道一致: {ge_feat_channels}")
    else:
        print(f"❌ Feature 通道不一致! GE={ge_feat_channels}, LOTSA={lotsa_feat_channels}")
        success = False

    # 3. 对比 Start 类型
    # timestamp[s] 在 python 中可能表现为 int/float 或 datetime，取决于 datasets 版本
    # 我们只要确保它们能互相兼容即可
    print(f"ℹ️  Start 类型: GE={ge_info['start_type']}, LOTSA={lotsa_info['start_type']}")

    if success:
        print("\n🎉 结论: 格式完全兼容！LOTSA 数据可以欺骗您的代码。")
    else:
        print("\n⚠️ 结论: 存在格式差异，可能会导致代码报错。")

if __name__ == "__main__":
    # 1. 检查原始 GE 数据
    ge_info = inspect_dataset(PATH_GE_ORIGIN, "原始 Gift-Eval (Bizitobs)")
    
    # 2. 检查加工后的 LOTSA 数据
    lotsa_info = inspect_dataset(PATH_LOTSA_NEW, "加工后 LOTSA")
    
    # 3. 执行对比
    compare_results(ge_info, lotsa_info)