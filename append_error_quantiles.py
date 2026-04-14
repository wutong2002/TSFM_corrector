import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from scipy.stats import rankdata

# =========================================================================
# 1. 配置区域
# =========================================================================
CONFIG = {
    # 指向您现有的数据集根目录
    "DATA_ROOT": "correction_datasets_double_res", 
}

# =========================================================================
# 2. 核心计算函数
# =========================================================================
def calculate_smape(truth, pred):
    """计算单个样本的 sMAPE (0 ~ 200)"""
    t = np.array(truth, dtype=np.float32).reshape(-1)
    p = np.array(pred, dtype=np.float32).reshape(-1)
    
    min_len = min(len(t), len(p))
    t = t[:min_len]
    p = p[:min_len]
    
    denom = np.abs(t) + np.abs(p)
    mask = denom > 1e-5
    
    if not np.any(mask): return 0.0
    return float(np.mean(200.0 * np.abs(t[mask] - p[mask]) / denom[mask]))

def calculate_mase(truth, pred, history):
    """计算单个样本的 MASE"""
    t = np.array(truth, dtype=np.float32).reshape(-1)
    p = np.array(pred, dtype=np.float32).reshape(-1)
    h = np.array(history, dtype=np.float32).reshape(-1)
    
    min_len = min(len(t), len(p))
    t = t[:min_len]
    p = p[:min_len]
    
    mae = np.mean(np.abs(t - p))
    
    if len(h) > 1:
        naive_diff = np.abs(h[1:] - h[:-1])
        naive_denom = np.mean(naive_diff)
    else:
        naive_denom = 0.0
        
    if naive_denom > 1e-5: return float(mae / naive_denom)
    else: return 0.0

# =========================================================================
# 3. 分组处理逻辑 (Two-Pass Architecture)
# =========================================================================
def process_group(group_key, file_paths):
    """
    处理特定组别 (如: moirai_small -> Train)
    通过两阶段实现内存安全和全局分位数计算
    """
    tsfm_name, split_type = group_key
    group_label = f"[{tsfm_name} | {split_type}]"
    
    # --- Pass 1: 扫描并计算所有原始误差 ---
    file_metrics = [] 
    all_smapes = []
    all_mases = []
    
    for path in tqdm(file_paths, desc=f"Gathering {group_label}", leave=False):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except: continue
            
        truths = data.get("truths", [])
        preds = data.get("preds", [])
        histories = data.get("histories", [])
        
        n = len(truths)
        if n == 0 or len(preds) != n or len(histories) != n: continue
            
        smapes, mases = [], []
        for t, p, h in zip(truths, preds, histories):
            smapes.append(calculate_smape(t, p))
            mases.append(calculate_mase(t, p, h))
            
        file_metrics.append({
            'path': path,
            'n': n,
            'smapes': smapes,
            'mases': mases
        })
        all_smapes.extend(smapes)
        all_mases.extend(mases)
        
    n_total = len(all_smapes)
    if n_total == 0: return 0
    
    # --- 计算全局百分位排名 (0.0 ~ 100.0) ---
    # rankdata 得到名次，减 1 后除以最大名次得到比例
    global_smapes = np.array(all_smapes)
    global_mases = np.array(all_mases)
    
    sq_ranks = rankdata(global_smapes, method='average')
    mq_ranks = rankdata(global_mases, method='average')
    
    # 防止除以 0
    denom = max(1.0, float(n_total - 1))
    sq_quantiles = (sq_ranks - 1.0) / denom * 100.0
    mq_quantiles = (mq_ranks - 1.0) / denom * 100.0
    
    # --- Pass 2: 切片并注入回原文件 ---
    current_idx = 0
    for fm in tqdm(file_metrics, desc=f"Injecting {group_label}", leave=False):
        n = fm['n']
        p = fm['path']
        
        # 从全局结果中截取属于当前文件的分位数段
        f_sq = sq_quantiles[current_idx : current_idx + n]
        f_mq = mq_quantiles[current_idx : current_idx + n]
        
        with open(p, 'rb') as f:
            data = pickle.load(f)
            
        # 物理注入
        data["smapes"] = np.array(fm['smapes'], dtype=np.float32)
        data["mases"] = np.array(fm['mases'], dtype=np.float32)
        data["smape_quantiles"] = np.array(f_sq, dtype=np.float32)
        data["mase_quantiles"] = np.array(f_mq, dtype=np.float32)
        
        with open(p, 'wb') as f:
            pickle.dump(data, f)
            
        current_idx += n
        
    return n_total

# =========================================================================
# 4. 主流程
# =========================================================================
def main():
    root_path = Path(CONFIG["DATA_ROOT"])
    if not root_path.exists():
        print(f"❌ 找不到数据根目录: {root_path}")
        return
        
    print(f"🚀 开始扫描并注入基于 [TSFM & 训练/测试集] 分组的全局分位数")
    print(f"📂 目标目录: {root_path}")
    print("=" * 70)
    
    pkl_files = list(root_path.rglob("*.pkl"))
    print(f"🔍 共发现 {len(pkl_files)} 个数据集文件。开始建立分组树...")
    
    # 将文件分组：字典的 Key 为 (tsfm_name, split_type)
    groups = defaultdict(list)
    
    for p in pkl_files:
        # 提取 TSFM 名称 (基于相对路径的第一级目录名，如 'moirai_small')
        rel_path = p.relative_to(root_path)
        tsfm_name = rel_path.parts[0] 
        
        # 提取数据划分 (Train / Test)
        filename = p.name.lower()
        if filename.startswith("ge_"):
            split_type = "Test"
        elif filename.startswith("lotsa_"):
            split_type = "Train"
        else:
            split_type = "Unknown"
            
        groups[(tsfm_name, split_type)].append(p)
        
    print(f"📁 成功划分为 {len(groups)} 个独立评估组。")
    print("=" * 70)
    
    total_samples_processed = 0
    
    # 逐组处理
    for group_key, file_paths in groups.items():
        tsfm_name, split_type = group_key
        print(f"▶️ 开始处理组别: 模型 = {tsfm_name} | 集合 = {split_type} ({len(file_paths)} 个文件)")
        
        n_processed = process_group(group_key, file_paths)
        total_samples_processed += n_processed
        
        print(f"   ✅ 完成! 本组共对齐并注入了 {n_processed} 条样本。\n")
        
    print("=" * 70)
    print("🎉 全局注入任务完成！")
    print(f"📉 共计处理并对齐了: {total_samples_processed} 条样本序列。")

if __name__ == "__main__":
    main()