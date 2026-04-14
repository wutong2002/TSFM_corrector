import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# === 引入项目组件 ===
from database.manager import SchoolwareDB
from encoder.statistical import StatisticalEncoder
from encoder.units_encoder import UnitsEncoder 
from encoder.timesfm_encoder import TimesFMEncoder
from encoder.utils import find_best_units_checkpoint 
from utils.missing import fill_missing

# =========================================================
# 2. 数据加载
# =========================================================
def load_data_subset(data_dir, max_samples_per_dataset=50, total_limit=2000):
    """
    加载数据，但为了可视化清晰，限制每个数据集的样本数，
    防止大类主导 t-SNE。
    """
    import pickle
    import glob
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    samples = []
    
    # 计数器，确保类别平衡
    from collections import defaultdict
    counts = defaultdict(int)
    
    for f in tqdm(files, desc="Loading Data"):
        try:
            with open(f, 'rb') as fp: data = pickle.load(fp)
            ds_name = data.get('dataset_name', 'unknown')
            
            if counts[ds_name] >= max_samples_per_dataset: continue
            
            hists = data.get('histories', [])
            
            n = len(hists)
            indices = np.random.permutation(n)
            
            for i in indices:
                if counts[ds_name] >= max_samples_per_dataset: break
                
                h = fill_missing(np.array(hists[i]).reshape(-1))
                # 简单截断对齐
                if len(h) > 512: h = h[-512:]
                elif len(h) < 512: h = np.pad(h, (512-len(h), 0))
                
                samples.append({
                    'history': h.astype(np.float32),
                    'dataset': ds_name
                })
                counts[ds_name] += 1
        except: continue
        
        if len(samples) >= total_limit: break
    
    print(f"✅ Loaded {len(samples)} samples from {len(counts)} datasets.")
    return samples

# =========================================================
# 3. 聚类评估核心逻辑
# =========================================================
def evaluate_clustering(encoder, samples, device='cuda', batch_size=64, plot_name=None):
    name = encoder.__class__.__name__
    print(f"\n📊 Evaluating Clustering: {name} ...")
    
    # 1. 提取所有 Embedding
    histories = torch.from_numpy(np.stack([s['history'] for s in samples]))
    labels = [s['dataset'] for s in samples]
    
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(histories), batch_size), desc="Encoding"):
            batch = histories[i:i+batch_size].to(device)
            # 兼容不同 Encoder 接口
            if hasattr(encoder, 'encode'):
                emb = encoder.encode(batch)
            else:
                emb = torch.zeros(len(batch), 10) # Dummy
                
            all_embs.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(all_embs, axis=0) # (N, D)
    
    # 2. 聚类指标计算
    # 将 dataset_name 转为数字标签
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    n_classes = len(le.classes_)
    
    metrics = {}
    
    # (A) Silhouette Score (轮廓系数)
    # 衡量样本与同类别的相似度 vs 与其他类别的相似度
    # 越高越好 (Max 1.0)
    try:
        sil = silhouette_score(embeddings, y_true, metric='cosine')
        metrics['Silhouette'] = sil
    except: metrics['Silhouette'] = 0.0
    
    # (B) Calinski-Harabasz Index
    # 越高越好
    try:
        ch = calinski_harabasz_score(embeddings, y_true)
        metrics['Calinski-Harabasz'] = ch
    except: metrics['Calinski-Harabasz'] = 0.0
    
    # (C) K-Means Alignment (AMI)
    # 假设我们完全不知道标签，强制聚成 N 类，看聚出来的类和真实标签对不对得上
    try:
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init='auto').fit(embeddings)
        ami = adjusted_mutual_info_score(y_true, kmeans.labels_)
        metrics['AMI (K-Means)'] = ami
    except: metrics['AMI (K-Means)'] = 0.0

    print(f"   📈 Silhouette: {metrics['Silhouette']:.4f}")
    print(f"   📈 AMI: {metrics['AMI (K-Means)']:.4f}")
    
    # 3. t-SNE 可视化
    if plot_name:
        print("   🎨 Generating t-SNE plot...")
        # 降维
        tsne = TSNE(n_components=2, perplexity=min(30, len(samples)-1), random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(embeddings)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        df_plot = pd.DataFrame({
            'x': emb_2d[:, 0], 
            'y': emb_2d[:, 1], 
            'dataset': labels
        })
        
        # 颜色太多会乱，只选 Top 10 大类加颜色，其他灰掉
        top_ds = df_plot['dataset'].value_counts().nlargest(10).index
        sns.scatterplot(
            data=df_plot[df_plot['dataset'].isin(top_ds)],
            x='x', y='y', hue='dataset', 
            palette='tab10', s=60, alpha=0.8
        )
        # 绘制其他的为灰色
        sns.scatterplot(
            data=df_plot[~df_plot['dataset'].isin(top_ds)],
            x='x', y='y', color='grey', s=20, alpha=0.2, label='Others'
        )
        
        plt.title(f"t-SNE of {name} Embeddings\n(Silhouette: {metrics['Silhouette']:.3f})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{plot_name}.png", dpi=150)
        plt.close()
        
    return metrics

# =========================================================
# 4. 主程序
# =========================================================

if __name__ == "__main__":
    import argparse
    
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="Benchmark different encoders for time series embedding")
    parser.add_argument("--data-dir", type=str, default="correction_datasets/chronos_bolt_tiny/cl_original/correction_data",
                        help="Path to the dataset directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")
    parser.add_argument("--encoder", type=str, default="all", choices=["all", "statistical", "units", "timesfm"],
                        help="Which encoder to use (default: all)")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples to use")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context length for encoders")
    args = parser.parse_args()
    
    DATA_DIR = args.data_dir
    DEVICE = args.device
    SELECTED_ENCODER = args.encoder
    MAX_SAMPLES = args.max_samples
    CONTEXT_LEN = args.context_len
    
    print(f"📋 Experiment Configuration:")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Device: {DEVICE}")
    print(f"   Encoder: {SELECTED_ENCODER}")
    print(f"   Max Samples: {MAX_SAMPLES}")
    print(f"   Context Length: {CONTEXT_LEN}")
    
    # 1. 加载数据
    samples = load_data_subset(DATA_DIR, max_samples_per_dataset=50, total_limit=MAX_SAMPLES)
    if not samples: 
        print("❌ No samples loaded, exiting.")
        exit()
    
    results = []

    # --- Test 1: Statistical ---
    if SELECTED_ENCODER in ["all", "statistical"]:
        try:
            enc_stat = StatisticalEncoder(input_len=CONTEXT_LEN)
            m1 = evaluate_clustering(enc_stat, samples, DEVICE, plot_name="viz_statistical")
            m1['Encoder'] = 'Statistical'
            results.append(m1)
        except Exception as e:
            print(f"❌ Statistical encoder failed: {e}")
    
    # --- Test 2: UniTS ---
    if SELECTED_ENCODER in ["all", "units"]:
        try:
            ckpt = find_best_units_checkpoint()
            if ckpt:
                enc_units = UnitsEncoder(ckpt_path=ckpt, context_len=CONTEXT_LEN, device=DEVICE)
                m2 = evaluate_clustering(enc_units, samples, DEVICE, plot_name="viz_units")
                m2['Encoder'] = 'UniTS'
                results.append(m2)
            else:
                print(f"⚠️ No UniTS checkpoint found, skipping.")
        except Exception as e:
            print(f"❌ UniTS encoder failed: {e}")
    
    # --- Test 3: TimesFM ---
    if SELECTED_ENCODER in ["all", "timesfm"]:
        try:
            enc_tfm = TimesFMEncoder(context_len=CONTEXT_LEN, device=DEVICE)
            m3 = evaluate_clustering(enc_tfm, samples, DEVICE, plot_name="viz_timesfm")
            m3['Encoder'] = 'TimesFM'
            results.append(m3)
        except Exception as e:
            print(f"❌ TimesFM encoder failed: {e}")

    # 4. 汇总表格
    if results:
        df_res = pd.DataFrame(results).set_index('Encoder')
        print("\n\n🏆 Clustering Leaderboard:")
        print(df_res)
        
        # 保存结果
        df_res.to_csv("encoder_clustering_benchmark.csv")
        print("✅ Benchmark finished. Check .png files and .csv.")
    else:
        print("❌ No results generated, all encoders failed.")