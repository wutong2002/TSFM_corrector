import pandas as pd
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from Dataset_Path.dataset_config import ALL_Fast_DATASETS
from Model_Path.model_zoo_config import Model_abbrev_map, Model_zoo_details


def check_results_file(csv_file_path,verbose=False):
    """检查结果文件的完整性和一致性"""
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"无法读取CSV文件: {e}")
        return None

    if verbose:
        print(f"\n{'=' * 50}\n检查结果文件: {csv_file_path}")

    # 1. 数据集完整性
    check_dataset_completeness(df, verbose)
    # 2. 去重
    df = check_duplicate_results(df, csv_file_path, verbose)
    # 3. 模型命名检查
    check_model_naming(df, verbose)
    # 4. 打印全局指标（只读）
    analyze_model_results(df, verbose)

    return df

def check_dataset_completeness(df,verbose):
    """检查是否包含所有数据集结果"""
    done_datasets = set(df["dataset"].unique())
    all_datasets = set(ALL_Fast_DATASETS)

    # 缺失的数据集
    missing_datasets = all_datasets - done_datasets
    if missing_datasets:
        if verbose:
            print(f"⚠️ 缺少 {len(missing_datasets)} 个数据集的结果:")
            for dataset in sorted(missing_datasets):
                print(f"  - {dataset}", end=" ")
            print()
    else:
        if verbose:
            print(f"✅ 所有{len(all_datasets)}个预期数据集都有结果")

    # 多出来的数据集
    extra_datasets = done_datasets - all_datasets
    if extra_datasets:
        if verbose:
            print(f"⚠️ 发现 {len(extra_datasets)} 个不在预期列表中的数据集:")
            for dataset in sorted(extra_datasets):
                print(f"  - {dataset}")



def analyze_model_results(df,verbose,verbose_grouped=False):
    """
    对单个模型做一个非常轻量的 sanity check：
    - 解析 dataset 字符串为 ds_key / ds_freq / term
    - 打印全局 sMAPE / MASE / CRPS 的平均值
    """
    if not verbose:
        return

    df = df.copy()
    if "dataset" not in df.columns:
        print("⚠️ DataFrame 中没有 'dataset' 列，跳过分析")
        return
    # 提取数据集信息
    df[['ds_key', 'ds_freq', 'term']] = df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')


    # 定义指标顺序
    metrics = ['sMAPE', 'MASE', 'CRPS']
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("⚠️ 未找到 sMAPE / MASE / CRPS 指标列，跳过全局分析")
        return

    print("df shape:", df.shape, end="  ")
    print("全局平均值:", end=" ")
    for metric in metrics:
        avg = df[metric].mean()
        print(f"{metric}: {avg:.4f}", end=" ")
    print()

    if verbose_grouped:
        # 按分组打印
        all_freqs = sorted(df['ds_freq'].dropna().unique())
        all_terms = sorted(df['term'].dropna().unique())
        all_domains = sorted(df['domain'].dropna().unique())

        # 按不同维度分组并打印
        def group_and_print(group_col, group_name, group_keys):
            metric_dict = {metric: [df[df[group_col] == key][metric].mean() for key in group_keys] for metric in metrics}
            _print_simple_table(metric_dict, group_name, group_keys)

        # 分组打印的辅助函数
        def _print_simple_table(metric_dict, group_name, groups):
            print(f"\n按{group_name}分组结果:")
            header = f"{'':<8}" + "".join([f"{g:<10}" for g in groups])
            print(header)
            for metric, values in metric_dict.items():
                row = f"{metric:<8}"
                for v in values:
                    if pd.isna(v):
                        row += f"{'':<10}"
                    else:
                        row += f"{v:.4f}".ljust(10)
                print(row)
        group_and_print('ds_freq', '数据频率', all_freqs)
        group_and_print('term', '预测长度', all_terms)
        group_and_print('domain', '领域', all_domains)
        group_and_print('num_variates', '变量数量', sorted(df['num_variates'].dropna().unique()))

def check_duplicate_results(df: pd.DataFrame, csv_file_path, verbose: bool = False) -> pd.DataFrame:
    """
    检查并清理“同一个 dataset 出现多条记录”的情况。
    逻辑：
        - 先找出 dataset 重复的行
        - 用 MASE检查是否在容差范围内一致
        - 一致则保留第一条，删除其它重复项
        - 不一致则提醒手动检查

    返回清理后的 DataFrame（如果写回文件成功，会覆盖原 CSV）。
    """
    df_cleaned = df.copy()

    dataset_counts = df_cleaned["dataset"].value_counts()
    duplicate_datasets = dataset_counts[dataset_counts > 1].index.tolist()

    if not duplicate_datasets:
        return df_cleaned

    print(f"发现 {len(duplicate_datasets)} 个数据集有多个结果:", end=" ")

    removed_datasets = []
    needs_save = False

    # 优先用原始列名
    metric_candidates = [ "MASE"]
    metric_col = next((c for c in metric_candidates if c in df_cleaned.columns), None)
    if metric_col is None:
        if verbose:
            print("\n⚠️ 未找到 MASE 列，无法比较重复条目的一致性，仅保留原始数据")
        return df_cleaned

    for dataset in duplicate_datasets:
        dup_rows = df_cleaned[df_cleaned["dataset"] == dataset]

        is_consistent = True
        col_values = dup_rows[metric_col].dropna().astype(float)
        if not col_values.empty and col_values.max() - col_values.min() > 1e-3:
            is_consistent = False

        if is_consistent:
            # 保留第一个结果，移除其他重复项
            first_index = dup_rows.index[0]
            to_remove = dup_rows.index[1:]
            df_cleaned = df_cleaned.drop(to_remove)

            removed_datasets.append((dataset, len(to_remove)))
            needs_save = True
        else:
            print(f" ⚠️  - {dataset}: 有 {len(dup_rows)} 个不一致的结果! 请手动检查")

    if removed_datasets:
        print("已处理的重复数据集:", removed_datasets)
    if needs_save:
        try:
            df_cleaned.to_csv(csv_file_path, index=False)
            print(f"✅ 已清理重复项并保存回原文件: {csv_file_path}")
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")

    return df_cleaned


def check_model_naming(df,verbose):
    """检查模型命名一致性"""

    model_names = df["model"].unique()

    if len(model_names) == 1:
        return

    print(f"⚠️ 发现多种模型命名方式: {model_names}, 请手动检查csv结果文件")

    # 检查是否有数据集混合了不同命名方式
    model_mix = df.groupby("dataset")["model"].unique()
    mixed_datasets = model_mix[model_mix.apply(len) > 1]

    if not mixed_datasets.empty:
        print("⚠️ 警告: 以下数据集混合了不同模型命名方式:")
        for dataset, names in mixed_datasets.items():
            print(f"  - {dataset}: {names}")



def standardize_model_names(baseline_data, model_col: str = "model") -> pd.DataFrame:
    """
    将 'family_size' 形式的 model 名统一映射为标准缩写。
    """
    # 合并 baseline 结果
    baseline_df = pd.concat(baseline_data, ignore_index=True)
    baseline_df[['ds_key', 'ds_freq', 'term']] = baseline_df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')
    def _normalize(name: str) -> str:
        parts = name.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1]}"
        return name

    df = baseline_df.copy()
    df[model_col] = df[model_col].apply(
        lambda x: Model_abbrev_map.get(_normalize(str(x)), x)
    )
    return df

def calculate_order_metrics(df_real, df_pred, k=None):
    """
    计算排序相关指标：
      - Spearman / KendallTau：始终基于完整 real_order / pred_order 计算（不按 K 截断）
      - Top-K 相关指标：
          * Acc_TopK{K}        : 前 K 个交集比例
          * Real1_in_PredK{K}  : real_order[0] 是否落在 pred_order 前 K 中的概率
          * Pred1_in_RealK{K}  : pred_order[0] 是否落在 real_order 前 K 中的概率
    """

    if k is None:
        k_list: list[int] = []
    elif isinstance(k, int):
        k_list = [k]
    else:
        k_list = list(k)

    spearman_vals = []
    kendall_vals = []
    acc_top = {kk: [] for kk in k_list}
    real1_in_pred = {kk: [] for kk in k_list}
    pred1_in_real = {kk: [] for kk in k_list}

    common_datasets = set(df_real["dataset"]).intersection(set(df_pred["dataset"]))

    for dataset in common_datasets:
        real_order = df_real[df_real["dataset"] == dataset]["model_order"].iloc[0]
        pred_order = df_pred[df_pred["dataset"] == dataset]["model_order"].iloc[0]

        if real_order is None or pred_order is None:
            continue
        if len(real_order) == 0 or len(pred_order) == 0:
            continue
        if len(real_order) != len(pred_order):
            # 如果长度不一致，可以根据需要裁剪到最短长度
            min_len = min(len(real_order), len(pred_order))
            real_order = real_order[:min_len]
            pred_order = pred_order[:min_len]

        # 1) Spearman / KendallTau：只按完整排序算一次
        corr, _ = spearmanr(real_order, pred_order)
        tau, _ = kendalltau(real_order, pred_order)
        spearman_vals.append(corr)
        kendall_vals.append(tau)

        # 2) Top-K 指标：针对每个 K 单独计算
        for kk in k_list:
            # 防止 K 大于序列长度
            kk_eff = min(kk, len(real_order), len(pred_order))
            if kk_eff <= 0:
                continue

            real_k = real_order[:kk_eff]
            pred_k = pred_order[:kk_eff]

            # Acc_TopK：交集比例
            correct = len(set(real_k) & set(pred_k)) / kk_eff
            acc_top[kk].append(correct)

            # real_order 第一个是否出现在 pred_order 前 K 中
            real1_in_pred[kk].append(1 if real_order[0] in pred_k else 0)

            # pred_order 第一个是否出现在 real_order 前 K 中
            pred1_in_real[kk].append(1 if pred_order[0] in real_k else 0)

    result = {}

    result["Spearman"] = np.nanmean(spearman_vals) if spearman_vals else np.nan
    result["KendallTau"] = np.nanmean(kendall_vals) if kendall_vals else np.nan

    for kk in k_list:
        if acc_top[kk]:
            result[f"Acc_TopK{kk}"] = np.nanmean(acc_top[kk])
            result[f"Real1_in_PredK{kk}"] = np.nanmean(real1_in_pred[kk])
            result[f"Pred1_in_RealK{kk}"] = np.nanmean(pred1_in_real[kk])
        else:
            result[f"Acc_TopK{kk}"] = np.nan
            result[f"Real1_in_PredK{kk}"] = np.nan
            result[f"Pred1_in_RealK{kk}"] = np.nan

    return result

def filter_models_by_key(model_zoo, select_date, select_key: str = "release_date"):
    """
    按日期字段（默认 release_date）筛选出 <= select_date 的模型，
    并按照日期排序、重新编号 id，返回：

    - filtered_zoo: {family: {size: details_with_id}}
    - sorted_filtered_models: 按 id 排好序的扁平列表
    """
    all_models = []
    for family, sizes in model_zoo.items():
        for size, details in sizes.items():
            if details[select_key] <= select_date:
                details_with_meta = details.copy()
                details_with_meta["_family"] = family
                details_with_meta["_size"] = size
                all_models.append(details_with_meta)

    all_models_sorted = sorted(all_models, key=lambda x: x["release_date"])

    filtered_zoo = {}
    for idx, model in enumerate(all_models_sorted):
        family = model["_family"]
        size = model["_size"]
        model_details = {k: v for k, v in model.items() if not k.startswith("_")}
        model_details["id"] = idx

        if family not in filtered_zoo:
            filtered_zoo[family] = {}
        filtered_zoo[family][size] = model_details

    filtered_models = [
        details for family in filtered_zoo.values() for details in family.values()
    ]
    sorted_filtered_models = sorted(filtered_models, key=lambda x: x["id"])

    return filtered_zoo, sorted_filtered_models



