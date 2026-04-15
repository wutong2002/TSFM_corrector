from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class WindowRecord:
    tsfm: str
    dataset: str
    timestamp: object
    history: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    local_residual: np.ndarray
    sample_metadata: Dict
    freq: str
    domain: str


def _safe_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def list_tsfms(output_root: str) -> List[str]:
    if not os.path.exists(output_root):
        return []
    return sorted([d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))])


def iter_correction_pickles(output_root: str, tsfm_name: str) -> Iterable[str]:
    tsfm_root = os.path.join(output_root, tsfm_name)
    for root, _, files in os.walk(tsfm_root):
        if "correction_data" not in root:
            continue
        for file in files:
            if file.endswith(".pkl"):
                yield os.path.join(root, file)


def _history_features(history: np.ndarray) -> Dict[str, float]:
    h = _safe_array(history)
    n = len(h)
    x = np.arange(n, dtype=np.float64)

    mean = float(np.mean(h))
    std = float(np.std(h) + 1e-9)
    p10 = float(np.percentile(h, 10))
    p90 = float(np.percentile(h, 90))

    slope = float(np.polyfit(x, h, 1)[0]) if n >= 3 else 0.0

    # 简单分解：trend(移动平均) + remainder
    k = max(3, min(25, n // 8 * 2 + 1))
    if k % 2 == 0:
        k += 1
    trend = pd.Series(h).rolling(window=k, center=True, min_periods=1).mean().to_numpy()
    remainder = h - trend

    trend_var = float(np.var(trend))
    rem_var = float(np.var(remainder) + 1e-9)
    total_var = float(np.var(h) + 1e-9)

    # 频域特征
    centered = h - mean
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    if len(power) <= 1:
        low_ratio = 0.0
        high_ratio = 0.0
        dom_freq = 0.0
    else:
        freqs = np.fft.rfftfreq(n)
        non_dc_power = power[1:]
        non_dc_freqs = freqs[1:]
        low_mask = non_dc_freqs <= 0.1
        high_mask = non_dc_freqs >= 0.3
        energy = np.sum(non_dc_power) + 1e-9
        low_ratio = float(np.sum(non_dc_power[low_mask]) / energy)
        high_ratio = float(np.sum(non_dc_power[high_mask]) / energy)
        dom_freq = float(non_dc_freqs[np.argmax(non_dc_power)])

    return {
        "hist_mean": mean,
        "hist_std": std,
        "hist_range_p10_p90": p90 - p10,
        "trend_slope": slope,
        "trend_strength": trend_var / total_var,
        "remainder_strength": rem_var / total_var,
        "fft_low_ratio": low_ratio,
        "fft_high_ratio": high_ratio,
        "fft_dominant_freq": dom_freq,
    }


def _error_features(truth: np.ndarray, prediction: np.ndarray, residual: Optional[np.ndarray] = None) -> Dict[str, float]:
    t = _safe_array(truth)
    p = _safe_array(prediction)
    if residual is None:
        e = t - p
    else:
        e = _safe_array(residual)

    abs_e = np.abs(e)
    mse = float(np.mean(e ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_e))

    denom = np.abs(t) + np.abs(p)
    valid = denom > 1e-9
    smape = float(np.mean(np.where(valid, 200.0 * abs_e / denom, 0.0)))

    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs_err": float(np.max(abs_e)),
        "std_abs_err": float(np.std(abs_e)),
        "smape": smape,
    }


def load_tsfm_window_dataframe(output_root: str, tsfm_name: str) -> pd.DataFrame:
    rows: List[Dict] = []

    for pkl_path in iter_correction_pickles(output_root, tsfm_name):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        windows = data.get("window_records")
        if windows is None or len(windows) == 0:
            # 向后兼容：旧格式重建
            histories = data.get("histories", [])
            truths = data.get("truths", [])
            preds = data.get("preds", [])
            residuals = data.get("residuals", [])
            local_residuals = data.get("local_residuals", [])
            metas = data.get("sample_metadata", [])
            timestamps = data.get("timestamps", [])
            count = min(len(histories), len(truths), len(preds), len(residuals))
            for i in range(count):
                windows = windows or []
                windows.append({
                    "timestamp": timestamps[i] if i < len(timestamps) else None,
                    "history": histories[i],
                    "truth": truths[i],
                    "prediction": preds[i],
                    "residual": residuals[i],
                    "local_residual": local_residuals[i] if i < len(local_residuals) else np.array([]),
                    "sample_metadata": metas[i] if i < len(metas) else {},
                    "dataset_name": data.get("dataset_name", "unknown"),
                    "freq": data.get("metadata", {}).get("freq", "unknown"),
                    "domain": data.get("metadata", {}).get("domain", "generic"),
                })

        for w in windows:
            history = _safe_array(w["history"])
            truth = _safe_array(w["truth"])
            prediction = _safe_array(w["prediction"])
            residual = _safe_array(w["residual"])
            local_res = _safe_array(w.get("local_residual", []))

            hfeat = _history_features(history)
            efeat = _error_features(truth, prediction, residual)

            rows.append({
                "tsfm": tsfm_name,
                "dataset": w.get("dataset_name", data.get("dataset_name", "unknown")),
                "timestamp": w.get("timestamp"),
                "freq": w.get("freq", data.get("metadata", {}).get("freq", "unknown")),
                "domain": w.get("domain", data.get("metadata", {}).get("domain", "generic")),
                "history_len": len(history),
                "future_len": len(truth),
                "local_residual_len": len(local_res),
                **hfeat,
                **efeat,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def add_quantile_buckets(df: pd.DataFrame, feature_cols: Sequence[str], n_bins: int = 5) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        out[f"{c}_bucket"] = pd.qcut(out[c], q=n_bins, duplicates="drop")
    return out


def compute_feature_combo_error_stats(
    df: pd.DataFrame,
    bucket_features: Sequence[str],
    error_metric: str = "mae",
) -> pd.DataFrame:
    bucket_cols = [f"{f}_bucket" for f in bucket_features]
    grouped = df.groupby(bucket_cols, observed=True)[error_metric]
    stats = grouped.agg(["mean", "max", "std", "count"]).reset_index()
    stats = stats.sort_values("count", ascending=False)
    return stats


def plot_feature_pair_heatmaps(
    stats_df: pd.DataFrame,
    x_bucket_col: str,
    y_bucket_col: str,
    figsize: Tuple[int, int] = (20, 5),
):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, metric, title in zip(axes, ["mean", "max", "std"], ["均值", "最大值", "标准差"]):
        pivot = stats_df.pivot_table(index=y_bucket_col, columns=x_bucket_col, values=metric, aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"误差{title}")
        ax.set_xlabel(x_bucket_col)
        ax.set_ylabel(y_bucket_col)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def build_error_interval_feature_stats(
    df: pd.DataFrame,
    error_metric: str,
    feature_cols: Sequence[str],
    n_error_bins: int = 6,
) -> pd.DataFrame:
    out = df.copy()
    out["error_bucket"] = pd.qcut(out[error_metric], q=n_error_bins, duplicates="drop")
    agg_dict = {}
    for c in feature_cols:
        agg_dict[c] = ["mean", "max", "std"]
    stats = out.groupby("error_bucket", observed=True).agg(agg_dict)
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    return stats.reset_index()


def plot_error_bucket_feature_heatmap(
    stats_df: pd.DataFrame,
    feature_cols: Sequence[str],
    stat: str = "mean",
    figsize: Tuple[int, int] = (12, 6),
):
    mat = []
    for f in feature_cols:
        mat.append(stats_df[f"{f}_{stat}"].to_numpy())
    mat = np.vstack(mat)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels([str(x) for x in stats_df["error_bucket"]], rotation=45, ha="right")
    ax.set_title(f"误差桶内特征 {stat}")
    ax.set_xlabel("error_bucket")
    ax.set_ylabel("feature")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
