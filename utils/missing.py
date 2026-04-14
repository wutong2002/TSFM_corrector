# utils/missing.py
import numpy as np
from scipy.interpolate import interp1d

def fill_missing_1d(
    x: np.ndarray,
    *,
    all_nan_strategy: str = "zero",      # "zero" 或 "linspace"
    interp_kind: str = "linear",        # "linear" 或 "nearest"
    add_noise: bool = False,            # 是否在插值结果上加一点噪声
    noise_ratio: float = 0.01,          # 噪声强度（相对于均值）
) -> np.ndarray:
    """
    对 1D 序列做缺失值处理：
    - 支持 NaN / Inf
    - all_nan_strategy 控制整条坏掉时的默认填充值
    - interp_kind 控制插值方式
    - add_noise 控制是否在结果上叠加少量高斯噪声（VisionTS 风格）
    """
    x = np.asarray(x, dtype=float)

    # finite_mask: 既不是 NaN 也不是 +/-Inf
    finite_mask = np.isfinite(x)

    # 整条都是坏值（NaN 或 Inf）
    if not finite_mask.any():
        if all_nan_strategy == "linspace":
            return np.linspace(0.0, 1.0, len(x), dtype=float)
        else:  # "zero" 或其它情况统一置零
            return np.zeros_like(x, dtype=float)

    # 已经全部是正常数值
    if finite_mask.all():
        filled = x.copy()
    else:
        idx = np.arange(len(x))
        known_idx = idx[finite_mask]
        known_vals = x[finite_mask]

        # 只有一个已知点：直接全填成这个常数
        if len(known_idx) == 1:
            filled = np.full_like(x, known_vals[0], dtype=float)
        else:
            if interp_kind == "nearest":
                kind = "nearest"
            else:
                kind = "linear"
            f = interp1d(
                known_idx,
                known_vals,
                kind=kind,
                fill_value="extrapolate",
            )
            filled = f(idx).astype(float)

    # 可选：在插值结果上加一点噪声
    if add_noise:
        base = float(np.mean(np.abs(filled))) if np.isfinite(filled).any() else 1.0
        scale = noise_ratio * max(base, 1e-8)
        noise = np.random.normal(0.0, scale, size=len(filled))
        filled = filled + noise

    return filled


def fill_missing(
    arr: np.ndarray,
    *,
    all_nan_strategy_1d: str = "zero",
    all_nan_strategy_2d_global: str | None = None,
    interp_kind_1d: str = "linear",
    interp_kind_2d: str = "linear",
    add_noise_1d: bool = False,
    noise_ratio_1d: float = 0.01,
) -> np.ndarray:
    """
    统一对 1D / 2D 序列做缺失值处理的入口：

    - 1D：直接调用 fill_missing_1d
    - 2D：按“行（通常对应 channel）”逐条调用 fill_missing_1d
        * 如果整个 2D 都是 NaN，并且 all_nan_strategy_2d_global="linspace"，
          则每一行都填 0~1 的 linspace
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return fill_missing_1d(
            arr,
            all_nan_strategy=all_nan_strategy_1d,
            interp_kind=interp_kind_1d,
            add_noise=add_noise_1d,
            noise_ratio=noise_ratio_1d,
        )

    if arr.ndim == 2:
        # 整个二维矩阵都是 NaN 的特殊情况
        if all_nan_strategy_2d_global == "linspace" and np.isnan(arr).all():
            T = arr.shape[1]
            base = np.linspace(0.0, 1.0, T, dtype=float)
            return np.vstack([base.copy() for _ in range(arr.shape[0])])

        out = np.empty_like(arr, dtype=float)
        for i in range(arr.shape[0]):
            out[i] = fill_missing_1d(
                arr[i],
                # 通道级别 all-NaN 一般更安全地置 0（与 VisionTS 原逻辑一致）
                all_nan_strategy="zero",
                interp_kind=interp_kind_2d,
                add_noise=False,
            )
        return out

    raise ValueError(f"fill_missing 目前只支持 1D 或 2D 数组，收到 arr.ndim = {arr.ndim}")
