import torch
import torch.fft
import numpy as np
from typing import Union
from .base import BaseEncoder

class AdvancedStatisticalEncoder(BaseEncoder):
    """
    [高级实现] 基于全面统计特征的时序编码器。
    
    能够提取包括趋势、季节性、分布形态、频域特性和复杂性在内的多种学术界认可特征。
    旨在作为 Units 等深度学习 Encoder 的轻量级、可解释替代方案。
    """
    def __init__(self, input_len=96, embedding_dim=128):
        super().__init__()
        self.input_len = input_len
        self._dim = embedding_dim
        
        # 预计算时间索引，用于线性回归 (Batch无关，节省计算)
        self.register_buffer('time_idx', torch.arange(input_len).float())

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def register_buffer(self, name, tensor):
        # 模拟 nn.Module 的 buffer 注册行为，确保 device 一致性
        setattr(self, name, tensor)

    def _ensure_device(self, x):
        if self.time_idx.device != x.device:
            self.time_idx = self.time_idx.to(x.device)

    def _get_distribution_features(self, x):
        """提取分布特征: Mean, Std, Skewness, Kurtosis, Quantiles"""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True) + 1e-6
        
        # 标准化用于计算高阶矩
        z = (x - mean) / std
        skewness = torch.mean(z**3, dim=1, keepdim=True)
        kurtosis = torch.mean(z**4, dim=1, keepdim=True) - 3.0 # Excess Kurtosis
        
        # 分位数 (Min, 25%, Median, 75%, Max)
        quantiles = torch.quantile(x, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device), dim=1).t()
        
        return [mean, std, skewness, kurtosis, quantiles]

    def _get_trend_features(self, x):
        """提取趋势特征: Slope, Intercept, R^2 (基于最小二乘法)"""
        B, L = x.shape
        t = self.time_idx # (L,)
        if L != len(t): # 处理实际输入长度不一致的情况
            t = torch.arange(L, device=x.device).float()
            
        t_mean = t.mean()
        x_mean = x.mean(dim=1, keepdim=True)
        
        # 协方差与方差
        numerator = torch.sum((t - t_mean) * (x - x_mean), dim=1, keepdim=True)
        denominator = torch.sum((t - t_mean)**2) + 1e-6
        
        slope = numerator / denominator
        intercept = x_mean - slope * t_mean
        
        # 计算 R^2 (Linearity)
        y_pred = slope * t + intercept
        ss_tot = torch.sum((x - x_mean)**2, dim=1, keepdim=True)
        ss_res = torch.sum((x - y_pred)**2, dim=1, keepdim=True)
        r2 = 1 - (ss_res / (ss_tot + 1e-6))
        
        return [slope, intercept, r2]

    def _get_spectral_features(self, x):
        """提取频域特征: FFT Top-K Freqs/Amps, Spectral Entropy"""
        # 实数傅里叶变换
        fft = torch.fft.rfft(x, dim=1)
        power_spectrum = fft.abs()**2
        
        # 1. 谱熵 (Spectral Entropy) - 衡量序列的复杂度/不可预测性
        ps_norm = power_spectrum / (power_spectrum.sum(dim=1, keepdim=True) + 1e-9)
        entropy = -(ps_norm * torch.log(ps_norm + 1e-9)).sum(dim=1, keepdim=True)
        # 归一化熵值
        entropy = entropy / np.log(power_spectrum.shape[1])
        
        # 2. Top-K 频率成分 (忽略直流分量 idx=0)
        # 取前3个主频及其强度
        k = 3
        amplitudes = fft.abs()[:, 1:] # (B, L/2)
        topk_amp, topk_indices = torch.topk(amplitudes, k=min(k, amplitudes.shape[1]), dim=1)
        
        # 归一化频率 (0~0.5)
        topk_freqs = topk_indices.float() / x.shape[1]
        
        return [entropy, topk_amp, topk_freqs]

    def _get_volatility_features(self, x):
        """提取波动性特征: CV, Differencing, Zero-Crossing"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        
        # 变异系数 (Coefficient of Variation)
        cv = std / (torch.abs(mean) + 1e-6)
        
        # 一阶差分的均值 (反映短期变化率)
        diff = x[:, 1:] - x[:, :-1]
        diff_mean = torch.mean(torch.abs(diff), dim=1, keepdim=True)
        
        # 过零率 (Zero Crossing Rate) - 归一化
        zcr = ((x[:, :-1] * x[:, 1:]) < 0).float().mean(dim=1, keepdim=True)
        
        return [cv, diff_mean, zcr]

    def _get_acf_features(self, x, max_lag=10):
        """提取自相关特征 (AutoCorrelation Function)"""
        # 利用 FFT 计算 ACF 能够加速
        n = x.shape[1]
        # Pad to avoid circular correlation effects
        x_pad = torch.nn.functional.pad(x, (0, n))
        x_fft = torch.fft.rfft(x_pad, dim=1)
        # Autocorrelation is inverse FFT of Power Spectrum
        acf = torch.fft.irfft(x_fft * torch.conj(x_fft), dim=1)[:, :n]
        
        # Normalize
        variance = acf[:, 0:1] + 1e-6
        acf = acf / variance
        
        # 取前 max_lag 个滞后 (忽略 lag 0)
        # 如果长度不够，动态调整
        real_lag = min(max_lag, n-1)
        return [acf[:, 1:real_lag+1]]

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        执行特征提取与编码
        Returns: (Batch, embedding_dim)
        """
        # 1. 输入处理
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
            
        # 处理维度 (Batch, Time)
        if x.dim() == 3: 
            x = x.squeeze(-1) # 假设单变量
        
        # 长度截断或对齐
        if x.shape[1] > self.input_len:
            x = x[:, -self.input_len:]
        
        self._ensure_device(x)
        
        # 2. 特征提取 pipeline
        features = []
        
        # -> Distribution (9 features)
        features.extend(self._get_distribution_features(x))
        
        # -> Trend (3 features)
        features.extend(self._get_trend_features(x))
        
        # -> Volatility (3 features)
        features.extend(self._get_volatility_features(x))
        
        # -> Spectral (1 + 3 + 3 = 7 features)
        features.extend(self._get_spectral_features(x))
        
        # -> ACF (10 features)
        features.extend(self._get_acf_features(x, max_lag=12)) # 捕捉到月度季节性
        
        # 3. 特征拼接
        # Flatten all lists
        emb = torch.cat([f if f.dim() == 2 else f.unsqueeze(1) for f in features], dim=1)
        
        # 4. 维度对齐 (Padding / Truncation)
        # 确保输出维度严格等于 embedding_dim
        B, current_dim = emb.shape
        
        if current_dim < self._dim:
            # 维度不足：补零
            padding = torch.zeros(B, self._dim - current_dim, device=emb.device)
            emb = torch.cat([emb, padding], dim=1)
        elif current_dim > self._dim:
            # 维度过大：截断 (优先保留前面的基础统计和趋势特征)
            emb = emb[:, :self._dim]
            
        # 5. Nan 处理 (防止统计计算中的除零导致 NaNs)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return emb.cpu()