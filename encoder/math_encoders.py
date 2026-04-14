import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

# 尝试导入 PyWavelets，如果没有安装则标记
try:
    import pywt
    _pywt_available = True
except ImportError:
    _pywt_available = False

# 引入基类
from encoder.base import BaseEncoder

# # =========================================================
# # 1. FFT Encoder (频谱分析)
# # =========================================================
# class FFTEncoder(BaseEncoder):
#     """
#     利用快速傅里叶变换 (FFT) 提取频域幅值特征。
#     """
#     def __init__(self, output_dim: int = 128):
#         super().__init__()
#         self.output_dim = output_dim
#         self.name = "FFT(Spectral)"

#     @property
#     def embedding_dim(self) -> int:
#         return self.output_dim

#     def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
#         """
#         x: (Batch, Seq_Len)
#         """
#         # 1. 格式统一
#         if isinstance(series_data, np.ndarray):
#             x = torch.from_numpy(series_data)
#         else:
#             x = series_data
            
#         # 确保输入是 float
#         x = x.float()
#         device = x.device
        
#         # 2. 执行实数 FFT
#         # rfft 返回复数张量，形状 (B, L/2 + 1)
#         fft_coeffs = torch.fft.rfft(x, dim=-1)
        
#         # 3. 获取幅值 (Amplitude)
#         fft_amp = fft_coeffs.abs()
        
#         # 4. 维度调整 (截断或补零到 output_dim)
#         B, F_len = fft_amp.shape
        
#         if F_len >= self.output_dim:
#             # 截取低频部分
#             emb = fft_amp[:, :self.output_dim]
#         else:
#             # 补零
#             padding = torch.zeros(B, self.output_dim - F_len, device=device)
#             emb = torch.cat([fft_amp, padding], dim=1)
            
#         # 5. 归一化 (L2)
#         return F.normalize(emb, p=2, dim=1)

# =========================================================
# 2. Statistical Encoder (统计特征)
# =========================================================
class StatisticsEncoder(BaseEncoder):
    """
    提取类似于 CATCH22 的统计特征。
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.name = "Stats(CATCH22-like)"
        self.num_base_features = 9
        
        # 随机正交投影 (仅用于降维模式)
        self.projection = nn.Linear(self.num_base_features, output_dim, bias=False)
        rng = torch.Generator()
        rng.manual_seed(42)
        for param in self.projection.parameters():
            param.requires_grad = False
            nn.init.orthogonal_(param)

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def _get_stats_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 计算 9 种基础统计量"""
        B, L = x.shape
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        centered = x - mean
        skew = (centered ** 3).mean(dim=1, keepdim=True) / (std ** 3)
        kurt = (centered ** 4).mean(dim=1, keepdim=True) / (std ** 4)
        max_val = x.max(dim=1, keepdim=True).values
        min_val = x.min(dim=1, keepdim=True).values
        
        # Zero Crossing Rate
        norm_x = (x - mean)
        zcr = ((norm_x[:, :-1] * norm_x[:, 1:]) < 0).float().mean(dim=1, keepdim=True)
        
        # Autocorrelation (Lag-1)
        lag1_num = ((x[:, :-1] - mean) * (x[:, 1:] - mean)).mean(dim=1, keepdim=True)
        acf_1 = lag1_num / (std ** 2)
        
        # Trend (Slope)
        trend = (x[:, -1] - x[:, 0]).unsqueeze(-1) / L

        features = torch.cat([mean, std, skew, kurt, max_val, min_val, zcr, acf_1, trend], dim=1)
        features = torch.nan_to_num(features, nan=0.0)
        return features

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data)
        else:
            x = series_data
        x = x.float()
        
        base_feats = self._get_stats_torch(x)
        
        if self.projection.weight.device != x.device:
            self.projection = self.projection.to(x.device)
            
        emb = self.projection(base_feats)
        return F.normalize(emb, p=2, dim=1)

# =========================================================
# 3. Wavelet Encoder (小波变换)
# =========================================================
class WaveletEncoder(BaseEncoder):
    def __init__(self, output_dim: int = 128, wavelet: str = 'db4', level: int = 3):
        super().__init__()
        if not _pywt_available:
            raise ImportError("请先安装 PyWavelets: pip install PyWavelets")
        self.output_dim = output_dim
        self.wavelet = wavelet
        self.level = level
        self.name = "Wavelet(DWT)"

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, torch.Tensor):
            x_np = series_data.detach().cpu().numpy()
            device = series_data.device
        else:
            x_np = series_data
            device = 'cpu' 
            
        B, L = x_np.shape
        embeddings = []
        
        for i in range(B):
            ts = x_np[i]
            try:
                coeffs = pywt.wavedec(ts, self.wavelet, level=self.level)
            except ValueError:
                coeffs = pywt.wavedec(ts, self.wavelet, level=None)
            
            flat_coeffs = np.concatenate(coeffs)
            
            if len(flat_coeffs) >= self.output_dim:
                feat = flat_coeffs[:self.output_dim]
            else:
                feat = np.pad(flat_coeffs, (0, self.output_dim - len(flat_coeffs)), 'constant')
            
            embeddings.append(feat)
            
        emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
        if device != 'cpu': emb_tensor = emb_tensor.to(device)
        
        return F.normalize(emb_tensor, p=2, dim=1)


    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

# 尝试导入 PyWavelets
try:
    import pywt
    _pywt_available = True
except ImportError:
    _pywt_available = False

# 引入基类 (假设您的项目中包含此基类)
from encoder.base import BaseEncoder

# =========================================================
# 1. FFT Encoder (频谱分析)
# =========================================================
class FFTEncoder(BaseEncoder):
    """
    利用快速傅里叶变换 (FFT) 提取频域幅值特征。
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.name = "FFT(Spectral)"
        self.device = torch.device("cpu") # 默认设备

    def to(self, device):
        """支持 .to(device) 操作"""
        self.device = device
        return self

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # 1. 格式统一
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
        
        # 确保数据在正确设备
        x = x.to(self.device)
            
        # 2. FFT 变换 (Real FFT)
        fft_res = torch.fft.rfft(x, dim=-1)
        
        # 3. 取幅值 (Modulus)
        fft_amp = torch.abs(fft_res) # (Batch, Freq_Bins)
        
        # 4. 截断或插值到目标维度
        curr_dim = fft_amp.shape[-1]
        if curr_dim > self.output_dim:
            fft_amp = fft_amp[..., :self.output_dim]
        elif curr_dim < self.output_dim:
            fft_amp = F.pad(fft_amp, (0, self.output_dim - curr_dim))
            
        # 归一化 (Log1p + L2)
        fft_amp = torch.log1p(fft_amp)
        fft_amp = F.normalize(fft_amp, p=2, dim=-1)
        
        return fft_amp


# =========================================================
# 2. Chebyshev Encoder (多项式拟合特征)
# =========================================================
class ChebyshevEncoder(BaseEncoder):
    """
    使用切比雪夫多项式拟合时间序列，提取拟合系数作为特征。
    捕捉整体趋势和低频形状。
    """
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.name = "Chebyshev(Trend)"
        self.device = torch.device("cpu")

    def to(self, device):
        """关键：保存设备信息，用于生成 linspace"""
        self.device = device
        return self

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
        
        # 确保输入在正确设备
        x = x.to(self.device)
            
        batch_size, seq_len = x.shape
        degree = self.output_dim - 1
        
        # -----------------------------------------------------------
        # 🛠️ [修复] 确保生成的 linspace 在正确的设备 (GPU) 上
        # -----------------------------------------------------------
        t = torch.linspace(-1, 1, steps=seq_len, device=self.device)
        
        # 生成切比雪夫基底矩阵 (Seq_Len, Degree+1)
        # T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1} - T_{n-2}
        cheb_basis = [torch.ones_like(t), t]
        for i in range(2, degree + 1):
            cheb_basis.append(2 * t * cheb_basis[-1] - cheb_basis[-2])
            
        # Stack -> (Seq_Len, Output_Dim)
        A = torch.stack(cheb_basis, dim=1) 
        
        # 最小二乘法求解系数: Coeff = (A^T A)^-1 A^T * X
        # 由于 A 是固定的 (取决于 seq_len)，可以简化计算
        # (Output_Dim, Seq_Len)
        A_pinv = torch.linalg.pinv(A) 
        
        # (Batch, Seq_Len) @ (Seq_Len, Output_Dim) -> (Batch, Output_Dim)
        # 注意: x 是 (Batch, Seq), A_pinv.T 是 (Seq, Out)
        coeffs = torch.matmul(x, A_pinv.T)
        
        return F.normalize(coeffs, p=2, dim=-1)


# =========================================================
# 3. Spectral Entropy Encoder (谱熵/复杂度)
# =========================================================
class SpectralEntropyEncoder(BaseEncoder):
    """
    计算功率谱熵，反映序列的复杂度/混乱度。
    并结合简单的统计特征 (Mean, Std, Skewness近似)。
    """
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.name = "SpectralEntropy"
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
        
        x = x.to(self.device)

        # 1. 功率谱密度 (PSD)
        fft_res = torch.fft.rfft(x, dim=-1)
        psd = torch.abs(fft_res)**2
        
        # 归一化为概率分布
        psd_sum = torch.sum(psd, dim=-1, keepdim=True) + 1e-9
        psd_prob = psd / psd_sum
        
        # 2. 谱熵 (Spectral Entropy)
        # SE = -Sum(p * log(p))
        se = -torch.sum(psd_prob * torch.log(psd_prob + 1e-9), dim=-1, keepdim=True)
        # 归一化熵
        se = se / np.log(psd.shape[-1])
        
        # 3. 补充统计特征以填满维度
        # 为了凑齐 output_dim，我们可以对 psd 进行分桶平均 (Band Power)
        target_bands = self.output_dim - 1 # 留一位给 Entropy
        if psd_prob.shape[-1] >= target_bands:
            # 简单的池化
            chunks = torch.chunk(psd_prob, target_bands, dim=-1)
            band_powers = torch.stack([c.mean(dim=-1) for c in chunks], dim=-1)
        else:
            band_powers = F.pad(psd_prob, (0, target_bands - psd_prob.shape[-1]))
            
        features = torch.cat([se, band_powers], dim=-1)
        
        return F.normalize(features, p=2, dim=-1)


# =========================================================
# 4. AR Coefficients Encoder (自回归特征)
# =========================================================
class ARCoefficientsEncoder(BaseEncoder):
    """
    利用 Yule-Walker 方程估计 AR(p) 系数。
    捕捉序列的线性相关性结构。
    """
    def __init__(self, output_dim: int = 32, order: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.order = min(order, output_dim)
        self.name = "AR(Coeffs)"
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        简化版：使用 Levinson-Durbin 算法的近似或自相关矩阵求解
        """
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
        
        x = x.to(self.device)
        batch_size, n = x.shape
        
        # 1. 计算自相关函数 (ACF)
        # 通过 FFT 计算: ACF = IFFT(|FFT(x)|^2)
        f_x = torch.fft.rfft(x, n=2*n, dim=-1)
        psd = torch.abs(f_x)**2
        acf = torch.fft.irfft(psd, n=2*n, dim=-1)[..., :self.order+1]
        
        # 归一化 ACF (R[0] = 1)
        r = acf / (acf[..., 0:1] + 1e-9)
        
        # 2. 求解 Yule-Walker 方程 R * phi = r
        # Toeplitz 矩阵求解比较慢，这里直接把 ACF 作为特征
        # 因为 ACF 本身就包含了 AR 系数的信息
        features = r[..., 1:] # 去掉滞后0
        
        # 补齐或截断
        curr_dim = features.shape[-1]
        if curr_dim < self.output_dim:
            features = F.pad(features, (0, self.output_dim - curr_dim))
        else:
            features = features[..., :self.output_dim]
            
        return F.normalize(features, p=2, dim=-1)

# =========================================================
# 4. [FIXED] Hybrid Math Encoder (兼容接口)
# =========================================================
class HybridMathEncoder(BaseEncoder):
    """
    [新] 混合数学编码器
    直接拼接 FFT谱 + 统计特征 + 小波系数，不做任何维度截断（Padding除外）。
    """
    # === [关键修复] 增加 output_dim 参数，虽然内部不使用，但 Trainer 需要传参 ===
    def __init__(self, output_dim: int = None, wavelet: str = 'db4', level: int = 3):
        super().__init__()
        if not _pywt_available:
            raise ImportError("请先安装 PyWavelets: pip install PyWavelets")
        self.name = "Hybrid(FFT+Stats+Wavelet)"
        
        # 即使传入了 output_dim，我们也将其存储但忽略，或者在 embed_dim 中返回它（如果是强制的话）
        # 这里我们选择忽略它，因为混合编码器的核心就是全特征
        self.output_dim = output_dim 
        
        # 复用 StatisticsEncoder 的计算逻辑（但不使用其投影层）
        self.stats_helper = StatisticsEncoder(output_dim=1) 
        
        # 小波参数
        self.wavelet = wavelet
        self.level = level

    @property
    def embedding_dim(self) -> int:
        # 维度是动态的，如果必须返回一个固定值供模型初始化，
        # 返回配置中的 output_dim (如果 config 里算好了)，或者 -1 表示动态
        return self.output_dim if self.output_dim else -1

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        输出维度 = (L/2 + 1) [FFT] + 9 [Stats] + ~L [Wavelet]
        """
        if isinstance(series_data, np.ndarray):
            x_torch = torch.from_numpy(series_data).float()
            x_np = series_data
        else:
            x_torch = series_data.float()
            x_np = series_data.detach().cpu().numpy()
            
        device = x_torch.device
        B, L = x_torch.shape
        
        # 1. FFT
        fft_coeffs = torch.fft.rfft(x_torch, dim=-1)
        feat_fft = fft_coeffs.abs() 
        
        # 2. Stats
        feat_stats = self.stats_helper._get_stats_torch(x_torch) 
        
        # 3. Wavelet
        wavelet_feats = []
        for i in range(B):
            ts = x_np[i]
            try:
                coeffs = pywt.wavedec(ts, self.wavelet, level=self.level)
            except ValueError:
                coeffs = pywt.wavedec(ts, self.wavelet, level=None)
            flat_coeffs = np.concatenate(coeffs)
            wavelet_feats.append(flat_coeffs)
        
        try:
            feat_wavelet = torch.tensor(np.array(wavelet_feats), dtype=torch.float32).to(device)
        except ValueError:
            max_w_len = max(len(f) for f in wavelet_feats)
            padded_feats = [np.pad(f, (0, max_w_len - len(f)), 'constant') for f in wavelet_feats]
            feat_wavelet = torch.tensor(np.array(padded_feats), dtype=torch.float32).to(device)

        # 4. 拼接前归一化
        feat_fft = F.normalize(feat_fft, p=2, dim=1)
        feat_stats = F.normalize(feat_stats, p=2, dim=1)
        feat_wavelet = F.normalize(feat_wavelet, p=2, dim=1)
        
        combined = torch.cat([feat_fft, feat_stats, feat_wavelet], dim=1)
        
        # 5. 最终归一化
        return F.normalize(combined, p=2, dim=1)
# =========================================================
# 5. Balanced Hybrid Encoder (混合编码器) - 核心修改对象
# =========================================================
class BalancedHybridEncoder(BaseEncoder):
    """
    综合多种数学特征的编码器，并使用 LayerNorm 进行平衡。
    """
    def __init__(self):
        super().__init__()
        
        # 实例化子编码器
        self.enc_cheb = ChebyshevEncoder(output_dim=32)
        self.enc_entropy = SpectralEntropyEncoder(output_dim=16)
        self.enc_ar = ARCoefficientsEncoder(output_dim=32, order=16)
        self.enc_fft = FFTEncoder(output_dim=32)
        
        # 计算总维度 (32 + 16 + 32 + 32 = 112)
        self.total_dim = 32 + 16 + 32 + 32
        
        # === 特征平衡层 ===
        # 这些 LayerNorm 包含可学习参数 (Weights)，必须移至 GPU
        self.ln_cheb = nn.LayerNorm(32)
        self.ln_entropy = nn.LayerNorm(16)
        self.ln_ar = nn.LayerNorm(32)
        self.ln_fft = nn.LayerNorm(32)
        
        self.device = torch.device("cpu")

    def to(self, device):
        """
        🛠️ [关键修复]：递归移动所有子模块和参数到指定设备
        """
        self.device = device
        
        # 1. 移动 PyTorch 层 (LayerNorm)
        self.ln_cheb.to(device)
        self.ln_entropy.to(device)
        self.ln_ar.to(device)
        self.ln_fft.to(device)
        
        # 2. 移动子编码器 (让它们更新内部的 self.device)
        self.enc_cheb.to(device)
        self.enc_entropy.to(device)
        self.enc_ar.to(device)
        self.enc_fft.to(device)
        
        return self

    @property
    def embedding_dim(self) -> int:
        return self.total_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # 1. 统一输入格式和设备
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
            
        # 确保输入数据在正确设备
        if x.device != self.device:
            x = x.to(self.device)
        
        # 2. 分别提取 (子编码器会自动处理 device)
        e1 = self.enc_cheb.encode(x)   # (B, 32)
        e2 = self.enc_entropy.encode(x) # (B, 16)
        e3 = self.enc_ar.encode(x)      # (B, 32)
        e4 = self.enc_fft.encode(x)     # (B, 32)
        
        # 3. 独立归一化 (使用 LayerNorm)
        # 这里的 self.ln_xxx 已经在 .to() 中移动到了正确的 device
        e1 = self.ln_cheb(e1)
        e2 = self.ln_entropy(e2)
        e3 = self.ln_ar(e3)
        e4 = self.ln_fft(e4)
        
        # 4. 拼接
        return torch.cat([e1, e2, e3, e4], dim=-1)
    
class RandomNNEncoder(BaseEncoder):
    """
    随机初始化且不进行训练的神经网络编码器 (Random Projection Network)。
    利用随机权重的非线性映射将时间序列投影到固定维度的特征空间。
    """
    def __init__(self, output_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.name = "RandomNN"
        self.device = torch.device("cpu")

        # 构建一个能够自适应处理可变长度时序输入的网络结构
        self.net = nn.Sequential(
            # 使用 1D 卷积提取局部特征 (支持任意长度序列)
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            # 全局最大池化：不管序列多长，都在时间维度上压缩成 1 个点
            nn.AdaptiveMaxPool1d(1), 
            nn.Flatten(),
            # 线性映射到目标输出维度
            nn.Linear(hidden_dim, output_dim)
        )

        # 【核心约束】冻结所有参数，确保它是一个“随机网络”，永不参与训练
        for param in self.net.parameters():
            param.requires_grad = False
        
        # 切换到评估模式 (防止 BatchNorm/Dropout 产生随机性，如果未来加入的话)
        self.net.eval()

    def to(self, device):
        self.device = device
        self.net.to(device)
        return self

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        将时间序列编码为固定维度的向量。
        """
        # 1. 数据类型转换
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data).float()
        else:
            x = series_data.float()
            
        x = x.to(self.device)
        
        # 2. 维度对齐处理
        # 确保输入是 2D: (Batch, Seq_Len)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            x = x.squeeze(-1) # 简单去除多余的特征维度(假设单变量)

        # Conv1d 期望的输入形状为: (Batch, Channels, Seq_Len)
        x = x.unsqueeze(1)

        # 3. 前向传播 (双重保险：不计算梯度)
        with torch.no_grad():
            features = self.net(x)

        # 4. 特征归一化 (对于基于余弦相似度或L2距离的向量检索至关重要)
        return F.normalize(features, p=2, dim=-1)