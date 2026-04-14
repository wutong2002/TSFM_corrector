import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

from encoder.base import BaseEncoder

# 尝试导入 uni2ts
try:
    from uni2ts.model.moirai import MoiraiModule
    from uni2ts.common.torch_util import mask_fill, packed_attention_mask
    _moirai_available = True
except ImportError:
    _moirai_available = False

# =========================================================
# Custom Feature Extractor Class (Fixes Config Issue)
# =========================================================
if _moirai_available:
    class MoiraiFeatureExtractor(MoiraiModule):
        """
        继承 MoiraiModule，主要修复初始化时的 Config 解析问题，
        并重写 forward 以仅输出特征。
        """
        def __init__(self, *args, **kwargs):
            # [Fix 1] 拦截 distr_output 参数
            # 如果它是字典 (config)，则用一个 Dummy 对象替换它
            if 'distr_output' in kwargs and isinstance(kwargs['distr_output'], dict):
                class DummyDistrOutput:
                    def get_param_proj(self, d_model, patch_sizes):
                        return nn.Linear(d_model, 1)
                    def distribution(self, *args, **kwargs): return None
                kwargs['distr_output'] = DummyDistrOutput()
            
            super().__init__(*args, **kwargs)

        def forward(
            self,
            target: torch.Tensor,
            observed_mask: torch.Tensor,
            sample_id: torch.Tensor,
            time_id: torch.Tensor,
            variate_id: torch.Tensor,
            prediction_mask: torch.Tensor,
            patch_size: torch.Tensor,
        ) -> torch.Tensor:
            """
            只执行 Encoder 部分，跳过 Distribution Projection
            """
            # 1. Scaling
            loc, scale = self.scaler(
                target,
                observed_mask * ~prediction_mask.unsqueeze(-1),
                sample_id,
                variate_id,
            )
            scaled_target = (target - loc) / scale

            # 2. Projection
            reprs = self.in_proj(scaled_target, patch_size)

            # 3. Masking
            masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

            # 4. Transformer Encoding
            reprs = self.encoder(
                masked_reprs,
                packed_attention_mask(sample_id),
                time_id=time_id,
                var_id=variate_id,
            )
            
            return reprs

# =========================================================
# Encoder Wrapper
# =========================================================
class MoiraiEncoder(BaseEncoder):
    def __init__(self, ckpt_path: str, context_len: int = 512, device: str = 'cuda', patch_size: int = 32):
        super().__init__()
        if not _moirai_available:
            raise ImportError("请先安装 uni2ts 库")
            
        self.device = torch.device(device)
        self.name = "Moirai"
        self.patch_size = patch_size
        self.max_patch_size = 128 # [Fix 2] Moirai Base 模型要求的最大 patch size
        self.model_loaded = False
        self._embed_dim = 384 # Small 模型通常是 384，Base 是 768 (可从 config 读取)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ Moirai 权重路径不存在: {ckpt_path}")

        print(f"⏳ [MoiraiEncoder] 加载权重: {ckpt_path}")
        try:
            self.model = MoiraiFeatureExtractor.from_pretrained(
                ckpt_path, 
                map_location="cpu"
            )
            self.model.to(self.device)
            self.model.eval()
            self._embed_dim = self.model.d_model
            self.model_loaded = True
            print(f"✅ [MoiraiEncoder] 就绪 (Dim: {self._embed_dim})")
        except Exception as e:
            print(f"⚠️ [MoiraiEncoder] 加载失败: {e}")
            raise e

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args: series_data (B, L) or (B, L, 1)
        """
        if not self.model_loaded: 
            return torch.zeros(len(series_data), self.embedding_dim).to(self.device)
        
        # 1. 格式统一
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data)
        else:
            x = series_data
        
        x = x.float().to(self.device)
        if x.dim() == 2: x = x.unsqueeze(-1)
        
        B, L, V = x.shape
        P = self.patch_size       # 32
        MAX_P = self.max_patch_size # 128
        
        # 2. Padding 时间维度 (补齐 32)
        pad_len = 0
        if L % P != 0:
            pad_len = P - (L % P)
            x = F.pad(x, (0, 0, pad_len, 0)) # Pad time dim
        
        # 3. Reshape to Patches: (B, Num_Patches, 32)
        num_patches = x.shape[1] // P
        target_raw = x.squeeze(-1).view(B, num_patches, P) 
        
        # 4. [Fix 3] Padding Patch维度 (32 -> 128)
        target = F.pad(target_raw, (0, MAX_P - P)) # (B, N, 128)
        
        # 5. 构造参数
        observed_mask = torch.ones_like(target_raw, dtype=torch.bool)
        if pad_len > 0:
            flat_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=self.device)
            flat_mask[:, :pad_len] = False 
            observed_mask = flat_mask.view(B, num_patches, P)
        
        # Mask 也要 Pad 到 128
        observed_mask = F.pad(observed_mask, (0, MAX_P - P), value=False)

        sample_id = torch.zeros((B, num_patches), dtype=torch.long, device=self.device)
        time_id = torch.arange(num_patches, device=self.device).expand(B, -1)
        variate_id = torch.zeros((B, num_patches), dtype=torch.long, device=self.device)
        prediction_mask = torch.zeros((B, num_patches), dtype=torch.bool, device=self.device)
        
        # Patch Size Tensor (告诉模型有效部分是前32)
        patch_size_tensor = torch.full((B, num_patches), P, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            reprs = self.model(
                target=target,
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                patch_size=patch_size_tensor
            )
            # Mean Pooling
            emb = reprs.mean(dim=1)
            
        return F.normalize(emb, p=2, dim=1)