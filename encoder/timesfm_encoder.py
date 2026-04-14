import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

from encoder.base import BaseEncoder

class TimesFMEncoder(BaseEncoder):
    def __init__(self, ckpt_path: str, context_len: int = 512, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.name = "TimesFM"
        self.model_loaded = False
        self._embed_dim = 1280 # TimesFM 200M hidden dim is usually 1280
        
        print(f"⏳ [TimesFMEncoder] 加载模型: {ckpt_path}")
        
        try:
            # 假设 model_zoo 在项目根目录，且已加入 sys.path
            from model_zoo.TSFM_src.timesfm.timesfm_2p5 import timesfm_2p5_torch
            
            self.tfm_wrapper = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
                ckpt_path,
                local_files_only=True
            )
            self.model = self.tfm_wrapper.model
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            # 尝试获取 hidden size
            if hasattr(self.model.config, 'hidden_size'):
                self._embed_dim = self.model.config.hidden_size
                
            print(f"✅ [TimesFMEncoder] 就绪 (Dim: {self._embed_dim})")
        except Exception as e:
            print(f"⚠️ [TimesFMEncoder] 加载失败: {e}")
            raise e

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        x: (B, L) or (B, L, 1)
        """
        if not self.model_loaded: 
            return torch.zeros(len(series_data), self.embedding_dim).to(self.device)
        
        if isinstance(series_data, np.ndarray): 
            x = torch.from_numpy(series_data)
        else:
            x = series_data
            
        x = x.float().to(self.device)
        if x.dim() == 3: x = x.squeeze(-1) 
        
        B, L = x.shape
        patch_len = 32
        
        # [Fix 1] Patching Logic
        if L % patch_len != 0:
            pad_len = patch_len - (L % patch_len)
            x = F.pad(x, (pad_len, 0)) # Pad Left
        
        x_patched = x.view(B, -1, patch_len)
        mask_patched = torch.zeros_like(x_patched) # 0=Valid
        
        # Mask Logic (Mask out left padding)
        if L % patch_len != 0:
            pad_len = patch_len - (L % patch_len)
            flat_mask = torch.zeros((B, x.shape[1]), device=self.device)
            flat_mask[:, :pad_len] = 1.0 
            mask_patched = flat_mask.view(B, -1, patch_len)
            
        with torch.no_grad():
            try:
                outputs = self.model(x_patched, mask_patched)
                
                # [Fix 2] Recursive Unpack
                output_tensor = outputs
                while isinstance(output_tensor, (tuple, list)):
                    if len(output_tensor) > 0: output_tensor = output_tensor[0]
                    else: break
                
                # output_tensor: (B, Num_Patches, Hidden_Dim)
                # Take last patch as embedding (common for Decoder-only)
                embeddings = output_tensor[:, -1, :]
                
            except Exception as e:
                # Fallback zero
                print(f"⚠️ TimesFM Inference Error: {e}")
                return torch.zeros(B, self.embedding_dim).to(self.device)
                
        return F.normalize(embeddings, p=2, dim=1)