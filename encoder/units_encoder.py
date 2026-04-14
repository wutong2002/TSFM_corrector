import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import re
import numpy as np
from typing import Union

from encoder.base import BaseEncoder

class UnitsEncoder(BaseEncoder):
    """
    基于 UniTS 的编码器
    """
    def __init__(self, ckpt_path: str, context_len: int = 96, device: str = 'cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.name = "UniTS"
        self.context_len = context_len
        self._embed_dim = 128
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ 权重文件不存在: {ckpt_path}")

        print(f"🧠 [UnitsEncoder] 初始化... 加载权重: {os.path.basename(ckpt_path)}")
        
        # 1. 动态添加路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        units_path = os.path.join(project_root, 'UniTS-main')
        if os.path.exists(units_path) and units_path not in sys.path:
            sys.path.insert(0, units_path)

        try:
            from models.UniTS_zeroshot import Model as UniTSZeroShotModel
            
            # 2. 推断 d_model
            match = re.search(r'_x(\d+)_', os.path.basename(ckpt_path))
            if match: 
                self._embed_dim = int(match.group(1))
            
            # 3. 参数配置
            args = argparse.Namespace(
                d_model=self._embed_dim, n_heads=8 if self._embed_dim <= 128 else 16, e_layers=3,
                patch_len=16, stride=16, dropout=0.1, prompt_num=10,
                right_prob=0.5, min_mask_ratio=0.5, max_mask_ratio=0.8, 
                min_keep_ratio=0.5, prompt_len=10
            )
            dummy_config = [['Generic_Task', {'task_name': 'classification', 'dataset': 'Generic', 'enc_in': 1, 'num_class': 1, 'seq_len': context_len, 'label_len': 0, 'pred_len': 0}]]
            
            self.model = UniTSZeroShotModel(args, dummy_config, pretrain=False)
            
            # 4. 加载权重
            try:
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(ckpt_path, map_location=device)

            state_dict = checkpoint.get('state_dict', checkpoint)
            if 'student' in state_dict: state_dict = state_dict['student']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'forecast_head' not in k and 'cls_head' not in k}
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ [UnitsEncoder] 就绪 (d_model={self._embed_dim})")
            
        except Exception as e:
            print(f"⚠️ [UnitsEncoder] 加载失败: {e}")
            raise e

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(series_data, np.ndarray):
            x = torch.from_numpy(series_data)
        else:
            x = series_data
            
        x = x.float().to(self.device)
        if x.dim() == 2: x = x.unsqueeze(-1)
        
        with torch.no_grad():
            x_enc, _, _, n_vars, _ = self.model.tokenize(x)
            
            if hasattr(self.model, 'prompt_token'):
                prefix_prompt = self.model.prompt_token.repeat(x_enc.shape[0], n_vars, 1, 1)
                task_prompt = self.model.cls_token.repeat(x_enc.shape[0], n_vars, 1, 1)
            else:
                prefix_prompt = torch.zeros(x_enc.shape[0], n_vars, 10, self.model.args.d_model, device=self.device)
                task_prompt = torch.zeros(x_enc.shape[0], n_vars, 1, self.model.args.d_model, device=self.device)

            x_enc = torch.reshape(x_enc, (-1, n_vars, x_enc.shape[-2], x_enc.shape[-1]))
            x_enc = x_enc + self.model.position_embedding(x_enc)
            x_final = torch.cat((prefix_prompt, x_enc, task_prompt), dim=2)
            
            enc_out = self.model.backbone(x_final, prefix_len=10, seq_len=x_final.shape[-2]-10)
            valid_tokens = enc_out[:, :, 10:-1, :]
            emb = valid_tokens.mean(dim=2).mean(dim=1)
            
        return F.normalize(emb, p=2, dim=1)