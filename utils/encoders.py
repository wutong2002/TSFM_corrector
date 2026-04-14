import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import glob

# =========================================================================
#  0. 智能导入 UniTS 模型 (自动搜索路径)
# =========================================================================
UniTSModel = None
UniTSZeroShotModel = None

def _auto_import_units():
    global UniTSModel, UniTSZeroShotModel
    
    current_dir = os.getcwd()
    search_paths = [current_dir]
    
    deep_paths = glob.glob(os.path.join(current_dir, "**", "UniTS-*"), recursive=True)
    for p in deep_paths:
        if os.path.isdir(os.path.join(p, "models")):
            search_paths.append(p)

    for path in search_paths:
        if path not in sys.path:
            sys.path.append(path)
        
        try:
            if os.path.exists(os.path.join(path, "models", "UniTS.py")):
                from models.UniTS import Model as M1
                from models.UniTS_zeroshot import Model as M2
                UniTSModel = M1
                UniTSZeroShotModel = M2
                return True
        except ImportError:
            continue
        except Exception:
            continue
            
    return False

_units_available = _auto_import_units()


# =========================================================================
#  基类定义
# =========================================================================

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    @property
    def embedding_dim(self):
        raise NotImplementedError
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# =========================================================================
#  1. 直通编码器 (Simple / Identity)
# =========================================================================
class SimpleEncoder(BaseEncoder):
    def __init__(self, input_len):
        super().__init__()
        self.input_len = input_len
        
    @property
    def embedding_dim(self):
        return self.input_len
        
    def encode(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std

# =========================================================================
#  2. 统计特征编码器 (Statistical)
# =========================================================================
class StatisticalEncoder(BaseEncoder):
    def __init__(self, input_len):
        super().__init__()
        self.input_len = input_len
        self._dim = 5
        
    @property
    def embedding_dim(self):
        return self._dim

    def encode(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        min_val = x.min(dim=1, keepdim=True).values
        max_val = x.max(dim=1, keepdim=True).values
        median = x.median(dim=1, keepdim=True).values
        feats = torch.cat([mean, std, min_val, max_val, median], dim=1)
        return torch.nn.functional.normalize(feats, p=2, dim=1)

# =========================================================================
#  3. UniTS 强力编码器 (修复加载问题版)
# =========================================================================
class UnitsEncoder(BaseEncoder):
    def __init__(self, ckpt_path, context_len, device='cuda', model_type='zeroshot', d_model=None):
        super().__init__()
        self.device = device
        self.context_len = context_len
        
        if not _units_available or UniTSZeroShotModel is None:
            raise ImportError("❌ 无法导入 UniTS 模型定义，请检查 'models' 文件夹。")

        print(f"🧠 [UniTS] 初始化编码器，加载权重: {os.path.basename(ckpt_path)}")
        
        if d_model is None:
            import re
            match = re.search(r'_x(\d+)_', os.path.basename(ckpt_path))
            d_model = int(match.group(1)) if match else 128

        self.args = argparse.Namespace(
            d_model=d_model, 
            n_heads=8 if d_model <= 128 else 16,
            e_layers=3,
            patch_len=16,
            stride=16,
            dropout=0.1,
            prompt_num=10,
            right_prob=0.5,
            min_mask_ratio=0.5,
            max_mask_ratio=0.8,
            min_keep_ratio=0.5
        )
        
        dummy_config = [['Generic_Task', {'task_name': 'classification', 'dataset': 'Generic', 'enc_in': 1, 'num_class': 1, 'seq_len': context_len, 'label_len': 0, 'pred_len': 0}]]

        try:
            if model_type == 'zeroshot':
                self.model = UniTSZeroShotModel(self.args, dummy_config, pretrain=False)
            else:
                self.model = UniTSModel(self.args, dummy_config, pretrain=False)
        except Exception as e:
            raise RuntimeError(f"构建 UniTS 模型失败: {e}")

        # === [修复点] 加载权重 ===
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")
            
        try:
            # 兼容新旧 PyTorch 版本
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=device)
        # =======================
        
        state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
        if 'student' in state_dict: state_dict = state_dict['student']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if 'forecast_head' not in name and 'cls_head' not in name:
                new_state_dict[name] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
        self._embed_dim = self.args.d_model

    @property
    def embedding_dim(self):
        return self._embed_dim

    def _forward_backbone(self, x):
        x_enc, means, stdev, n_vars, padding = self.model.tokenize(x)
        
        if hasattr(self.model, 'prompt_token'):
            prefix_prompt = self.model.prompt_token.repeat(x_enc.shape[0], n_vars, 1, 1)
            task_prompt = self.model.cls_token.repeat(x_enc.shape[0], n_vars, 1, 1)
        else:
            prefix_prompt = torch.zeros(x_enc.shape[0], n_vars, self.args.prompt_num, self.args.d_model, device=x.device)
            task_prompt = torch.zeros(x_enc.shape[0], n_vars, 1, self.args.d_model, device=x.device)

        x_enc = torch.reshape(x_enc, (-1, n_vars, x_enc.shape[-2], x_enc.shape[-1]))
        x_enc = x_enc + self.model.position_embedding(x_enc)
        x_final = torch.cat((prefix_prompt, x_enc, task_prompt), dim=2)
        
        seq_len = x_final.shape[-2]
        enc_out = self.model.backbone(x_final, prefix_len=self.args.prompt_num, seq_len=seq_len - self.args.prompt_num)
        
        valid_tokens = enc_out[:, :, self.args.prompt_num:-1, :] 
        ctx_emb = valid_tokens.mean(dim=2) 
        final_emb = ctx_emb.mean(dim=1) 
        
        return final_emb

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.device != self.device:
            x = x.to(self.device)
            
        with torch.no_grad():
            emb = self._forward_backbone(x)
            if torch.isnan(emb).any():
                # 将 NaN 替换为 0 (极少数情况下的兜底)
                emb = torch.nan_to_num(emb, nan=0.0)
        return torch.nn.functional.normalize(emb, p=2, dim=1)