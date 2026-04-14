# model_zoo/chronos_utils.py
# 说明：
# - 提供 Chronos 的 monkey-patch（MeanScaleUniformBins 相关）
# - 提供 DataLoader 辅助：identity_collate / SeriesDataset

from typing import Any, List

import torch
from torch.utils.data import Dataset

from chronos.chronos import MeanScaleUniformBins

__all__ = [
    "SeriesDataset",
    "identity_collate",
]

# ====================== Monkey-patch：Chronos 内部 GPU/CPU 设备不一致问题 ======================

# 1) 备份原方法
_orig_input_transform = MeanScaleUniformBins._input_transform
_orig_append_eos = MeanScaleUniformBins._append_eos_token
_orig_output_transform = MeanScaleUniformBins.output_transform


def _patched_input_transform(self, context: torch.Tensor, scale=None):
    """
    保证 self.boundaries 与 context 在同一设备，避免 bucketize 跨设备报错
    """
    if self.boundaries.device != context.device:
        self.boundaries = self.boundaries.to(context.device)
    return _orig_input_transform(self, context, scale)


def _patched_append_eos(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    保证 eos_tokens / eos_mask / attention_mask 都在 token_ids.device 上，
    避免 cuda / cpu 混用导致 cat 失败。
    """
    device = token_ids.device
    batch_size = token_ids.shape[0]

    eos_tokens = torch.full(
        (batch_size, 1),
        fill_value=self.config.eos_token_id,
        device=device,
    )
    eos_mask = torch.full(
        (batch_size, 1),
        fill_value=True,
        device=device,
    )

    attention_mask = attention_mask.to(device)

    token_ids = torch.concat((token_ids, eos_tokens), dim=1)
    attention_mask = torch.concat((attention_mask, eos_mask), dim=1)
    return token_ids, attention_mask


def _patched_output_transform(self, samples: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    确保 self.centers / scale / samples 在同一设备，
    避免 'indices should be either on cpu or on the same device as the indexed tensor' 错误。
    """
    target_device = samples.device if samples.device.type != "cpu" else scale.device

    if self.centers.device != target_device:
        self.centers = self.centers.to(target_device)
    if scale.device != target_device:
        scale = scale.to(target_device)

    return _orig_output_transform(self, samples.to(target_device), scale)


# 实际打上 patch：只要 import 了本模块，就会生效
MeanScaleUniformBins._input_transform = _patched_input_transform
MeanScaleUniformBins._append_eos_token = _patched_append_eos
MeanScaleUniformBins.output_transform = _patched_output_transform

# ====================== DataLoader 辅助：保持 batch 为 list[dict] ======================


def identity_collate(batch: List[Any]) -> List[Any]:
    """collate_fn: 直接返回 batch 原始 list[dict]"""
    return batch


class SeriesDataset(Dataset):
    """把原始 Iterable[dict] 先转 list，再支持多进程采样"""

    def __init__(self, raw):
        self.raw = list(raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        return self.raw[idx]
