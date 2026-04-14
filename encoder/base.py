import abc
import numpy as np
import torch
from typing import Union

class BaseEncoder(abc.ABC):
    """
    [接口] 编码器
    负责将时序数据映射为固定维度的向量。
    """
    @abc.abstractmethod
    def encode(self, series_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            series_data: (N, T) 或 (N, T, C)
        Returns:
            embeddings: (N, embedding_dim)
        """
        pass

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        pass