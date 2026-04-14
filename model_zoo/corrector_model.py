import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

# =========================================================================
# 1. 抽象基类 (Interface)
# =========================================================================

class BaseCorrector(nn.Module, abc.ABC):
    """
    [接口] 残差校正器基类
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.pred_len = config.get("pred_len", 96)
        self.embed_dim = config.get("embed_dim", 64)

    @abc.abstractmethod
    def forward(self, 
                target_emb: torch.Tensor, 
                retrieved_embs: torch.Tensor, 
                retrieved_residuals: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            target_emb: (B, D) 当前任务的 Embedding
            retrieved_embs: (B, K, D) 检索到的参考序列 Embedding
            retrieved_residuals: (B, K, L) 检索到的参考序列残差
        Returns:
            pred_residual: (B, L) 预测的目标残差
            info: Dict 额外的调试信息 (如 attention weights)
        """
        pass

    def get_config(self):
        return self.config

# =========================================================================
# 2. 具体实现：基于 Attention 的校正器
# =========================================================================

class AttentionCorrector(BaseCorrector):
    """
    [实现] 使用 Cross-Attention 机制融合检索信息
    结构：Query(Target) -> Attention(Key=Ref, Value=Res) -> MLP -> Output
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        hidden_dim = config.get("hidden_dim", 128)
        num_heads = config.get("num_heads", 4)
        dropout = config.get("dropout", 0.1)
        
        # 特征适配层
        self.query_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.key_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.value_proj = nn.Linear(self.pred_len, hidden_dim) # 将残差映射到隐空间
        
        # 核心注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 融合与输出层
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.output_head = nn.Linear(hidden_dim, self.pred_len)

    def forward(self, target_emb, retrieved_embs, retrieved_residuals):
        # === [调试] 输入完整性检查 ===
        if torch.isnan(target_emb).any():
            raise ValueError("❌ 输入错误: target_emb 包含 NaN！请检查 Encoder 输出。")
        if torch.isnan(retrieved_embs).any():
            raise ValueError("❌ 输入错误: retrieved_embs 包含 NaN！请检查数据库 Embedding。")
        if torch.isnan(retrieved_residuals).any():
            raise ValueError("❌ 输入错误: retrieved_residuals 包含 NaN！请检查数据库中存储的残差数据。")
        # ===========================

        # 1. 构造 Query: (B, 1, H)
        query = self.query_proj(target_emb).unsqueeze(1)
        
        # 2. 构造 Key: (B, K, H)
        key = self.key_proj(retrieved_embs)
        
        # 3. 构造 Value: (B, K, H)
        value = self.value_proj(retrieved_residuals)
        
        # 4. Attention
        attn_out, weights = self.attention(query, key, value)
        
        # 5. 残差连接 + FFN
        x = self.norm(query + attn_out)
        x = x + self.ffn(x)
        
        # 6. 输出
        pred_res = self.output_head(x).squeeze(1) # (B, L)
        
        return pred_res, {"attn_weights": weights}

# =========================================================================
# 3. 简单实现：基于加权平均的线性校正器 (Baseline)
# =========================================================================

class LinearWeightedCorrector(BaseCorrector):
    """
    [实现] 仅根据 Embedding 相似度对残差进行简单的加权平均
    不涉及复杂的神经网络，作为 Baseline 使用
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 可学习的温度系数
        self.temperature = nn.Parameter(torch.tensor(config.get("temperature", 1.0)))

    def forward(self, target_emb, retrieved_embs, retrieved_residuals):
        # target: (B, D) -> (B, 1, D)
        # ref: (B, K, D)
        # Sim: (B, 1, K)
        sim = torch.bmm(target_emb.unsqueeze(1), retrieved_embs.transpose(1, 2))
        
        weights = F.softmax(sim / self.temperature, dim=-1) # (B, 1, K)
        
        # Weighted Sum: (B, 1, K) @ (B, K, L) -> (B, 1, L)
        pred_res = torch.bmm(weights, retrieved_residuals).squeeze(1)
        
        return pred_res, {"attn_weights": weights}
    

# =========================================================================
# 4. 零输出校正器 (Ablation Baseline)
# =========================================================================

class ZeroCorrector(BaseCorrector):
    """
    [实现] 零输出校正器
    始终输出 0，相当于“关闭”校正功能，只保留原始模型的预测结果。
    
    用途：
    1. 作为 Baseline，验证检索增强是否真的带来了提升。
    2. 用于调试，排除模型结构问题导致的 NaN（如果用它训练 Loss 还是 NaN，说明数据标签本身有毒）。
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 不需要任何可学习参数

    def forward(self, target_emb, retrieved_embs, retrieved_residuals):
        """
        忽略所有输入，直接返回全 0 张量
        """
        B = target_emb.shape[0]
        
        # 构造全 0 输出: (B, Pred_Len)
        # 确保设备 (device) 和 数据类型 (dtype) 与输入一致
        pred_res = torch.zeros(
            (B, self.pred_len), 
            device=target_emb.device, 
            dtype=target_emb.dtype
        )
        
        # 返回一个空的 info 字典
        return pred_res, {"type": "zero_baseline"}