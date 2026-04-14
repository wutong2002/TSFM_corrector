import abc
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
from utils.encoders import BaseEncoder

# 打印标记，确保新文件生效
print("✅ [SchoolwareCore] Module Loaded! Using RIGHT-SIDE truncation.")

class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def add_vectors(self, vectors: torch.Tensor): pass
    @abc.abstractmethod
    def search(self, query_vectors: torch.Tensor, k: int, filter_mask: Optional[torch.Tensor] = None) -> torch.Tensor: pass

class BaseScopeStrategy(abc.ABC):
    @abc.abstractmethod
    def generate_mask(self, query_meta: Dict[str, Any], db_metas: List[Dict[str, Any]]) -> Optional[torch.Tensor]: pass

class SchoolwareDB:
    def __init__(self, encoder: BaseEncoder, retriever: BaseRetriever, debug: bool = False):
        self.encoder = encoder
        self.retriever = retriever
        self.debug = debug
        self.residuals = []   
        self.metadata = []    
        self._built = False

    def add_batch(self, history_data: torch.Tensor, residual_data: torch.Tensor, metas: List[Dict[str, Any]]):
        embeddings = self.encoder.encode(history_data)
        if embeddings.dim() > 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
            
        self.retriever.add_vectors(embeddings, metas=metas)
        self.residuals.extend([r for r in residual_data])
        self.metadata.extend(metas)
        self._built = True

    def query(self, target_history, target_meta=None, scope_strategy=None, top_k=5, output_len=None, debug=False):
        # 1. 编码
        query_emb = self.encoder.encode(target_history)
        if query_emb.dim() > 2: query_emb = query_emb.view(query_emb.size(0), -1)
        
        target_device = getattr(self.retriever, 'device', 'cpu')
        if hasattr(self.retriever, 'keys') and self.retriever.keys is not None:
            target_device = self.retriever.keys.device
            
        query_emb = query_emb.to(target_device)
        
        # 2. Mask
        mask = None
        if scope_strategy is not None and target_meta is not None:
            mask = scope_strategy.generate_mask(target_meta, self.metadata, device=target_device)
            
        # 3. 检索
        top_scores, top_indices = self.retriever.search(query_emb, k=top_k, filter_mask=mask, debug=debug)
        
        # 4. 提取 (Payload Retrieval)
        flat_indices = top_indices.view(-1)
        
        if flat_indices.numel() == 0:
            retrieved_residuals = torch.zeros((0, output_len if output_len else 96), device=target_device)
            retrieved_embs = torch.zeros((0, query_emb.shape[-1]), device=target_device)
        else:
            # Embedding
            if hasattr(self.retriever, 'keys') and self.retriever.keys is not None:
                retrieved_embs = self.retriever.keys[flat_indices]
            else:
                retrieved_embs = torch.zeros((len(flat_indices), query_emb.shape[-1]), device=target_device)

            # Residuals
            temp_res_list = []
            for i, idx in enumerate(flat_indices):
                r = self.residuals[idx.item()]
                if isinstance(r, np.ndarray): r = torch.from_numpy(r)
                r = r.to(target_device)
                
                # === [回滚逻辑] 数据已修正为靠左对齐，所以取前 N 个 ===
                if output_len is not None:
                    if r.shape[0] > output_len:
                        # 修正：取前 output_len 个
                        r = r[:output_len] 
                    elif r.shape[0] < output_len:
                        # 补零 (右侧补零，保持对齐)
                        r = F.pad(r, (0, output_len - r.shape[0]))
                
                # 简单的非零检查
                if debug and i == 0:
                    print(f"🔍 [Query-Debug] Retrieved Residual -> Shape: {tuple(r.shape)} | MeanAbs: {r.abs().mean().item():.6f}")

                temp_res_list.append(r)
            
            retrieved_residuals = torch.stack(temp_res_list) if temp_res_list else torch.zeros((0, output_len or 96), device=target_device)

        return {
            "residuals": retrieved_residuals, 
            "embs": retrieved_embs,           
            "scores": top_scores,             
            "indices": top_indices            
        }