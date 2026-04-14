# import abc
# import torch
# import torch.nn.functional as F
# import numpy as np
# from typing import Optional, List, Tuple

# class BaseRetriever(abc.ABC):
#     """
#     [接口] 检索核心算法
#     负责底层向量的存储和相似度搜索 (如 Cosine, L2, FAISS 等)。
#     """
#     @abc.abstractmethod
#     def add_vectors(self, vectors: torch.Tensor):
#         """添加向量到索引"""
#         pass

#     @abc.abstractmethod
#     def search(self, query_vectors: torch.Tensor, k: int, filter_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         搜索 Top-K
#         Args:
#             query_vectors: (B, dim)
#             k: int
#             filter_mask: (Total_N,) 布尔张量，True 表示该样本允许被检索，False 表示屏蔽
#         Returns:
#             scores: (B, k) 相似度分数
#             indices: (B, k) 检索到的索引
#         """
#         pass
import abc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

class BaseRetriever(abc.ABC):
    @abc.abstractmethod
    def add_vectors(self, seq_vectors: torch.Tensor, err_vectors: Optional[torch.Tensor] = None, metas: List[dict] = None):
        pass

    @abc.abstractmethod
    def search(self, q_seq: torch.Tensor, q_err: Optional[torch.Tensor], k: int, 
               alpha: float = 0.5, beta: float = 1.0, filter_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            beta: (0-1) 控制两种策略邻居的占比。
                  beta=1.0: 100% 使用 alpha 权重 (默认行为)
                  beta=0.5: 50% 使用 alpha 权重, 50% 使用 (1-alpha) 权重
        """
        pass
    
    @abc.abstractmethod
    def get_vectors(self, indices: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

class DualMetricRetriever(BaseRetriever):
    def __init__(self, device='cpu', max_per_dataset=100):
        self.device = device
        self.keys_seq = None
        self.keys_err = None
        self.metas = []
        self.max_per_dataset = max_per_dataset
        
    def add_vectors(self, seq_vectors: torch.Tensor, err_vectors: Optional[torch.Tensor] = None, metas: List[dict] = None):
        if isinstance(seq_vectors, np.ndarray): seq_vectors = torch.from_numpy(seq_vectors)
        seq_vectors = seq_vectors.to(self.device)
        seq_vectors = F.normalize(seq_vectors, p=2, dim=1)
        
        if self.keys_seq is None:
            self.keys_seq = seq_vectors
        else:
            self.keys_seq = torch.cat([self.keys_seq, seq_vectors], dim=0)
            
        if err_vectors is not None:
            if isinstance(err_vectors, np.ndarray): err_vectors = torch.from_numpy(err_vectors)
            err_vectors = err_vectors.to(self.device)
            if self.keys_err is None:
                self.keys_err = err_vectors
            else:
                self.keys_err = torch.cat([self.keys_err, err_vectors], dim=0)
        else:
            dummy = torch.zeros_like(seq_vectors)
            if self.keys_err is None: self.keys_err = dummy
            else: self.keys_err = torch.cat([self.keys_err, dummy], dim=0)
            
        if metas:
            self.metas.extend(metas)

    def _compute_l2_similarity(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(q, k, p=2) 
        sim = 1.0 / (1.0 + dist)
        return sim

    def search(self, 
               q_seq: torch.Tensor, 
               q_err: Optional[torch.Tensor], 
               k: int, 
               alpha: float = 0.5,
               beta: float = 1.0, 
               filter_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.keys_seq is None:
            return torch.empty(q_seq.shape[0], 0).to(self.device), torch.empty(q_seq.shape[0], 0, dtype=torch.long).to(self.device)

        B = q_seq.shape[0]
        
        # 1. 计算基础相似度 (按需计算，节省算力)
        # 如果 alpha=0 且 beta=1，其实不需要算 seq sim (除非为了 beta < 1 的策略 B)
        # 但为了代码简单，我们还是都算出来，重点优化后面的组合逻辑
        
        q_seq = q_seq.to(self.device)
        q_seq = F.normalize(q_seq, p=2, dim=1)
        scores_seq = torch.matmul(q_seq, self.keys_seq.t()) 
        
        if q_err is not None and self.keys_err is not None:
            q_err = q_err.to(self.device)
            scores_err = self._compute_l2_similarity(q_err, self.keys_err)
        else:
            scores_err = torch.zeros_like(scores_seq)

        # 2. 计算混合分数 (内存优化版)
        # [关键修改] 避免 alpha=0 或 1 时产生巨大的临时全零矩阵
        
        # --- Strategy A (Primary) ---
        if alpha == 1.0:
            scores_A = scores_seq
        elif alpha == 0.0:
            scores_A = scores_err
        else:
            scores_A = alpha * scores_seq + (1.0 - alpha) * scores_err
        
        # --- Strategy B (Secondary) ---
        scores_B = None
        if beta < 1.0:
            # 只有当需要混合策略时才计算
            # Strategy B 是权重的反转: (1-alpha) * seq + alpha * err
            if alpha == 0.0:   # B uses 1.0*seq + 0.0*err
                scores_B = scores_seq
            elif alpha == 1.0: # B uses 0.0*seq + 1.0*err
                scores_B = scores_err
            else:
                scores_B = (1.0 - alpha) * scores_seq + alpha * scores_err

        # 3. 确定数量
        k_A = int(round(k * beta))
        k_B = k - k_A
        
        # 4. 应用 Mask
        if filter_mask is not None:
            filter_mask = filter_mask.to(self.device)
            # 注意：如果 scores_A 是引用，masked_fill 会修改原数据
            # 但 scores_seq/err 后面不再使用了，所以原地修改是安全的且省内存
            scores_A = scores_A.masked_fill(~filter_mask, -float('inf'))
            if scores_B is not None:
                scores_B = scores_B.masked_fill(~filter_mask, -float('inf'))

        if torch.all(scores_A == -float('inf')):
            return torch.empty(B, 0).to(self.device), torch.empty(B, 0, dtype=torch.long).to(self.device)

        # 5. 执行筛选 (Pre-fetch)
        pre_fetch_k = min(k * 20, scores_A.size(1))
        
        vals_A, inds_A = torch.topk(scores_A, pre_fetch_k, dim=1)
        vals_A_np = vals_A.cpu().numpy()
        inds_A_np = inds_A.cpu().numpy()
        
        vals_B_np, inds_B_np = None, None
        if k_B > 0 and scores_B is not None:
            vals_B, inds_B = torch.topk(scores_B, pre_fetch_k, dim=1)
            vals_B_np = vals_B.cpu().numpy()
            inds_B_np = inds_B.cpu().numpy()

        # 6. CPU 混合逻辑 (保持不变)
        final_indices = []
        final_scores = []
        
        for i in range(B):
            selected_indices = []
            selected_scores = []
            seen_indices = set()
            ds_counter = {} 
            
            # --- Phase A ---
            count_A = 0
            for idx, score in zip(inds_A_np[i], vals_A_np[i]):
                if count_A >= k_A: break
                
                # Diversity Check
                if idx < len(self.metas):
                    ds_name = self.metas[idx].get('dataset_name', 'unknown')
                else: ds_name = 'unknown'
                
                if ds_counter.get(ds_name, 0) >= self.max_per_dataset: continue
                
                selected_indices.append(idx)
                selected_scores.append(score)
                seen_indices.add(idx)
                ds_counter[ds_name] = ds_counter.get(ds_name, 0) + 1
                count_A += 1
            
            # --- Phase B ---
            if k_B > 0 and vals_B_np is not None:
                count_B = 0
                for idx, score in zip(inds_B_np[i], vals_B_np[i]):
                    if count_B >= k_B: break
                    if idx in seen_indices: continue
                    
                    if idx < len(self.metas):
                        ds_name = self.metas[idx].get('dataset_name', 'unknown')
                    else: ds_name = 'unknown'
                    
                    if ds_counter.get(ds_name, 0) >= self.max_per_dataset: continue
                    
                    selected_indices.append(idx)
                    selected_scores.append(score)
                    seen_indices.add(idx)
                    ds_counter[ds_name] = ds_counter.get(ds_name, 0) + 1
                    count_B += 1
            
            # --- Phase C (Fill) ---
            needed = k - len(selected_indices)
            if needed > 0:
                for idx, score in zip(inds_A_np[i], vals_A_np[i]):
                    if len(selected_indices) >= k: break
                    if idx not in seen_indices:
                        selected_indices.append(idx)
                        selected_scores.append(score)
                        seen_indices.add(idx)
                
                if len(selected_indices) < k and vals_B_np is not None:
                    for idx, score in zip(inds_B_np[i], vals_B_np[i]):
                        if len(selected_indices) >= k: break
                        if idx not in seen_indices:
                            selected_indices.append(idx)
                            selected_scores.append(score)
                            seen_indices.add(idx)

            final_indices.append(selected_indices)
            final_scores.append(selected_scores)
            
        return (
            torch.tensor(final_scores, dtype=torch.float32, device=self.device),
            torch.tensor(final_indices, dtype=torch.long, device=self.device)
        )

    def get_vectors(self, indices: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        flat_indices = indices.view(-1)
        seq_embs = self.keys_seq.index_select(0, flat_indices).view(indices.shape[0], indices.shape[1], -1)
        err_embs = None
        if self.keys_err is not None:
            err_embs = self.keys_err.index_select(0, flat_indices).view(indices.shape[0], indices.shape[1], -1)
        return seq_embs, err_embs

    def __len__(self):
        return self.keys_seq.shape[0] if self.keys_seq is not None else 0

class ExactCosineRetriever(BaseRetriever):
    """
    基于 PyTorch 的精确余弦相似度检索器。
    支持多样性过滤 (Max Per Dataset)。
    """
    def __init__(self, device='cpu', max_per_dataset=2):
        self.device = device
        self.keys = None      # (N, D) embeddings
        self.metas = []       # List[dict] 用于多样性过滤
        
        # [关键参数]
        self.max_per_dataset = max_per_dataset
        
    def add_vectors(self, vectors: torch.Tensor, metas: List[dict] = None):
        """
        向索引中添加向量。
        Args:
            vectors: (Batch, Dim) 的 Tensor 或 Numpy 数组
            metas: 对应的元数据列表，用于多样性过滤
        """
        # 1. 统一转为 Tensor 并移动到设备
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)
        
        vectors = vectors.to(self.device)
        
        # 2. 预先做 L2 归一化 (优化 Cosine 计算: Dot Product = Cosine Similarity)
        vectors = F.normalize(vectors, p=2, dim=1)
        
        # 3. 拼接到主索引
        if self.keys is None:
            self.keys = vectors
        else:
            self.keys = torch.cat([self.keys, vectors], dim=0)
            
        if metas:
            self.metas.extend(metas)
            
    def search(self, query_vectors: torch.Tensor, k: int, filter_mask: Optional[torch.Tensor] = None, debug=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索 Top-K 相似向量 (包含多样性过滤逻辑)。
        """
        if debug:
            print(f"[DEBUG] ExactCosineRetriever.search: keys shape={self.keys.shape if self.keys is not None else 'None'}")
        
        if self.keys is None:
            if debug:
                print(f"[DEBUG] ExactCosineRetriever.search: keys is None, returning empty results")
            return torch.empty(query_vectors.shape[0], 0).to(self.device), \
                   torch.empty(query_vectors.shape[0], 0, dtype=torch.long).to(self.device)
            
        # 1. 处理查询向量
        query_vectors = query_vectors.to(self.device)
        query_vectors = F.normalize(query_vectors, p=2, dim=1)
        
        if debug:
            print(f"[DEBUG] ExactCosineRetriever.search: query_vectors shape={query_vectors.shape}, mean={query_vectors.mean().item():.4f}, std={query_vectors.std().item():.4f}")
            print(f"[DEBUG] ExactCosineRetriever.search: keys shape={self.keys.shape}, mean={self.keys.mean().item():.4f}, std={self.keys.std().item():.4f}")
        
        # 2. 计算相似度矩阵 (B, N)
        # Cosine Sim = A . B (因为已归一化)
        scores = torch.matmul(query_vectors, self.keys.t())
        
        if debug:
            print(f"[DEBUG] ExactCosineRetriever.search: scores shape={scores.shape}, mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")
        
        # 3. 应用 filter_mask (如果提供)
        if filter_mask is not None:
            filter_mask = filter_mask.to(self.device)
            # 将被屏蔽的位置分数设为 -inf
            scores = scores.masked_fill(~filter_mask, -float('inf'))
            if debug:
                print(f"[DEBUG] ExactCosineRetriever.search: after filter_mask, scores shape={scores.shape}, mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")

        # === 检查 scores 是否全为 -inf ===
        # 如果所有分数都是 -inf，说明filter_mask过滤掉了所有数据
        if torch.all(scores == -float('inf')):
            if debug:
                print(f"[DEBUG] ExactCosineRetriever.search: 所有分数都是 -inf，过滤掩码可能错误地过滤掉了所有数据")
            return torch.empty(query_vectors.shape[0], 0).to(self.device), \
                   torch.empty(query_vectors.shape[0], 0, dtype=torch.long).to(self.device)

        # === 多样性过滤逻辑 ===
        # 为了进行过滤，我们需要先取比 K 更多的候选者 (Pre-Fetch)
        pre_fetch_k = min(k * 10, scores.size(1))
        top_scores, top_indices = torch.topk(scores, pre_fetch_k, dim=1)
        
        # 如果不需要多样性过滤 (max_per_dataset 很大)，直接返回
        if self.max_per_dataset >= 100 or len(self.metas) == 0:
            return top_scores[:, :k], top_indices[:, :k]

        # 转为 CPU 进行逻辑处理
        top_indices_np = top_indices.cpu().numpy()
        top_scores_np = top_scores.cpu().numpy()
        
        final_indices = []
        final_scores = []
        
        B = query_vectors.shape[0]
        
        for i in range(B):
            candidates = top_indices_np[i]
            c_scores = top_scores_np[i]
            
            selected = []
            selected_s = []
            ds_counter = {} # {dataset_name: count}
            
            for idx, score in zip(candidates, c_scores):
                if len(selected) >= k:
                    break
                
                # 安全获取 Meta
                if idx < len(self.metas):
                    ds_name = self.metas[idx].get('dataset_name', 'unknown')
                else:
                    ds_name = 'unknown'
                
                # 检查频率约束
                curr_count = ds_counter.get(ds_name, 0)
                if curr_count >= self.max_per_dataset:
                    continue 
                
                selected.append(idx)
                selected_s.append(score)
                ds_counter[ds_name] = curr_count + 1
            
            # 如果凑不够 K 个，用剩下的补齐 (即使违反约束)
            if len(selected) < k:
                for idx, score in zip(candidates, c_scores):
                    if len(selected) >= k: break
                    if idx not in selected:
                        selected.append(idx)
                        selected_s.append(score)
            
            final_indices.append(selected)
            final_scores.append(selected_s)
            
        # 转回 Tensor
        final_indices = torch.tensor(final_indices, dtype=torch.long, device=self.device)
        final_scores = torch.tensor(final_scores, dtype=torch.float32, device=self.device)
        
        return final_scores, final_indices

    def __len__(self):
        return self.keys.shape[0] if self.keys is not None else 0
