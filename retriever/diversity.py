# retriever/diversity.py (建议新建这个文件，或者直接放在 engine.py 里)

import torch
import numpy as np
from collections import defaultdict

class DiversityFilter:
    def __init__(self, max_per_dataset=2, similarity_threshold=0.95):
        """
        :param max_per_dataset: 同一个数据集名称下，最多允许保留多少个样本 (硬约束)
        :param similarity_threshold: 如果新样本与已选样本的余弦相似度超过此值，则视为冗余丢弃 (软约束)
        """
        self.max_per_dataset = max_per_dataset
        self.similarity_threshold = similarity_threshold

    def filter(self, candidate_indices, candidate_scores, meta_infos, candidate_embeddings=None):
        """
        :param candidate_indices: 检索出的 Top-N 索引 (List[int])
        :param candidate_scores: 对应的相似度分数 (List[float])
        :param meta_infos: 所有候选项的元数据列表 (List[dict]) - 需要包含 'dataset_name'
        :param candidate_embeddings: (可选) 候选项的 Embedding，用于计算彼此间的相似度
        
        :return: final_indices (List[int])
        """
        selected_indices = []
        dataset_counts = defaultdict(int)
        
        # 如果需要计算内容相似度，先初始化已选集合的 embedding 列表
        selected_embeddings = []
        
        for idx, score in zip(candidate_indices, candidate_scores):
            # 获取当前样本的元数据
            # 注意：这里的 meta_infos 应该是全局的，通过 idx 访问
            meta = meta_infos[idx]
            ds_name = meta.get('dataset_name', 'unknown')
            
            # --- 1. 硬约束: 来源限制 ---
            if dataset_counts[ds_name] >= self.max_per_dataset:
                continue
                
            # --- 2. 软约束: 内容相似度 (MMR 简化版) ---
            # 只有当传入了 embeddings 且已经选了一些样本时才计算
            if candidate_embeddings is not None and len(selected_embeddings) > 0:
                curr_emb = candidate_embeddings[idx] # Tensor or Numpy
                
                # 计算与已选样本的最大相似度
                # 假设 emb 已经 normalized
                if isinstance(curr_emb, torch.Tensor):
                    sims = torch.matmul(torch.stack(selected_embeddings), curr_emb)
                    max_sim = sims.max().item()
                else:
                    sims = np.dot(np.stack(selected_embeddings), curr_emb)
                    max_sim = sims.max()
                
                if max_sim > self.similarity_threshold:
                    continue # 太像了，跳过

            # --- 通过所有检查，加入结果集 ---
            selected_indices.append(idx)
            dataset_counts[ds_name] += 1
            
            if candidate_embeddings is not None:
                selected_embeddings.append(candidate_embeddings[idx])
                
        return selected_indices