import torch
import numpy as np
import faiss
import os

class FaissRetriever:
    def __init__(self, input_dim, index_type='HNSW', device='cuda'):
        """
        Args:
            input_dim: Embedding 维度 (例如 128)
            index_type: 'HNSW' (最快/精度高) 或 'IVF' (省内存) Or 'Flat' (暴力)
            device: 'cuda' or 'cpu'
        """
        self.input_dim = input_dim
        self.index_type = index_type
        self.device = device
        self.index = None
        self.is_trained = False
        
        # HNSW 参数
        self.M = 32          # 每个节点的邻居数 (越大精度越高，构建越慢)
        self.ef_search = 64  # 搜索时的深度 (越大精度越高，搜索越慢)
        self.ef_construction = 64 # 构建时的深度

    def _init_index(self):
        if self.index_type == 'HNSW':
            # HNSW 使用 Inner Product (IP) 度量
            # 注意：只要输入向量是归一化的，Inner Product 等价于 Cosine Similarity
            self.index = faiss.IndexHNSWFlat(self.input_dim, self.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            self.is_trained = True # HNSW 不需要训练
            
        elif self.index_type == 'IVF':
            # 倒排索引 (Inverted File Index) - 类似聚类
            # 适合百万级以上数据，显存/内存受限时使用
            quantizer = faiss.IndexFlatIP(self.input_dim)
            nlist = 100 # 聚类中心数量
            self.index = faiss.IndexIVFFlat(quantizer, self.input_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.is_trained = False
            
        elif self.index_type == 'Flat':
            # 暴力搜索 (基准)
            self.index = faiss.IndexFlatIP(self.input_dim)
            self.is_trained = True

        # 如果有 GPU 且安装了 faiss-gpu，可以转移到 GPU
        if self.device == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def fit(self, vectors):
        """
        构建索引
        vectors: (N, D) torch.Tensor or numpy array
        """
        if torch.is_tensor(vectors):
            vectors = vectors.detach().cpu().numpy()
        
        # [重要] Faiss 计算 Cosine 需要先归一化向量
        faiss.normalize_L2(vectors)
        
        if self.index is None:
            self._init_index()
            
        if not self.is_trained and self.index_type == 'IVF':
            print("⚙️ Training Faiss Index (IVF)...")
            self.index.train(vectors)
            self.is_trained = True
            
        print(f"🏗️ Adding {len(vectors)} vectors to Faiss ({self.index_type})...")
        self.index.add(vectors)
        print(f"✅ Index built. Total vectors: {self.index.ntotal}")

    def search(self, query, k=5):
        """
        检索 Top-K
        query: (B, D)
        Return: scores (B, K), indices (B, K)
        """
        if torch.is_tensor(query):
            query_np = query.detach().cpu().numpy()
        else:
            query_np = query
            
        # [重要] Query 也要归一化
        faiss.normalize_L2(query_np)
        
        # 搜索
        # D: Distances (Cos Sim), I: Indices
        D, I = self.index.search(query_np, k)
        
        # 转回 Tensor
        return torch.from_numpy(D).to(self.device), torch.from_numpy(I).to(self.device).long()