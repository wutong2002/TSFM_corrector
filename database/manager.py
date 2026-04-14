# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import re

# # 防止 OpenMP 冲突
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# class SchoolwareDB:
#     def __init__(self, encoder, retriever, debug=False):
#         self.encoder = encoder
#         self.retriever = retriever
#         self.debug = debug
#         self.metadata = []
#         self._built = False
        
#         if hasattr(retriever, 'device'): self.device = retriever.device
#         elif hasattr(encoder, 'device'): self.device = encoder.device
#         else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.db_vectors = torch.empty(0, 0).to(self.device)
#         self.db_residuals = torch.empty(0, 0).to(self.device)
#         self.db_dataset_ids = torch.empty(0, dtype=torch.long).to(self.device)
#         self.dataset_name_to_id = {}
#         self.next_dataset_id = 0

#     def _normalize_name(self, name):
#         """
#         [关键修复] 名称归一化
#         剔除 _train, _test, _val, _clean 等后缀，确保同源数据集映射到相同 ID
#         """
#         if not name: return "unknown"
#         # 1. 转小写并去空
#         name = str(name).lower().strip()
        
#         # 2. 剔除常见后缀 (使用正则，避免误伤文件名中间的单词)
#         # 比如: 'lotsa_weather_train_clean' -> 'lotsa_weather'
#         # 移除列表: train, test, val, valid, clean, correction_data, pkl
#         suffix_pattern = r'(_train|_test|_val|_valid|_clean|_correction_data|\.pkl)$'
#         prefix_pattern = r'^(train_|test_|val_|valid_|clean_|LOTSA_|lotsa_|GE_|ge_)'
        
#         name = re.sub(prefix_pattern, '', name)
#         name = re.sub(suffix_pattern, '', name)
            
#         return name

#     def add_batch(self, history, residual, metas=None):
#         with torch.no_grad():
#             embs = self.encoder.encode(history)
        
#         embs = embs.to(self.device)
#         residual = residual.to(self.device)
#         embs_norm = F.normalize(embs, p=2, dim=1)
        
#         if self.db_vectors.numel() == 0:
#             self.db_vectors = embs_norm
#             self.db_residuals = residual
#         else:
#             self.db_vectors = torch.cat([self.db_vectors, embs_norm], dim=0)
#             self.db_residuals = torch.cat([self.db_residuals, residual], dim=0)
            
#         if metas:
#             self.metadata.extend(metas)
#             new_ids = []
#             for m in metas:
#                 # 获取原始名称 -> 归一化 -> 获取 ID
#                 raw_name = m.get('dataset_name', 'unknown')
#                 norm_name = self._normalize_name(raw_name)
                
#                 if norm_name not in self.dataset_name_to_id:
#                     self.dataset_name_to_id[norm_name] = self.next_dataset_id
#                     self.next_dataset_id += 1
#                 new_ids.append(self.dataset_name_to_id[norm_name])
            
#             id_tensor = torch.tensor(new_ids, device=self.device, dtype=torch.long)
#             self.db_dataset_ids = torch.cat([self.db_dataset_ids, id_tensor])
            
#         self._built = True
#         if self.debug and len(self.metadata) % 1000 == 0:
#             print(f"   [DB] Added batch. Total size: {len(self.metadata)}")

#     def query_batch(self, query_batch, meta_batch, scope_mode='global', top_k=5, output_len=None):
#         if not self._built or self.db_vectors.numel() == 0:
#             return self._empty_batch_result(len(meta_batch), top_k, output_len)

#         query_batch = query_batch.to(self.device)
#         if query_batch.shape[1] != self.db_vectors.shape[1]:
#             with torch.no_grad():
#                 query_emb = self.encoder.encode(query_batch)
#         else:
#             query_emb = query_batch
            
#         query_norm = F.normalize(query_emb, p=2, dim=1)
#         scores = torch.mm(query_norm, self.db_vectors.T)

#         # === [核心修复] 极速过滤逻辑 ===
#         if scope_mode == 'cross_dataset':
#             q_ids = []
#             for m in meta_batch:
#                 raw_name = m.get('dataset_name', '')
#                 norm_name = self._normalize_name(raw_name)
#                 # 获取 ID，如果不存在则为 -1
#                 q_ids.append(self.dataset_name_to_id.get(norm_name, -1))
                
#                 # [Debug] 如果 ID 为 -1，说明这个数据集在 DB 中从未出现过，Cross 限制自然失效
#                 # 但如果是 Train/Test 命名差异导致的 -1，就是 Bug。
#                 # 现在的 _normalize_name 应该修复了这个问题。
            
#             q_id_tensor = torch.tensor(q_ids, device=self.device, dtype=torch.long).unsqueeze(1)
#             db_id_tensor = self.db_dataset_ids.unsqueeze(0)
            
#             # Mask: Q_ID == DB_ID 的位置设为 -inf (禁止检索)
#             # 注意：如果 q_id 为 -1 (新数据集)，mask 全为 False，允许全局检索 (符合逻辑)
#             mask = (q_id_tensor == db_id_tensor)
            
#             # 只有当 q_id 有效(>=0)时才进行屏蔽
#             valid_q_mask = (q_id_tensor >= 0)
#             final_mask = mask & valid_q_mask
            
#             scores.masked_fill_(final_mask, -1e9)

#         top_scores, top_indices = torch.topk(scores, top_k, dim=1)

#         flat_indices = top_indices.view(-1)
#         flat_res = self.db_residuals.index_select(0, flat_indices)
#         flat_emb = self.db_vectors.index_select(0, flat_indices)
        
#         retrieved_residuals = flat_res.view(top_indices.shape[0], top_indices.shape[1], -1)
#         retrieved_embs = flat_emb.view(top_indices.shape[0], top_indices.shape[1], -1)

#         if output_len is not None:
#             curr_len = retrieved_residuals.shape[2]
#             if curr_len > output_len:
#                 retrieved_residuals = retrieved_residuals[:, :, :output_len]
#             elif curr_len < output_len:
#                 pad_len = output_len - curr_len
#                 retrieved_residuals = torch.nn.functional.pad(retrieved_residuals, (0, pad_len))

#         return {
#             'residuals': retrieved_residuals,
#             'embs': retrieved_embs,
#             'scores': top_scores
#         }

#     def _empty_batch_result(self, batch_size, top_k, output_len):
#         dim = self.db_vectors.shape[1] if self.db_vectors.numel() > 0 else 128
#         out_len = output_len if output_len else 96
#         return {
#             'residuals': torch.zeros(batch_size, top_k, out_len, device=self.device),
#             'embs': torch.zeros(batch_size, top_k, dim, device=self.device),
#             'scores': torch.zeros(batch_size, top_k, device=self.device)
#         }
import torch
import torch.nn.functional as F
import numpy as np
import os
import re

# 防止 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class SchoolwareDB:
    def __init__(self, encoder, retriever, debug=False):
        self.encoder = encoder
        self.retriever = retriever
        self.debug = debug
        self.metadata = []
        self._built = False
        
        # 确定设备
        if hasattr(retriever, 'device'): self.device = retriever.device
        elif hasattr(encoder, 'device'): self.device = encoder.device
        else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 存储管理 ===
        # 向量 (Vectors/Keys) 现在完全委托给 self.retriever 管理
        # 数据库仅管理 残差 (Values) 和 ID (Metadata)
        
        # Value 存储
        self.db_residuals = torch.empty(0, 0).to(self.device)
        
        # ID 存储 (用于 Cross-Dataset 过滤)
        self.db_dataset_ids = torch.empty(0, dtype=torch.long).to(self.device)
        self.dataset_name_to_id = {}
        self.next_dataset_id = 0

        # === [新增] 元数据索引存储 ===
        # 1. Dataset ID (用于 cross-dataset)
        self.db_dataset_ids = torch.empty(0, dtype=torch.long).to(self.device)
        self.dataset_name_to_id = {}
        self.next_dataset_id = 0
        # 🌟 [新增] 序列序号与窗口起始位置的存储
        self.db_seq_ids = torch.empty(0, dtype=torch.long).to(self.device)
        self.db_hist_starts = torch.empty(0, dtype=torch.long).to(self.device)
        # 👇 [新增] 目标片段起止点存储
        self.db_hist_ends = torch.empty(0, dtype=torch.long).to(self.device)   
        self.db_target_ends = torch.empty(0, dtype=torch.long).to(self.device)
        # 2. Freq ID (用于 filter_freq)
        self.db_freq_ids = torch.empty(0, dtype=torch.long).to(self.device)
        self.freq_to_id = {}
        self.next_freq_id = 0
        
        # 3. Domain ID (用于 filter_domain)
        self.db_domain_ids = torch.empty(0, dtype=torch.long).to(self.device)
        self.domain_to_id = {}
        self.next_domain_id = 0

        
    def _get_id(self, val, mapping, next_id_attr):
        """通用 ID 映射辅助函数"""
        # 转字符串并小写，确保 "5T" 和 "5t" 视为同一个
        key = str(val).lower().strip() if val else "unknown"
        if key not in mapping:
            curr_id = getattr(self, next_id_attr)
            mapping[key] = curr_id
            setattr(self, next_id_attr, curr_id + 1)
        return mapping[key]

    def _normalize_name(self, name):
        """名称归一化"""
        if not name: return "unknown"
        name = str(name).lower().strip()
        # 剔除常见后缀和前缀
        suffix_pattern = r'(_train|_test|_val|_valid|_clean|_correction_data|\.pkl)$'
        prefix_pattern = r'^(train_|test_|val_|valid_|clean_|LOTSA_|lotsa_|GE_|ge_)'
        name = re.sub(prefix_pattern, '', name)
        name = re.sub(suffix_pattern, '', name)
        return name

    def add_batch(self, history, residual, metas=None, local_residuals=None):
        """
        [更新] 添加数据到数据库
        向量交给 Retriever，残差自己存。
        """
        # 1. 编码历史序列 (Channel 1: Sequence)
        with torch.no_grad():
            embs = self.encoder.encode(history)
        embs = embs.to(self.device) # Retriever 会自动处理归一化
        
        residual = residual.to(self.device)
        
        # 2. 编码局部残差 (Channel 2: Error Pattern)
        err_embs = None
        if local_residuals is not None:
            with torch.no_grad():
                err_embs = self.encoder.encode(local_residuals)
            err_embs = err_embs.to(self.device) # Retriever 会保留原值用于 L2

        # 3. 添加到索引 (Retriever)
        # 注意：add_vectors 负责处理 embs 和 err_embs 的存储
        self.retriever.add_vectors(embs, err_embs, metas=metas)

        # 4. 本地存储 Residuals (Value)
        if self.db_residuals.numel() == 0:
            self.db_residuals = residual
        else:
            self.db_residuals = torch.cat([self.db_residuals, residual], dim=0)
            
        # === [修改] 处理元数据 ID ===
        if metas:
            self.metadata.extend(metas)
            new_ds_ids = []
            new_freq_ids = []
            new_domain_ids = []
            new_seq_ids = []      # 🌟 新增
            new_hist_starts = []  # 🌟 新增
            new_hist_ends = []    
            new_target_ends = []
            for m in metas:
                # 1. Dataset Name ID
                raw_name = m.get('dataset_name', 'unknown')
                norm_name = self._normalize_name(raw_name)
                if norm_name not in self.dataset_name_to_id:
                    self.dataset_name_to_id[norm_name] = self.next_dataset_id
                    self.next_dataset_id += 1
                new_ds_ids.append(self.dataset_name_to_id[norm_name])
                
                # 2. Freq ID (读取 'freq')
                freq_val = m.get('freq', 'unknown')
                new_freq_ids.append(self._get_id(freq_val, self.freq_to_id, 'next_freq_id'))
                
                # 3. Domain ID (读取 'domain')
                domain_val = m.get('domain', 'generic')
                new_domain_ids.append(self._get_id(domain_val, self.domain_to_id, 'next_domain_id'))

                # 🌟 提取序列 ID 和起始位置
                new_seq_ids.append(m.get('seq_id', -1))
                new_hist_starts.append(m.get('hist_start', -1))
                # 👇 [新增] 计算目标窗口的起始与结束索引
                h_end = m.get('hist_end', -1)
                v_len = m.get('valid_len', 0)
                new_hist_ends.append(h_end)
                new_target_ends.append(h_end + v_len if h_end >= 0 else -1)
            # 拼接到 Tensor
            self.db_dataset_ids = torch.cat([self.db_dataset_ids, torch.tensor(new_ds_ids, device=self.device)])
            self.db_freq_ids = torch.cat([self.db_freq_ids, torch.tensor(new_freq_ids, device=self.device)])
            self.db_domain_ids = torch.cat([self.db_domain_ids, torch.tensor(new_domain_ids, device=self.device)])
            # 🌟 保存状态
            self.db_seq_ids = torch.cat([self.db_seq_ids, torch.tensor(new_seq_ids, device=self.device)])
            self.db_hist_starts = torch.cat([self.db_hist_starts, torch.tensor(new_hist_starts, device=self.device)])
            # 👇 [新增] 保存起止点
            self.db_hist_ends = torch.cat([self.db_hist_ends, torch.tensor(new_hist_ends, device=self.device)])
            self.db_target_ends = torch.cat([self.db_target_ends, torch.tensor(new_target_ends, device=self.device)])
        self._built = True
        
        if self.debug and len(self.metadata) % 1000 == 0:
            print(f"   [DB] Added batch. Total Size: {len(self.metadata)}")

    def query_batch(self, query_batch, meta_batch, scope_mode='global', top_k=5, output_len=None, 
                    query_local_res=None, alpha=0.5, beta=1.0, filter_by_freq=False, filter_by_domain=False,
                    exclude_self=False):
        """
        [终极加固版] 批量检索接口：预分配 Buffer，双向 Clamp 抵御显存越界崩溃。
        """
        B = len(meta_batch)
        db_size = self.db_residuals.shape[0]
        
        # 0. 基础检查
        if not self._built or db_size == 0:
            return self._empty_batch_result(B, top_k, output_len)

        # === 1. 准备 Query Vectors ===
        query_batch = query_batch.to(self.device)
        def _get_emb(data_batch):
            if self.retriever.keys_seq is not None:
                db_dim = self.retriever.keys_seq.shape[-1]
                if data_batch.dim() > 2 or data_batch.shape[-1] != db_dim:
                    with torch.no_grad():
                        return self.encoder.encode(data_batch)
            return data_batch

        query_emb = _get_emb(query_batch)
        q_err_emb = _get_emb(query_local_res.to(self.device)) if query_local_res is not None else None

        # 🌟 2. 构造精细化屏蔽掩码 (Masking)
        # 获取 Query 的各种 ID
        q_ids = [self.dataset_name_to_id.get(self._normalize_name(m.get('dataset_name', '')), -1) for m in meta_batch]
        q_id_t = torch.tensor(q_ids, device=self.device, dtype=torch.long).unsqueeze(1)
        
        q_seq_ids = [m.get('seq_id', -1) for m in meta_batch]
        q_seq_t = torch.tensor(q_seq_ids, device=self.device, dtype=torch.long).unsqueeze(1)
        
        # 👇 [新增] 获取 Query 的目标片段起止点
        q_hist_ends = [m.get('hist_end', -1) for m in meta_batch]
        q_target_ends = [m.get('hist_end', -1) + m.get('valid_len', 0) if m.get('hist_end', -1) >= 0 else -1 for m in meta_batch]
        
        q_h_end_t = torch.tensor(q_hist_ends, device=self.device, dtype=torch.long).unsqueeze(1)
        q_t_end_t = torch.tensor(q_target_ends, device=self.device, dtype=torch.long).unsqueeze(1)

        # 定义匹配条件
        same_ds = (q_id_t == self.db_dataset_ids.unsqueeze(0)) & (q_id_t >= 0)
        same_seq = (q_seq_t == self.db_seq_ids.unsqueeze(0)) & (q_seq_t >= 0)
        
        # 👇 [核心修改] 判断目标片段是否交叠
        # 区间 [start1, end1) 和 [start2, end2) 交叠的充要条件是：start1 < end2 AND start2 < end1
        db_h_ends = self.db_hist_ends.unsqueeze(0)
        db_t_ends = self.db_target_ends.unsqueeze(0)
        is_overlapping = (q_h_end_t < db_t_ends) & (db_h_ends < q_t_end_t)
        
        # 确保参与判断的索引是合法提取出来的（不为 -1）
        valid_overlap = is_overlapping & (q_h_end_t >= 0) & (db_h_ends >= 0)

        # 🌟 应用 4 种查搜模式 (False 表示屏蔽)
        if scope_mode == 'cross_dataset':
            base_mask = ~same_ds
        elif scope_mode == 'exclude_seq':
            base_mask = ~(same_ds & same_seq)
        elif scope_mode == 'exclude_self':
            # 👇 [修改] 只要同数据集、同序列，且目标窗口有交叠，一律屏蔽！
            base_mask = ~(same_ds & same_seq & valid_overlap)
        elif scope_mode == 'allow_self':
            base_mask = torch.ones((B, db_size), device=self.device, dtype=torch.bool)
        else:
            base_mask = torch.ones((B, db_size), device=self.device, dtype=torch.bool)

        strict_mask = base_mask.clone()

        # === 3. 构造严格限制 Mask ===
        strict_mask = base_mask.clone()
        if filter_by_freq:
            q_freq_ids = [self.freq_to_id.get(str(m.get('freq', 'unknown')).lower().strip(), -1) for m in meta_batch]
            q_freq_t = torch.tensor(q_freq_ids, device=self.device).unsqueeze(1)
            strict_mask &= (q_freq_t == self.db_freq_ids.unsqueeze(0))
            
        if filter_by_domain:
            q_dom_ids = [self.domain_to_id.get(str(m.get('domain', 'generic')).lower().strip(), -1) for m in meta_batch]
            q_dom_t = torch.tensor(q_dom_ids, device=self.device).unsqueeze(1)
            strict_mask &= (q_dom_t == self.db_domain_ids.unsqueeze(0))

        # === 4. 第一阶段：严格检索 ===
        search_k = top_k + 5 if exclude_self else top_k
        top_scores_raw, top_indices_raw = self.retriever.search(
            q_seq=query_emb, q_err=q_err_emb, k=search_k, alpha=alpha, beta=beta, filter_mask=strict_mask 
        )

        final_indices = torch.full((B, top_k), -1, device=self.device, dtype=torch.long)
        final_scores = torch.full((B, top_k), -1e9, device=self.device, dtype=torch.float32)
        valid_counts = torch.zeros(B, dtype=torch.long, device=self.device)

        if top_indices_raw.numel() > 0 and top_indices_raw.shape[1] > 0:
            for i in range(B):
                r_idx = top_indices_raw[i]
                r_sc = top_scores_raw[i]
                
                # 过滤出合法的索引 (不能小于 0，也不能大于等于 db_size)
                mask_v = (r_idx >= 0) & (r_idx < db_size)
                
                if exclude_self:
                    mask_v &= (r_sc < 0.99999999)
                
                v_idx = r_idx[mask_v]
                v_sc = r_sc[mask_v]
                k1 = min(len(v_idx), top_k)
                if k1 > 0:
                    final_indices[i, :k1] = v_idx[:k1]
                    final_scores[i, :k1] = v_sc[:k1]
                valid_counts[i] = k1

        # === 5. 第二阶段：兜底补全 ===
        if (valid_counts < top_k).any():
            for i in range(B):
                k1 = valid_counts[i].item()
                if k1 < top_k:
                    k_needed = top_k - k1
                    fill_mask = base_mask[i].clone()
                    if k1 > 0:
                        # 确保用作掩码的索引合法
                        valid_fill_idx = final_indices[i, :k1]
                        valid_fill_idx = valid_fill_idx[(valid_fill_idx >= 0) & (valid_fill_idx < db_size)]
                        fill_mask[valid_fill_idx] = False
                        
                    if fill_mask.any():
                        actual_k_req = min(k_needed + (5 if exclude_self else 0), int(fill_mask.sum().item()))
                        f_scores, f_indices = self.retriever.search(
                            q_seq=query_emb[i:i+1],
                            q_err=q_err_emb[i:i+1] if q_err_emb is not None else None,
                            k=actual_k_req, alpha=alpha, beta=beta, filter_mask=fill_mask.unsqueeze(0)
                        )
                        
                        if f_indices.numel() > 0:
                            f_idx_row = f_indices.squeeze(0)
                            f_sc_row = f_scores.squeeze(0)
                            
                            mask_f = (f_idx_row >= 0) & (f_idx_row < db_size)
                            if exclude_self:
                                mask_f &= (f_sc_row < 0.9999999)
                                
                            v_f_idx = f_idx_row[mask_f]
                            v_f_sc = f_sc_row[mask_f]
                            
                            num_filled = min(len(v_f_idx), k_needed)
                            if num_filled > 0:
                                final_indices[i, k1:k1+num_filled] = v_f_idx[:num_filled]
                                final_scores[i, k1:k1+num_filled] = v_f_sc[:num_filled]

        # === 6. 提取结果 ===
        top_indices = final_indices
        top_scores = final_scores
        
        # 严苛校验掩码：只保留 [0, db_size - 1] 区间内的索引
        final_valid_mask = (top_indices >= 0) & (top_indices < db_size)
        
        # 👑 [绝杀保护] 双向 Clamp，生成绝对安全的索引
        max_idx = max(0, db_size - 1)
        safe_indices = top_indices.clamp(min=0, max=max_idx)
        flat_indices = safe_indices.view(-1)

        # 1. 提取残差
        flat_res = self.db_residuals[flat_indices]
        retrieved_residuals = flat_res.view(B, top_k, -1)

        # 2. 提取 Embs (必须使用 safe_indices 防止内部 Embedding 算子崩溃)
        retrieved_embs, retrieved_err_embs = self.retriever.get_vectors(safe_indices)

        # 3. 统一使用 Mask 刷掉无效数据 (将借用 0 号索引补位的假数据刷成全 0)
        if not final_valid_mask.all():
            valid_mask_f = final_valid_mask.unsqueeze(-1).float()
            retrieved_residuals = retrieved_residuals * valid_mask_f
            
            # 对序列指纹应用 mask
            retrieved_embs = retrieved_embs * valid_mask_f
            
            # 对误差指纹应用 mask
            if retrieved_err_embs is not None:
                retrieved_err_embs = retrieved_err_embs * valid_mask_f

        # === 7. 长度调整 ===
        if output_len is not None:
            curr_len = retrieved_residuals.shape[2]
            if curr_len > output_len:
                retrieved_residuals = retrieved_residuals[:, :, :output_len]
            elif curr_len < output_len:
                retrieved_residuals = torch.nn.functional.pad(retrieved_residuals, (0, output_len - curr_len))

        return {
            'residuals': retrieved_residuals, 
            'embs': retrieved_embs,           
            'err_embs': retrieved_err_embs,   
            'scores': top_scores
        }

    def _empty_batch_result(self, batch_size, top_k, output_len):
        # 尝试从 retriever 获取维度，如果为空则默认 128
        dim = 128
        if self.retriever.keys_seq is not None:
            dim = self.retriever.keys_seq.shape[1]
            
        out_len = output_len if output_len else 96
        return {
            'residuals': torch.zeros(batch_size, top_k, out_len, device=self.device),
            'embs': torch.zeros(batch_size, top_k, dim, device=self.device),
            'err_embs': None,
            'scores': torch.zeros(batch_size, top_k, device=self.device)
        }