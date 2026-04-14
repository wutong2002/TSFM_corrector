import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseScopeStrategy(ABC):
    @abstractmethod
    def filter(self, db_metadata: list, query_meta: dict) -> list:
        pass

class GlobalScopeStrategy(BaseScopeStrategy):
    def filter(self, db_metadata: list, query_meta: dict) -> list:
        return list(range(len(db_metadata)))

class CausalTimeScopeStrategy(BaseScopeStrategy):
    def filter(self, db_metadata: list, query_meta: dict) -> list:
        q_time = query_meta.get('timestamp')
        if q_time is None: return list(range(len(db_metadata)))
        
        valid_indices = []
        for idx, meta in enumerate(db_metadata):
            m_time = meta.get('timestamp')
            if m_time is not None and m_time < q_time:
                valid_indices.append(idx)
        return valid_indices

class DatasetScopeStrategy(BaseScopeStrategy):
    """
    [数据集内因果策略] - 仅检索同名数据集且时间在前的样本
    """
    def filter(self, db_metadata: list, query_meta: dict) -> list:
        q_ds = query_meta.get('dataset_name') or query_meta.get('dataset')
        q_time = query_meta.get('timestamp')
        
        if q_ds is None: return []

        q_ds_norm = str(q_ds).lower().strip()

        valid_indices = []
        for idx, meta in enumerate(db_metadata):
            m_ds = meta.get('dataset_name') or meta.get('dataset')
            if m_ds is None: continue
            
            # 1. 名称必须匹配
            m_ds_norm = str(m_ds).lower().strip()
            if m_ds_norm != q_ds_norm:
                continue
                
            # 2. 时间必须在前
            if q_time is not None:
                m_time = meta.get('timestamp')
                if m_time is None or m_time >= q_time:
                    continue
            
            valid_indices.append(idx)
            
        return valid_indices

class CrossDatasetScopeStrategy(BaseScopeStrategy):
    """
    [跨数据集策略] - 严格剔除同名数据集 (防止自检索)
    """
    def filter(self, db_metadata: list, query_meta: dict) -> list:
        q_ds = query_meta.get('dataset_name') or query_meta.get('dataset')
        
        # === [Fix] 安全防御 ===
        # 如果不知道查询样本来自哪个数据集，绝对不能进行全局搜索！
        # 否则会直接导致数据泄露（检索到自身）。
        if q_ds is None:
            # print("⚠️ [Strategy Warning] Query dataset name is None! Returning empty scope to prevent leakage.")
            return []
        
        q_ds_norm = str(q_ds).lower().strip()
        
        # === [Debug] 仅在第一次调用时打印，用于诊断 ===
        # 如果您发现日志中打印了这行，且 DB Total > 0，说明 filter 正在工作
        if not hasattr(self, '_debug_printed'):
            # print(f"🔍 [CrossDataset Debug] Query: '{q_ds_norm}'. Filtering out same-name datasets...")
            self._debug_printed = True
        
        valid_indices = []
        for idx, meta in enumerate(db_metadata):
            m_ds = meta.get('dataset_name') or meta.get('dataset')
            
            if m_ds:
                m_ds_norm = str(m_ds).lower().strip()
                # 只有当名字不相等时，才允许检索 (Keep if DIFFERENT)
                if m_ds_norm != q_ds_norm:
                    valid_indices.append(idx)
            else:
                # 如果 DB 样本没有名字，为了安全起见，通常也保留(假设它是外部数据)或者丢弃
                # 这里选择保留，但在同源数据集场景下风险较低
                valid_indices.append(idx)
                
        return valid_indices