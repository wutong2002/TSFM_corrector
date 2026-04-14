import os
import importlib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any
import math

# === 动态导入依赖 ===
try:
    from database.manager import SchoolwareDB
except ImportError:
    try:
        from .manager import SchoolwareDB
    except ImportError:
        SchoolwareDB = Any

# ------------------------------------------------------------------
# 核心组件 1：物理一致的时间序列扰动器
# ------------------------------------------------------------------
def generate_consistent_perturbations(history, truth, num_variations, strength, debug=False):
    """
    对给定的 time series (history + truth) 产生多个科学的物理扰动版本。
    确保历史和未来之间的趋势、缩放和振幅保持连续性。
    [新增] 当 debug=True 或 strength=0 时，直接返回原始数据的精确副本。
    """
    L_hist = len(history)
    L_truth = len(truth)
    
    # 🐛 Debug / 0 强度 短路通道：直接原样返回，不做任何计算
    if debug or strength <= 0:
        return [history.copy() for _ in range(num_variations)], [truth.copy() for _ in range(num_variations)]

    full_seq = np.concatenate([history, truth])
    L_total = len(full_seq)
    
    std_val = np.std(full_seq) + 1e-6
    
    perturbed_histories = []
    perturbed_truths = []
    
    for _ in range(num_variations):
        seq = full_seq.copy()
        
        # 1. 缩放 (Scaling): 整体变大或变小
        scale_factor = np.random.normal(1.0, strength * 0.2)
        scale_factor = np.clip(scale_factor, 0.5, 2.0)
        seq = seq * scale_factor
        
        # 2. 高频抖动 (Jittering)
        noise = np.random.normal(0, std_val * strength * 0.3, size=L_total)
        seq = seq + noise
        
        # 3. 随机趋势注入 (Random Trend)
        # 模拟序列整体上扬或下降的微小趋势改变
        trend_slope = np.random.normal(0, strength * 0.1)
        trend_line = np.linspace(-1, 1, L_total) * std_val * trend_slope
        seq = seq + trend_line
        
        perturbed_histories.append(seq[:L_hist])
        perturbed_truths.append(seq[L_hist:])
        
    return perturbed_histories, perturbed_truths

# ------------------------------------------------------------------
# 核心组件 2：内存数据集包装器 (兼容 TSFM Predictor)
# ------------------------------------------------------------------
class _DummyTestData:
    def __init__(self, entries):
        self.input = entries

class _DummyMemDataset:
    """
    一个极简的鸭子类型数据集，用于将内存中的伪造序列直接喂给 TSFM Predictor。
    完美模拟 utils.data.Dataset 的属性结构，适应 base_model.py 的标准流程。
    """
    def __init__(self, entries, pred_len, freq="H"):
        self.gluonts_dataset = entries
        self.prediction_length = pred_len
        self.freq = freq
        
        # --- 维度定义属性 ---
        self.target_dim = 1
        # [核心修复] 满足 Moirai 等模型对协变量维度的强校验
        self.feat_dynamic_real_dim = 0
        self.past_feat_dynamic_real_dim = 0
        
        # 模拟 base_model.py 中的 dataset.test_data.input 结构
        self.test_data = _DummyTestData(entries)
        
    def __iter__(self): return iter(self.gluonts_dataset)
    def __len__(self): return len(self.gluonts_dataset)

# ------------------------------------------------------------------
# 主数据集类
# ------------------------------------------------------------------
class CorrectionDataset(Dataset):
    def __init__(
        self,
        db,
        samples: list,
        context_len: int,
        top_k: int,
        pred_len=96,
        retrieval_strategy='encoder',
        scope_strategy=None,
        precompute_embeddings=True,
        debug=False,
        shuffle_retrieved_order=False,
        pseudo_ratio=0.0,
        pseudo_aug_strength=0,
        tsfm_predictor=None,               # 用于生成扰动历史的预测值
        retrieval_alpha=1.0,
        retrieval_beta=1.0,
        filter_by_freq=False,
        filter_by_domain=False,
    ):
        self.db = db
        self.samples = samples
        self.context_len = context_len
        self.top_k = top_k
        self.pred_len = pred_len
        self.retrieval_strategy = retrieval_strategy
        self.debug = debug
        self.shuffle_order = shuffle_retrieved_order
        self.pseudo_ratio = pseudo_ratio
        self.aug_strength = pseudo_aug_strength
        self.tsfm_predictor = tsfm_predictor
        self.retrieval_alpha = retrieval_alpha
        self.retrieval_beta = retrieval_beta
        self.filter_by_freq = filter_by_freq
        self.filter_by_domain = filter_by_domain
        
        # 解析 Scope 模式
        self.scope_mode = 'global'
        if scope_strategy:
            s_name = getattr(scope_strategy, '__class__', '').__name__
            if 'CrossDataset' in s_name or 'cross' in str(scope_strategy):
                self.scope_mode = 'cross_dataset'
            elif 'SameModel' in s_name:
                self.scope_mode = 'same_model'
            elif 'Global' in s_name:
                self.scope_mode = 'global'

        # ==========================================
        # 1. 预处理 (Tensorizing) 与异常过滤
        # ==========================================
        self.history_tensors = []
        self.history_norm_tensors = []
        self.local_res_tensors = []
        self.local_res_norm_tensors = []
        self.raw_res_tensors = []
        self.scaled_res_tensors = []
        self.truth_tensors = []
        self.scales = [] 
        self.valid_lens = [] 
        self.freqs = []
        self.smape_quantiles = [] # <--- [新增] 存储每个样本的排位
        
        # 🛡️ 预设 float32 的安全阈值 (约 3.4e38，我们取安全边界)
        f32_max = np.finfo(np.float32).max * 0.9 
        valid_samples = []
        discard_stats = {"NaN_Inf_Overflow": 0} # 🌟 新增追踪
        for item in tqdm(self.samples, desc="Tensorizing & Filtering"):
            
            # --- [核心新增] 异常与溢出严苛校验 ---
            def has_overflow(arr):
                if arr is None or len(arr) == 0: return False
                # 必须用 float64 承接，防止在检查环节自身就引发 float32 溢出警告
                arr_np = np.array(arr, dtype=np.float64).reshape(-1)
                # 检查 NaN, Inf, 或超出 float32 极限的数值
                if np.isnan(arr_np).any() or np.isinf(arr_np).any(): return True
                if np.max(np.abs(arr_np)) > f32_max: return True
                return False
            
            # 如果该样本的任何关键序列存在溢出，直接记录并丢弃！
            if has_overflow(item.get('history')) or \
               has_overflow(item.get('truth')) or \
               has_overflow(item.get('residual')) or \
               has_overflow(item.get('local_residual')):
                discard_stats["NaN_Inf_Overflow"] += 1
                continue
                
            # 只有干净的样本才会留下来
            valid_samples.append(item)
            
            # 1. History
            hist_t = self._process_tensor(item['history'], self.context_len, is_history=True)
            self.history_tensors.append(hist_t)
            
            # 2. Scale
            hist_abs = torch.abs(hist_t)
            valid_mask = hist_abs > 1e-9
            scale = torch.mean(hist_abs[valid_mask]).item() if valid_mask.sum() > 0 else 1.0
            scale = max(scale, 1e-6) 
            self.scales.append(scale)
            
            # 3. Norm History
            h_norm_t = hist_t / scale
            self.history_norm_tensors.append(h_norm_t)
            
            # 4. Valid Len & Residuals (Future)
            raw_residual = item.get('residual', [])
            v_len = item.get('valid_len')
            if v_len is None: v_len = len(raw_residual)
            if v_len > self.pred_len: v_len = self.pred_len
            self.valid_lens.append(v_len)
            self.freqs.append(item.get('freq', None))
            raw_res_t, truth_t = self._get_relative_residual(raw_residual, item.get('truth', []))
            
            # 5. Norm Residuals (Future)
            norm_res_t = raw_res_t / scale
            clamped_norm_res_t = torch.clamp(norm_res_t, -20.0, 20.0)
            clamped_raw_res_t = clamped_norm_res_t * scale
            diff = raw_res_t - clamped_raw_res_t
            clamped_truth_t = truth_t - diff
            
            self.raw_res_tensors.append(clamped_raw_res_t)
            self.truth_tensors.append(clamped_truth_t)
            self.scaled_res_tensors.append(clamped_norm_res_t)
            
            # 6. Local Residuals (Past Fingerprint)
            raw_loc_res = item.get('local_residual', []) 
            loc_res_t = self._process_tensor(raw_loc_res, self.context_len, is_history=True)
            self.local_res_tensors.append(loc_res_t)
            
            loc_res_norm_t = loc_res_t / scale
            self.local_res_norm_tensors.append(loc_res_norm_t)

            # [新增] 提取并保存样本的分位数
            self.smape_quantiles.append(item.get('smape_quantile', 100.0))
        # 🌟 [新增] 在 tqdm 循环结束后，如果剔除了样本，打印信息
        if discard_stats["NaN_Inf_Overflow"] > 0:
            print(f"\n⚠️ [Dataset 构建层] 深度 Tensor 化筛查: 额外剔除了 {discard_stats['NaN_Inf_Overflow']} 个潜在溢出/NaN样本。\n")
        # 🚀 [关键一步] 用干净的样本覆盖原始列表，保证后续的 Meta 提取和总量对齐绝对一致
        self.samples = valid_samples

        # ==========================================
        # 2. Embedding (Batch Encoding)
        # ==========================================
        self.precomputed_embeddings = []     # 序列指纹
        self.precomputed_err_embeddings = [] # 误差指纹
        
        if precompute_embeddings:
            enc_bs = 512 
            enc_device = getattr(self.db.encoder, 'device', 'cpu')
            if hasattr(self.db.encoder, 'eval'): self.db.encoder.eval()

            with torch.no_grad():
                # A. 编码序列 (History)
                if len(self.history_norm_tensors) > 0:
                    all_hists = torch.cat(self.history_norm_tensors, dim=0)
                    for i in tqdm(range(0, len(all_hists), enc_bs), desc="Encoding Seq"):
                        batch = all_hists[i:i+enc_bs].to(enc_device)
                        embs = self.db.encoder.encode(batch).cpu()
                        if embs.dim() == 3: embs = embs.squeeze(1)
                        self.precomputed_embeddings.extend(torch.split(embs, 1, dim=0))
                
                # B. 编码误差 (Local Residuals)
                if len(self.local_res_norm_tensors) > 0:
                    all_errs = torch.cat(self.local_res_norm_tensors, dim=0)
                    for i in tqdm(range(0, len(all_errs), enc_bs), desc="Encoding Err"):
                        batch = all_errs[i:i+enc_bs].to(enc_device)
                        embs = self.db.encoder.encode(batch).cpu()
                        if embs.dim() == 3: embs = embs.squeeze(1)
                        self.precomputed_err_embeddings.extend(torch.split(embs, 1, dim=0))

        # ==========================================
        # 3. [核心新增] 批量预计算物理伪造样本 
        # ==========================================
        self.pseudo_pool = {}
        if self.pseudo_ratio > 0.0:
            self._precompute_physical_pseudo_samples()

        # ==========================================
        # 4. 极速批量检索 (Batch Retrieval)
        # ==========================================
        self.retrieved_embs_cache = []      # 邻居序列指纹
        self.retrieved_err_embs_cache = []  # 邻居误差指纹
        self.retrieved_residuals_cache = [] # 邻居未来残差
        
        if self.retrieval_strategy in ['encoder', 'causal'] and getattr(self.db, '_built', True):
            original_db_debug = self.db.debug
            self.db.debug = False 
            
            retrieval_bs = 512 
            total_samples = len(self.samples)
            
            try:
                all_metas = [{
                    'dataset_name': s.get('dataset', 'unknown'),
                    'source_model': s.get('source_model', None),
                    'freq': s.get('freq'),
                    'domain': s.get('domain'),
                    'seq_id': s.get('sample_meta', {}).get('seq_id', -1),
                    'hist_start': s.get('sample_meta', {}).get('hist_start', -1),
                    'hist_end': s.get('sample_meta', {}).get('hist_end', -1),
                    'valid_len': s.get('valid_len', 96)
                } for s in self.samples]
                
                self.all_metas = all_metas
                
                for i in tqdm(range(0, total_samples, retrieval_bs), desc="Retrieving"):
                    end_idx = min(i + retrieval_bs, total_samples)
                    torch.cuda.empty_cache()
                    
                    # A. 准备 Sequence Query
                    if self.precomputed_embeddings:
                        batch_query = torch.cat(self.precomputed_embeddings[i:end_idx], dim=0)
                    else:
                        batch_query = torch.cat(self.history_norm_tensors[i:end_idx], dim=0)
                    
                    # B. 准备 Error Query
                    batch_query_err = None
                    if self.precomputed_err_embeddings:
                        batch_query_err = torch.cat(self.precomputed_err_embeddings[i:end_idx], dim=0)
                    elif self.local_res_norm_tensors:
                        batch_query_err = torch.cat(self.local_res_norm_tensors[i:end_idx], dim=0)
                    
                    # C. 准备 Meta
                    batch_meta = all_metas[i:end_idx]
                    
                    # D. 调用双路检索接口
                    retrieved = self.db.query_batch(
                        query_batch=batch_query, 
                        meta_batch=batch_meta, 
                        scope_mode=self.scope_mode, 
                        top_k=self.top_k + 5,
                        output_len=self.pred_len,
                        query_local_res=batch_query_err,
                        alpha=self.retrieval_alpha,
                        beta=self.retrieval_beta,
                        filter_by_freq=self.filter_by_freq,
                        filter_by_domain=self.filter_by_domain,
                        # scope_mode=self.scope_mode,
                    )
                    
                    # E. 缓存结果
                    self.retrieved_residuals_cache.extend(torch.unbind(retrieved['residuals'].cpu(), dim=0))
                    self.retrieved_embs_cache.extend(torch.unbind(retrieved['embs'].cpu(), dim=0))
                    
                    if retrieved.get('err_embs') is not None:
                        self.retrieved_err_embs_cache.extend(torch.unbind(retrieved['err_embs'].cpu(), dim=0))
                    else:
                        self.retrieved_err_embs_cache.extend([None] * (end_idx - i))
                    
            finally:
                self.db.debug = original_db_debug
    # 在 dataset.py 的 RetrievalCorrectionDataset 类中，或者外部定义此辅助函数
    def _extract_ts_physics_features(self, hist_array, local_res_array, scale):
        """提取时序物理特征与误差特征"""
        # 1. 归一化局部残差 (最重要的特征)
        norm_res = local_res_array / (scale + 1e-8)
        res_mean_abs = np.mean(np.abs(norm_res))
        res_max_abs = np.max(np.abs(norm_res))
        res_std = np.std(norm_res)
        
        # 2. 原始序列的波动与趋势
        hist_std = np.std(hist_array)
        diff1 = np.diff(hist_array)
        trend_proxy = np.mean(diff1)
        complexity = np.mean(np.abs(diff1))
        
        # 3. 频域特征 (计算主导频率占比)
        fft_vals = np.abs(np.fft.rfft(hist_array - np.mean(hist_array)))
        dom_freq = np.max(fft_vals[1:]) / (np.sum(fft_vals[1:]) + 1e-8) if len(fft_vals) > 1 else 0.0
        
        # 组装为 8 维特征向量
        features = np.array([
            res_mean_abs, res_max_abs, res_std, 
            hist_std, trend_proxy, complexity, dom_freq, scale
        ], dtype=np.float32)
        return features
# ------------------------------------------------------------------
    # 核心新增：物理级批量预计算
    # ------------------------------------------------------------------
    def _precompute_physical_pseudo_samples(self):
        k_pseudo = int(self.top_k * self.pseudo_ratio)
        if k_pseudo <= 0: return
        
        # 获取 Dataset 级别的 debug 标志 (默认 False)
        # debug_mode = getattr(self, 'debug', False)
        debug_mode = True
        
        print(f"🧬 [Dataset] 启动物理级扰动生成伪造样本 (Ratio: {self.pseudo_ratio}, Strength: {self.aug_strength}, Debug: {debug_mode})")
        
        # =========================================================================
        # 🐛 [核心新增] Debug 模式 / Strength=0 快速通道：直接复制目标样本，跳过 TSFM 预测
        # =========================================================================
        if debug_mode or self.aug_strength <= 0:
            print("   ⏩ [Fast-Path] 启用直接复制模式：将目标样本原样作为伪邻居，跳过 TSFM 推理。")
            for idx in tqdm(range(len(self.samples)), desc="Copying Original"):
                self.pseudo_pool[idx] = {'embs': [], 'err_embs': [], 'res': []}
                
                # 获取原样本已算好的序列指纹、误差指纹、真实残差 (维度皆为 [1, Dim])
                orig_emb = self.precomputed_embeddings[idx] if self.precomputed_embeddings else torch.zeros(1, getattr(self.db.encoder, 'embedding_dim', 128))
                orig_err_emb = self.precomputed_err_embeddings[idx] if self.precomputed_err_embeddings else torch.zeros(1, getattr(self.db.encoder, 'embedding_dim', 128))
                orig_res = self.scaled_res_tensors[idx] # 在初始化时生成的 (1, pred_len)
                # print(f"   样本 {idx}: 原残差 (前5值) = {orig_res[:5].cpu().numpy()}, 规模 = {self.scales[idx]:.4f}")
                for _ in range(k_pseudo):
                    self.pseudo_pool[idx]['embs'].append(orig_emb)
                    self.pseudo_pool[idx]['err_embs'].append(orig_err_emb)
                    self.pseudo_pool[idx]['res'].append(orig_res)
            return
        # =========================================================================

        if self.tsfm_predictor is None:
            print("   ⚠️ 警告: 未提供 tsfm_predictor，将使用备用高斯噪声合成逻辑。")
            return
            
        all_mock_entries = []
        mapping_info = []

        # 1. 批量生成所有扰动序列并打包为字典
        for idx, item in enumerate(tqdm(self.samples, desc="Perturbing Series")):
            self.pseudo_pool[idx] = {'embs': [], 'err_embs': [], 'res': []}
            hist = item.get('history', [])
            truth = item.get('truth', [])
            freq = self.freqs[idx] if self.freqs[idx] else "H"
            
            # 传入 debug 参数
            p_hists, p_truths = generate_consistent_perturbations(hist, truth, k_pseudo, self.aug_strength, debug=debug_mode)
            
            for p_idx, (p_h, p_t) in enumerate(zip(p_hists, p_truths)):
                entry = {
                    'target': p_h,
                    'start': pd.Period("2020-01-01", freq=freq),
                    'item_id': f"pseudo_{idx}_{p_idx}"
                }
                all_mock_entries.append(entry)
                mapping_info.append({'sample_idx': idx, 'truth': p_t, 'hist': p_h})

        if not all_mock_entries: return

        # 2. 构造兼容 Predictor 的内存 Dataset
        mem_dataset = _DummyMemDataset(all_mock_entries, pred_len=self.pred_len, freq=self.freqs[0] if self.freqs else "H")

        # =========================================================================
        # 3. TSFM 批量推理 (优雅复用 BaseModel._make_forecasts)
        # =========================================================================
        print(f"   🧠 [TSFM] 正在对 {len(all_mock_entries)} 个伪造序列进行前向预测...")
        
        # 临时借用/设置 tsfm_predictor 的 batch_size (避免初始值过大引发多次 OOM 重试)
        original_bs = getattr(self.tsfm_predictor, 'batch_size', 256)
        self.tsfm_predictor.batch_size = min(original_bs, 256) 
        
        # 直接调用 BaseModel 的统一预测流程
        forecasts, _ = self.tsfm_predictor._make_forecasts(
            dataset=mem_dataset,
            dataset_name="pseudo_data",          # 用于辨识的虚拟名称
            ds_config="Pseudo_Generation",       # tqdm 进度条显示的名称
            fixed_model_order=None,              # 非 Select 模式无需传入
            debug_mode=False
        )
        
        # 恢复原本的 batch_size
        self.tsfm_predictor.batch_size = original_bs
        # =========================================================================

        # 4. 提取与张量化
        pseudo_hist_tensors = []
        pseudo_res_tensors = []
        
        for info, fcst in tqdm(zip(mapping_info, forecasts), total=len(forecasts), desc="Extracting Residuals"):
            # A. 提取点预测
            if hasattr(fcst, "samples"): # SampleForecast (如 Chronos, TiRex)
                pred = np.mean(fcst.samples, axis=0)
            elif hasattr(fcst, "mean"):  # DistributionForecast
                pred = fcst.mean
            elif hasattr(fcst, "forecast_array"): # QuantileForecast (如 TimesFM)
                pred = np.mean(fcst.forecast_array, axis=0)
            elif isinstance(fcst, torch.Tensor):
                pred = fcst.mean(dim=0).cpu().numpy()
            else:
                pred = np.zeros(self.pred_len)
            
            # B. 长度对齐
            if len(pred) < self.pred_len: pred = np.pad(pred, (0, self.pred_len - len(pred)))
            pred = pred[:self.pred_len]
            
            p_truth = info['truth'][:self.pred_len]
            if len(p_truth) < self.pred_len: p_truth = np.pad(p_truth, (0, self.pred_len - len(p_truth)))
            
            # C. 计算物理误差残差
            p_res = p_truth - pred
            
            # D. 处理为 Tensor
            h_t = self._process_tensor(info['hist'], self.context_len, is_history=True)
            r_t = self._process_tensor(p_res, self.pred_len)
            
            # 共享原样本的 Scale
            scale = self.scales[info['sample_idx']]
            pseudo_hist_tensors.append(h_t / scale)
            pseudo_res_tensors.append(torch.clamp(r_t / scale, -20.0, 20.0))

        # 5. 批量抽取双重指纹 (Embedding)
        print("   🧬 [Encoder] 正在抽取伪样本双重指纹...")
        enc_device = getattr(self.db.encoder, 'device', 'cpu')
        all_hists_t = torch.cat(pseudo_hist_tensors, dim=0)
        all_res_t = torch.cat(pseudo_res_tensors, dim=0)
        
        pseudo_embs = []
        pseudo_err_embs = []
        enc_bs = 512
        
        with torch.no_grad():
            # 序列指纹
            for i in tqdm(range(0, len(all_hists_t), enc_bs), desc="Encoding Pseudo Seq"):
                batch = all_hists_t[i:i+enc_bs].to(enc_device)
                embs = self.db.encoder.encode(batch).cpu()
                if embs.dim() == 3: embs = embs.squeeze(1)
                pseudo_embs.extend(torch.split(embs, 1, dim=0))
                
            # 误差指纹
            for i in tqdm(range(0, len(all_res_t), enc_bs), desc="Encoding Pseudo Err"):
                batch = all_res_t[i:i+enc_bs].to(enc_device)
                if batch.shape[1] < self.context_len:
                    batch = torch.nn.functional.pad(batch, (self.context_len - batch.shape[1], 0))
                elif batch.shape[1] > self.context_len:
                    batch = batch[:, -self.context_len:]
                err_e = self.db.encoder.encode(batch).cpu()
                if err_e.dim() == 3: err_e = err_e.squeeze(1)
                pseudo_err_embs.extend(torch.split(err_e, 1, dim=0))

        # 6. 保存到 Pool 供 __getitem__ 直接拿取
        for i, info in enumerate(mapping_info):
            s_idx = info['sample_idx']
            self.pseudo_pool[s_idx]['embs'].append(pseudo_embs[i])
            self.pseudo_pool[s_idx]['err_embs'].append(pseudo_err_embs[i])
            self.pseudo_pool[s_idx]['res'].append(pseudo_res_tensors[i])           
    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _process_tensor(self, arr, target_len, is_history=False):
        """通用序列处理"""
        if arr is None or len(arr) == 0:
            return torch.zeros(1, target_len)
            
        t_np = np.array(arr, dtype=np.float32).reshape(-1)
        t = torch.from_numpy(t_np).unsqueeze(0)
        
        if t.shape[1] < target_len:
            pad = (target_len - t.shape[1], 0) if is_history else (0, target_len - t.shape[1])
            t = torch.nn.functional.pad(t, pad)
        else:
            t = t[:, -target_len:] if is_history else t[:, :target_len]
        return t

    def _get_relative_residual(self, res_arr, truth_arr):
        res_t = self._process_tensor(res_arr, self.pred_len)
        truth_t = self._process_tensor(truth_arr, self.pred_len)
        return res_t, truth_t
        
    def _generate_fallback_pseudo_samples(self, target_emb, target_err_emb, target_res, num_pseudo):
        """备用方案: 简单高斯噪声合成"""
        if num_pseudo <= 0:
            return (torch.empty(0, target_emb.shape[-1]), 
                    torch.empty(0, target_err_emb.shape[-1]), 
                    torch.empty(0, target_res.shape[-1]))

        pseudo_embs = target_emb.repeat(num_pseudo, 1)
        pseudo_err_embs = target_err_emb.repeat(num_pseudo, 1)
        pseudo_res = target_res.repeat(num_pseudo, 1)

        if self.aug_strength > 0:
            emb_noise = torch.randn_like(pseudo_embs) * self.aug_strength
            pseudo_embs = pseudo_embs + emb_noise

            err_noise = torch.randn_like(pseudo_err_embs) * (self.aug_strength * 0.8)
            pseudo_err_embs = pseudo_err_embs + err_noise

            scales = (torch.rand(num_pseudo, 1) * self.aug_strength*0.1 + 1 - self.aug_strength*0.05)
            res_noise = torch.randn_like(pseudo_res) * (self.aug_strength * 0.5)
            pseudo_res = pseudo_res * scales + res_noise

        return pseudo_embs, pseudo_err_embs, pseudo_res

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # 数据提取入口 (极速 + 门控高阶特征提取)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        hist_t = self.history_tensors[idx]
        target_res_scaled = self.scaled_res_tensors[idx] 
        truth_t = self.truth_tensors[idx]
        scale_t = self.scales[idx]
        valid_len = self.valid_lens[idx]
        local_res_t = self.local_res_norm_tensors[idx]
        
        # 1. 自身 Embedding
        sample_emb = self.precomputed_embeddings[idx] if self.precomputed_embeddings else torch.zeros(1, 128)
        sample_err_emb = self.precomputed_err_embeddings[idx] if self.precomputed_err_embeddings else torch.zeros(1, 128)

        k_total = self.top_k
        k_pseudo = int(k_total * self.pseudo_ratio) if self.pseudo_ratio > 0.0 else 0
        k_real = k_total - k_pseudo

        # 2. 获取真实邻居 (从 Cache 极速读取)
        real_embs = torch.zeros((0, sample_emb.shape[-1]))
        real_err_embs = torch.zeros((0, sample_err_emb.shape[-1]))
        real_res = torch.zeros((0, self.pred_len))
        
        if k_real > 0 and idx < len(self.retrieved_embs_cache):
            cached_embs = self.retrieved_embs_cache[idx]
            cached_res = self.retrieved_residuals_cache[idx]
            cached_err_embs = self.retrieved_err_embs_cache[idx] if idx < len(self.retrieved_err_embs_cache) else None
            
            if cached_embs.shape[0] > 0:
                limit = min(cached_embs.shape[0], k_real)
                real_embs = cached_embs[:limit]
                real_res = cached_res[:limit]
                
                if cached_err_embs is not None and cached_err_embs.shape[0] > 0:
                    real_err_embs = cached_err_embs[:limit]
                else:
                    real_err_embs = torch.zeros((limit, sample_err_emb.shape[-1]))

        # 3. 获取伪造邻居 (从 precomputed pool 读取)
        if k_pseudo > 0:
            if self.tsfm_predictor is not None and idx in self.pseudo_pool and len(self.pseudo_pool[idx]['embs']) > 0:
                pseudo_embs = torch.cat(self.pseudo_pool[idx]['embs'][:k_pseudo], dim=0)
                pseudo_err_embs = torch.cat(self.pseudo_pool[idx]['err_embs'][:k_pseudo], dim=0)
                pseudo_res = torch.cat(self.pseudo_pool[idx]['res'][:k_pseudo], dim=0)
            else:
                pseudo_embs, pseudo_err_embs, pseudo_res = self._generate_fallback_pseudo_samples(
                    sample_emb, sample_err_emb, target_res_scaled, k_pseudo)
        else:
            pseudo_embs = torch.zeros((0, sample_emb.shape[-1]))
            pseudo_err_embs = torch.zeros((0, sample_err_emb.shape[-1]))
            pseudo_res = torch.zeros((0, self.pred_len))
        
        # 4. 拼接 & 打乱 & 补齐
        combined_embs = torch.cat([real_embs, pseudo_embs], dim=0)
        combined_err_embs = torch.cat([real_err_embs, pseudo_err_embs], dim=0)
        combined_residuals = torch.cat([real_res, pseudo_res], dim=0)

        if self.shuffle_order:
            curr_count = combined_embs.shape[0]
            if curr_count > 1:
                perm = torch.randperm(curr_count)
                combined_embs = combined_embs[perm]
                combined_err_embs = combined_err_embs[perm]
                combined_residuals = combined_residuals[perm]

        curr_k = combined_embs.shape[0]
        if curr_k < self.top_k:
            pad_k = self.top_k - curr_k
            emb_dim = sample_emb.shape[-1]
            combined_embs = torch.cat([combined_embs, torch.zeros((pad_k, emb_dim))], dim=0)
            combined_err_embs = torch.cat([combined_err_embs, torch.zeros((pad_k, emb_dim))], dim=0)
            combined_residuals = torch.cat([combined_residuals, torch.zeros((pad_k, self.pred_len))], dim=0)
        
        combined_embs = combined_embs[:self.top_k]
        combined_err_embs = combined_err_embs[:self.top_k]
        combined_residuals = combined_residuals[:self.top_k]

        # ==================================================================
        # 🌟 5. [新增] 为门控模块提取时频物理与误差分布特征 (兼容版)
        # ==================================================================
        # 提取标量版的 scale，防止除零 (兼容 float, numpy, tensor)
        if isinstance(scale_t, torch.Tensor):
            s_val = scale_t.flatten()[0].item() + 1e-8
        elif isinstance(scale_t, np.ndarray):
            s_val = scale_t.flatten()[0] + 1e-8
        else:
            s_val = float(scale_t) + 1e-8
            
        # 安全地将输入转换为一维 Tensor
        local_res_ts = torch.as_tensor(local_res_t, dtype=torch.float32).view(-1)
        h_flat = torch.as_tensor(hist_t, dtype=torch.float32).view(-1)
        
        # --- (1) 目标序列 Local Residual 特征 (相对 Scale 的绝对波动) ---
        norm_local_res = local_res_ts / s_val
        res_mean_abs = torch.mean(torch.abs(norm_local_res))
        res_max_abs = torch.max(torch.abs(norm_local_res))
        res_std = torch.std(norm_local_res) if norm_local_res.numel() > 1 else torch.tensor(0.0)
        
        # --- (2) 目标序列 History 物理与动力学特征 ---
        h_std = torch.std(h_flat) if h_flat.numel() > 1 else torch.tensor(0.0)
        diff1 = torch.diff(h_flat)
        trend_proxy = torch.mean(diff1) if diff1.numel() > 0 else torch.tensor(0.0)
        complexity = torch.mean(torch.abs(diff1)) if diff1.numel() > 0 else torch.tensor(0.0)
        
        # --- (3) 目标序列 频域特征 (快速傅里叶变换寻找主导频率) ---
        h_centered = h_flat - torch.mean(h_flat)
        fft_vals = torch.abs(torch.fft.rfft(h_centered))
        if len(fft_vals) > 1:
            dom_freq = torch.max(fft_vals[1:]) / (torch.sum(fft_vals[1:]) + 1e-8)
        else:
            dom_freq = torch.tensor(0.0)
            
        # --- (4) 上下文邻居序列特征 (Neighbor Residuals) ---
        context_res_mean_abs = torch.mean(torch.abs(combined_residuals))
        
        # =======================================================
        # 🌟 (5) 更丰富的历史序列 (History) 物理/统计特征 (新增特征)
        # =======================================================
        if len(h_flat) > 1:
            # 10. 历史序列高频抖动 (一阶差分绝对均值)
            diff1_new = torch.diff(h_flat)
            diff_mad = torch.mean(torch.abs(diff1_new)) / s_val
            
            # 11. 历史序列 Lag-1 自相关性 (衡量趋势惯性)
            h_mean = torch.mean(h_flat)
            h_centered_new = h_flat - h_mean
            var = torch.sum(h_centered_new ** 2)
            autocorr_lag1 = torch.sum(h_centered_new[:-1] * h_centered_new[1:]) / var if var > 1e-5 else torch.tensor(0.0)
            
            # 12. 均值穿越率 (Zero-Crossing Rate, 反映震荡频率)
            zcr = torch.sum(h_centered_new[:-1] * h_centered_new[1:] < 0).float() / len(h_flat)
        else:
            diff_mad, autocorr_lag1, zcr = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            
        # 13. 历史序列极差 (Peak-to-Peak Range，反映波动边界)
        h_ptp = (torch.max(h_flat) - torch.min(h_flat)) / s_val
        
        # 14. 近期趋势偏移 (末端 10% 的均值偏离全局均值的程度，捕捉突变)
        recent_len = max(1, int(len(h_flat) * 0.1))
        recent_mean = torch.mean(h_flat[-recent_len:])
        recent_trend_shift = (recent_mean - torch.mean(h_flat)) / s_val

        # 🌟 组装为 14 维密集特征向量
        ts_physics_features = torch.stack([
            res_mean_abs,         # 1. 局部残差平均绝对幅度|验证窗口上除掉尺度后的      残差的平均值
            res_max_abs,          # 2. 局部残差最大偏离度|验证窗口上除掉尺度后的残差的  最大绝对值     
            res_std,              # 3. 局部残差波动率|验证窗口上除掉尺度后的残差的      标准差
            h_std / s_val,        # 4. 历史序列相对波动率|历史窗口的标准差除以尺度（归一化）
            trend_proxy / s_val,  # 5. 历史序列趋势 (归一化)|历史窗口的一阶差分均值除以尺度
            complexity / s_val,   # 6. 历史序列复杂度 (归一化)|历史窗口的一阶差分绝对均值除以尺度
            dom_freq,             # 7. 频域主导能量占比|快速傅里叶变换后最大振幅除以振幅总和|信号波动是否有规律、是否由单一频率主导（提取最大值与总和的比例）
            context_res_mean_abs, # 8. 检索邻居的残差平均幅度|邻居序列未来窗口残差均值（标量）
            
            torch.tensor(math.log1p(s_val), dtype=torch.float32), # 9. 目标历史绝对 Scale 的对数，衡量尺度级别（对数变换后更适合模型处理）
            diff_mad,             # 🌟 10. 目标历史高频抖动|目标序列历史的一阶差分绝对均值
            autocorr_lag1,        # 🌟 11. 目标历史自相关性|序列中当前时刻的值，与紧挨着它的下一个时刻的值，有多大程度的相关性
            zcr,                  # 🌟 12. 目标历史均值穿越率|序列在均值上下穿过的频率
            h_ptp,                # 🌟 13. 目标历史极差 极差除以均值
            recent_trend_shift    # 🌟 14. 近期局部突变偏移|目标历史末端 10% 的均值偏离全局均值的程度，捕捉突变
        ]).float()

        return {
            "history": hist_t,
            "truth": truth_t,
            "target_residual": target_res_scaled,
            "local_residual": local_res_t, 
            "retrieved_context": combined_embs,       
            "retrieved_err_context": combined_err_embs, 
            "retrieved_residual": combined_residuals, 
            "sample_embedding": sample_emb,           
            "sample_err_embedding": sample_err_emb,   
            "scale": scale_t,
            "valid_len": valid_len,
            "smape_quantile": self.smape_quantiles[idx],
            "ts_physics_features": ts_physics_features # 14维特征供门控使用
        }