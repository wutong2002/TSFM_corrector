import importlib
import os
import pickle
import re
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from collections import defaultdict, Counter
import torch.optim.lr_scheduler as lr_scheduler
import csv
import time
from types import SimpleNamespace
from torch.utils.data import Subset

# === 1. 引入依赖 ===
try:
    from utils.missing import fill_missing
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.missing import fill_missing

# === 2. 引入核心组件 ===
from database.manager import SchoolwareDB
from database.dataset import CorrectionDataset
from retriever.engine import ExactCosineRetriever, DualMetricRetriever
from retriever.strategies import (
    GlobalScopeStrategy, DatasetScopeStrategy, 
    CausalTimeScopeStrategy, CrossDatasetScopeStrategy
)

def dynamic_load_model_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

# === 同模型检索策略 ===
class SameModelScopeStrategy:
    def filter(self, metadata, target_meta):
        target_model = target_meta.get('source_model')
        if target_model is None:
            return list(range(len(metadata)))
        return [i for i, m in enumerate(metadata) if m.get('source_model') == target_model]

try:
    from Model_Path.model_zoo_config import Model_zoo_details
except ImportError:
    print("❌ 错误: 无法导入 Model_Path.model_zoo_config")
    exit(1)

# === 3. 引入所有编码器 (动态检查) ===
try:
    from encoder.units_encoder import UnitsEncoder
except ImportError: UnitsEncoder = None

try:
    from encoder.timesfm_encoder import TimesFMEncoder
except ImportError: TimesFMEncoder = None

try:
    from encoder.moirai_encoder import MoiraiEncoder
except ImportError: MoiraiEncoder = None

try:
    from encoder.math_encoders import FFTEncoder, StatisticsEncoder, WaveletEncoder, HybridMathEncoder, BalancedHybridEncoder, RandomNNEncoder
except ImportError: 
    FFTEncoder = StatisticsEncoder = WaveletEncoder = HybridMathEncoder = BalancedHybridEncoder = None

CONFIG = {
    "TARGET_FAMILIES": ["moirai"],  
    "TARGET_VARIANTS": [], 
    "RAW_DATA_ROOT": os.path.join("Datasets", "raw_data"), 
    "OUTPUT_ROOT": "correction_datasets_double_res",              
    "CONTEXT_LEN": 512,           
    "STRIDE": 32,                 
    "MAX_SAMPLES_PER_DS": 2000,   
    "MAX_WINDOWS_PER_SEQ": 50,    
    "BATCH_SIZE": 128,
    "GPU_ID": "0",
    "PROPERTIES_PATH": os.path.join("Datasets", "processed_datasets", "dataset_properties.json"),
}

def build_complete_args():
    """构造满足 BaseModel 所需的完整参数"""
    args = SimpleNamespace()
    args.output_dir = os.path.join("results", "inference_gen_temp") 
    args.output_root = CONFIG["OUTPUT_ROOT"] 
    args.run_mode = "zoo"                    
    args.save_pred = False
    args.save_for_correction = True
    args.skip_saved = False
    args.debug_mode = False
    args.context_len = CONFIG["CONTEXT_LEN"]
    args.fix_context_len = False
    args.batch_size = CONFIG["BATCH_SIZE"]
    args.num_workers = 0
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args
class StaticThresholdGate:
    def __init__(self, threshold=0.15):
        # 假设我们通过之前分析得出：当归一化残差 > 0.15 时，才有修正价值
        self.threshold = threshold 
        
    def __call__(self, target_local_res, scale):
        """返回 0/1 Mask"""
        norm_res = torch.abs(target_local_res / (scale + 1e-8))
        res_magnitude = torch.mean(norm_res, dim=-1) # (B,)
        mask = (res_magnitude >= self.threshold).float()
        return mask.unsqueeze(-1) # (B, 1)
class HybridFrequencyLoss(nn.Module):
    """
    专为打破“谱偏见 (Spectral Bias)”设计的高频时间序列修正损失函数。
    """
    def __init__(self, alpha=1.0, beta=0.2, gamma=0.5, delta=1.0):
        super().__init__()
        self.alpha = alpha   
        self.beta = beta     
        self.gamma = gamma   
        self.huber = nn.HuberLoss(delta=delta, reduction='none')

    def forward(self, pred, target):
        B, L = pred.shape
        loss_time = self.huber(pred, target) 
        
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_cosine = -torch.nn.functional.cosine_similarity(pred_diff, target_diff, dim=1)
        loss_shape = diff_cosine.unsqueeze(1).expand(B, L)
        
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        fft_loss_raw = torch.abs(pred_amp - target_amp)
        loss_freq = fft_loss_raw.mean(dim=1, keepdim=True).expand(B, L) 
        
        total_loss_matrix = self.alpha * loss_time + self.beta * loss_freq + self.gamma * loss_shape
        return total_loss_matrix

class CorrectionTrainer:
    def __init__(self, model, model_config: dict, db_config: dict, train_config: dict, preloaded_data=None):
        self.model = model
        self.model_config = model_config 
        self.db_config = db_config
        self.train_config = train_config
        
        self.tsfm_predictor = None
        self.logger = train_config.get('logger', None)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = self.device.type
        self.model = self.model.to(self.device)
        
        self.scaler = GradScaler(device=self.device_type, enabled=(self.device_type == 'cuda'))
        
        self.patience = self.train_config.get('early_stop_patience', 10)
        self.optimizer_type = self.train_config.get('optimizer', 'adam')
        self.scheduler_type = self.train_config.get('scheduler', 'cosine') 
        self.weight_decay = self.train_config.get('weight_decay', 1e-4)
        self.debug = bool(self.train_config.get('debug', 0))
        self.enable_metrics = True
        self.max_grad_norm = self.train_config.get('max_grad_norm', 5.0)
        
        self.train_samples = [] 
        self.test_samples_dict = defaultdict(list) 
        self.db = None

        self.train_ds = None
        self.val_loaders = None
        self.train_loader = None   
        self.preloaded_data = preloaded_data
        
        self.log(f"🚀 [Trainer] Device: {self.device} | Patience: {self.patience}")
        self._init_components()
        # self._scan_and_adapt_lengths()


    def log(self, message):
        if self.logger: self.logger.info(message)
        else: print(message)

    def _make_tsfm_predictor(self):
        if self.train_config.get('pseudo_method', "tsfm") == "random": return None
        if self.tsfm_predictor is not None: return self.tsfm_predictor

        target = self.train_config.get('target_tsfm_filter', None)
        if not target: return None
        target_lower = target.lower()

        chosen_model = None
        for family, variants in Model_zoo_details.items():
            if family.lower() not in target_lower and not any(variant.lower() in target_lower for variant in variants):
                continue
            for variant, info in variants.items():
                if variant.lower() in target_lower or family.lower() in target_lower:
                    model_full_name = f"{family}_{variant}"
                    print(f"\n🚀 [Loading TSFM model] {model_full_name} for filter '{target}'")
                    try:
                        ModelClass = dynamic_load_model_class(info["model_module"], info["model_class"])
                        args = build_complete_args()
                        args.model_family = family
                        args.model_size = variant
                        chosen_model = ModelClass(
                            args=args, module_name=info["module_name"],
                            model_name=model_full_name, model_local_path=info["model_local_path"]
                        )
                        self.tsfm_predictor = chosen_model 
                        return chosen_model
                    except Exception as e:
                        print(f"❌ TSFM 初始化失败 ({model_full_name}): {e}")
                        
        if chosen_model is None: print(f"⚠️ 未能加载 TSFM 模型, filter='{target}' 在 Model_zoo_details 中未匹配")
        return chosen_model
            
    def _init_components(self):
        if self.preloaded_data is not None:
            self.log("♻️ [Cache] 检测到预加载数据...")
            if len(self.preloaded_data) == 5:
                self.db, self.train_samples, self.test_samples_dict, self.train_ds, self.val_loaders = self.preloaded_data
            elif len(self.preloaded_data) == 3:
                self.db, self.train_samples, self.test_samples_dict = self.preloaded_data
            return

        div_max = self.train_config.get('diversity_max_per_dataset', 100)
        retriever = DualMetricRetriever(device=self.device, max_per_dataset=div_max)
        
        # 🌟 动态下发参数 3：度量机制给底层 retriever
        retriever.err_metric = self.train_config.get('err_sim_metric', 'l2')
        self.log(f"🛠️ 误差检索度量机制: {retriever.err_metric.upper()}")
        
        enc_type = self.db_config.get('encoder_type', 'units')
        ckpt_path = self.db_config.get('ckpt_path', None)
        ctx_len = self.db_config.get('context_len', 96)
        patch_size = self.db_config.get('patch_size', 32)
        output_dim = self.db_config.get('output_dim', 128)
        
        self.log(f"🛠️ 初始化编码器: [{enc_type}]")
        
        if enc_type == 'units': encoder = UnitsEncoder(ckpt_path=ckpt_path, context_len=ctx_len, device=self.device)
        elif enc_type == 'timesfm': encoder = TimesFMEncoder(ckpt_path=ckpt_path, context_len=ctx_len, device=self.device)
        elif enc_type == 'moirai': encoder = MoiraiEncoder(ckpt_path=ckpt_path, context_len=ctx_len, device=self.device, patch_size=patch_size)
        elif enc_type == 'fft': encoder = FFTEncoder(output_dim=output_dim)
        elif enc_type == 'stats': encoder = StatisticsEncoder(output_dim=output_dim)
        elif enc_type == 'wavelet': encoder = WaveletEncoder(output_dim=output_dim)
        elif enc_type == "advanced_hybrid_math": encoder = BalancedHybridEncoder()
        elif enc_type == 'hybrid_math': encoder = HybridMathEncoder(output_dim=output_dim, wavelet='db4', level=3)
        elif enc_type == 'random_nn_frozen': encoder = RandomNNEncoder(output_dim=output_dim)
        else: raise ValueError(f"❌ 未知的编码器类型: {enc_type}")
            
        self.db = SchoolwareDB(encoder, retriever, debug=self.debug)
        self._load_and_ingest_data()

    def _scan_and_adapt_lengths(self):
        all_samples = self.train_samples + [s for samples in self.test_samples_dict.values() for s in samples]
        if not all_samples: return

        max_len = 0
        for s in all_samples:
            v_len = s.get('valid_len', len(s['residual']))
            if v_len > max_len: max_len = v_len
            
        current_cfg_len = self.model_config.get('pred_len', 96)
        
        if max_len > current_cfg_len:
            self.log(f"📏 检测到数据最大长度 ({max_len}) > 配置 ({current_cfg_len})，正在调整模型...")
            self.model_config['pred_len'] = max_len
            target_len = max_len
            
            if hasattr(self.model, 'output_head') and isinstance(self.model.output_head, nn.Linear):
                old = self.model.output_head
                new_head = nn.Linear(old.in_features, target_len).to(old.weight.device)
                nn.init.constant_(new_head.weight, 0.0)
                nn.init.constant_(new_head.bias, 0.0)
                self.model.output_head = new_head
                
            if hasattr(self.model, 'value_proj'):
                if isinstance(self.model.value_proj, nn.Sequential) and isinstance(self.model.value_proj[0], nn.Linear):
                    old = self.model.value_proj[0]
                    if old.in_features != target_len:
                        new_proj = nn.Linear(target_len, old.out_features).to(old.weight.device)
                        nn.init.xavier_uniform_(new_proj.weight)
                        self.model.value_proj[0] = new_proj
                elif isinstance(self.model.value_proj, nn.Linear):
                    old = self.model.value_proj
                    if old.in_features != target_len:
                        new_proj = nn.Linear(target_len, old.out_features).to(old.weight.device)
                        nn.init.xavier_uniform_(new_proj.weight)
                        self.model.value_proj = new_proj
            
            if hasattr(self.model, 'pred_len'): self.model.pred_len = target_len
            self.log(f"✅ 模型已适配新的 pred_len: {target_len}")

    def _get_optimizer(self):
        if len(list(self.model.parameters())) == 0: return None
        # 兼容两套命名：优先读取网格配置里的 learning_rate
        lr = self.train_config.get('learning_rate', self.train_config.get('lr', 1e-4))
        wd = self.weight_decay
        if self.optimizer_type == 'adamw': return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.optimizer_type == 'adam': return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        return optim.Adam(self.model.parameters(), lr=lr)

    def _get_scheduler(self, optimizer):
        if optimizer is None: return None
        epochs = self.train_config.get('epochs', 100)
        warmup_epochs = self.train_config.get('warmup_epochs', 5)
        main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=1e-6)
        if warmup_epochs > 0:
            warmup = lr_scheduler.LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_epochs)
            return lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_epochs])
        return main_scheduler

    def _impute_series(self, arr):
        if len(arr) == 0: return arr
        arr = np.array(arr, dtype=np.float32).reshape(-1)
        if np.isnan(arr).any(): return fill_missing(arr, all_nan_strategy_1d="zero", interp_kind_1d="linear")
        return arr

    def _load_and_ingest_data(self):
        data_root = self.db_config.get('data_dir')
        raw_train_list = self.train_config.get('train_datasets_list', [])
        raw_test_list = self.train_config.get('test_datasets_list', [])
        target_tsfm_filter = self.train_config.get('target_tsfm_filter', None) 
        
        max_train = self.train_config.get('max_samples_per_dataset', -1)
        max_test = self.train_config.get('max_test_samples_per_dataset', -1)
        split_mode = self.train_config.get('train_test_split_mode', 'cross_dataset')
        
        # 构建大池子
        train_targets = [t.lower() for t in raw_train_list]
        test_targets = [t.lower() for t in raw_test_list]
        all_targets = set(train_targets + test_targets)
        
        tsfm_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        loaded_items_by_ds = defaultdict(list)
        stats = defaultdict(int)
        
        self.log(f"🔍 模式 [{split_mode.upper()}] 正在加载数据...")
        
        # 1. 把相关的数据全部载入内存大池子
        loaded_items_by_ds = defaultdict(list)
        stats = defaultdict(int)
        
        self.log(f"🔍 模式 [{split_mode.upper()}] 正在加载数据...")
        
        # 🌟 1. 新增：初始化数据过滤统计
        self.data_filter_stats = Counter()
        total_raw = 0
        
        # 1. 把相关的数据全部载入内存大池子
        for tsfm in tsfm_dirs:
            if target_tsfm_filter and tsfm.lower() != target_tsfm_filter.lower(): continue
            model_path = os.path.join(data_root, tsfm)
            
            pkl_files = []
            for root, _, files in os.walk(model_path):
                if "correction_data" in root:
                    for file in files:
                        if file.endswith('.pkl'): pkl_files.append(os.path.join(root, file))
            
            for pkl_path in pkl_files:
                try:
                    with open(pkl_path, 'rb') as fp: data = pickle.load(fp)
                except Exception: continue
                
                clean_name = os.path.basename(pkl_path).lower().replace('_correction_data.pkl', '').replace('.pkl', '')
                if clean_name not in all_targets: continue
                
                histories = data.get('histories', [])
                residuals = data.get('residuals', [])
                truths = data.get('truths', [])
                local_residuals = data.get('local_residuals', [])
                smape_quantiles = data.get('smape_quantiles', [])
                file_metadata = data.get('metadata', {}) 
                sample_metadata = data.get('sample_metadata', [])
                
                count = min(len(histories), len(residuals))
                for i in range(count):
                    total_raw += 1 # 累加读取总数
                    raw_l_r = local_residuals[i] if local_residuals is not None and i < len(local_residuals) else None
                    
                    # 🚨 规则 A: 拦截空值与过短序列
                    if raw_l_r is None or len(raw_l_r) < 2:
                        self.data_filter_stats["缺失或过短的局部残差 (len < 2)"] += 1
                        continue
                        
                    h = self._impute_series(histories[i])
                    r = self._impute_series(residuals[i])
                    t = self._impute_series(truths[i])
                    l_r = self._impute_series(raw_l_r)

                    # 🚨 规则 B: 严格的 NaN / Inf 检查
                    if np.isnan(h).any() or np.isnan(r).any() or np.isnan(t).any() or np.isnan(l_r).any():
                        self.data_filter_stats["序列包含 NaN"] += 1
                        continue
                    if np.isinf(h).any() or np.isinf(r).any() or np.isinf(t).any() or np.isinf(l_r).any():
                        self.data_filter_stats["序列包含 Inf (无限大值)"] += 1
                        continue

                    # 🚨 规则 C: 极值探测与常数序列检查
                    h_f64 = h.astype(np.float64)
                    l_r_f64 = l_r.astype(np.float64)
                    
                    if np.max(np.abs(h_f64)) > 1e6 or np.max(np.abs(l_r_f64)) > 1e6:
                        self.data_filter_stats["极端数值 (Absolute Value > 1e6)"] += 1
                        continue 

                    if np.std(h_f64) < 1e-5 or np.std(l_r_f64) < 1e-5:
                        self.data_filter_stats["方差过小/常数序列 (Std < 1e-5)"] += 1
                        continue

                    # ========================================
                    # 通过所有安检，正常放行
                    # ========================================
                    h_abs = np.abs(h)
                    valid_mask = h_abs > 1e-9
                    scale = max(np.mean(h_abs[valid_mask]) if np.sum(valid_mask) > 0 else 1.0, 1e-6)
                    
                    scaled_r_clamped = np.clip(r / scale, -20.0, 20.0) 
                    clamped_raw_r = scaled_r_clamped * scale
                    meta = sample_metadata[i] if i < len(sample_metadata) else {}
                    
                    item = {
                        'history': h, 'history_norm': h / scale,
                        'residual': clamped_raw_r, 'scaled_residual': scaled_r_clamped,
                        'truth': t - (r - clamped_raw_r),
                        'local_residual': l_r, 'local_residual_norm': l_r / scale,
                        'dataset': clean_name, 'source_model': tsfm,
                        'valid_len': len(residuals[i]),
                        'freq': file_metadata.get('freq', 'unknown'),
                        'domain': file_metadata.get('domain', 'generic'),
                        'smape_quantile': float(smape_quantiles[i]) if len(smape_quantiles) > i else 100.0,
                        'sample_meta': meta
                    }
                    loaded_items_by_ds[clean_name].append((tsfm, item))
                    
        # ====================================================================
        # 🌟🌟🌟 [核心修复] 动态计算样本难度分位数 (sMAPE Quantiles) 🌟🌟🌟
        # 解决 .pkl 中未保存 smape_quantiles 导致全部默认 100.0 的 Bug
        # ====================================================================
        self.log("🧮 正在动态计算样本难度分位数 (sMAPE Quantiles)...")
        
        # 必须按 TSFM 基座模型分别计算，因为不同模型的误差分布基准不同
        for tsfm_target in tsfm_dirs:
            tsfm_items = []
            for ds_name, items_list in loaded_items_by_ds.items():
                for m, item in items_list:
                    if m == tsfm_target:
                        tsfm_items.append(item)
                        
            if not tsfm_items: continue
            
            smapes = []
            for item in tsfm_items:
                # 真实值 - 残差 = 原始基础预测值
                base_pred = item['truth'] - item['residual'] 
                abs_p, abs_t = np.abs(base_pred), np.abs(item['truth'])
                denom = abs_p + abs_t
                valid_mask = denom > 1e-5
                # 计算这个样本的 Base sMAPE
                smape_num = np.sum(np.where(valid_mask, 200 * np.abs(base_pred - item['truth']) / denom, 0.0))
                smape_den = np.sum(valid_mask)
                smapes.append(smape_num / max(smape_den, 1e-8))
                
            smapes = np.array(smapes)
            
            # 计算百分位排名 (0~100)
            # argsort 两次可以得到每个元素的排名，0 代表误差最小，len-1 代表误差最大
            ranks = np.empty_like(smapes)
            ranks[np.argsort(smapes)] = np.arange(len(smapes))
            # 归一化到 0 ~ 100.0
            quantiles = (ranks / max(1, len(smapes) - 1)) * 100.0
            
            # 将算好的难度分位数覆写回字典
            for item, q in zip(tsfm_items, quantiles):
                item['smape_quantile'] = float(q)
        # ====================================================================
                    
        # 2. 🌟 执行高级分配逻辑 (Split Dispatcher)
        train_counters = defaultdict(int)
        test_counters = defaultdict(int)
        
        group_by_parent = bool(self.train_config.get('group_by_parent_item_id', 1))

        def _parent_key(item):
            meta = item.get('sample_meta', {}) or {}
            parent = meta.get('parent_item_id')
            if parent is not None and str(parent).strip() != "":
                return f"parent::{str(parent)}"
            item_id = str(meta.get('item_id', '')).strip()
            if item_id:
                return "item::" + re.sub(r'_dim\d+$', '', item_id)
            seq_id = meta.get('seq_id', -1)
            return f"seq::{seq_id}"

        for ds_name, items_list in loaded_items_by_ds.items():
            
            # --- 模式 3: Cross Dataset (完全互斥隔离) ---
            if split_mode == 'cross_dataset':
                allow_leakage = bool(self.train_config.get('allow_data_leakage', 0))
                is_train = ds_name in train_targets
                is_test = ds_name in test_targets if allow_leakage else (ds_name in test_targets and not is_train)
                
                for tsfm, item in items_list:
                    if is_train and (max_train < 0 or train_counters[ds_name] < max_train):
                        self.train_samples.append(item); train_counters[ds_name] += 1
                    if is_test and (max_test < 0 or test_counters[ds_name] < max_test):
                        self.test_samples_dict[tsfm].append(item); test_counters[ds_name] += 1

            # --- 模式 2: Seq Per Dataset (同子集，序列隔离，按 80:20 切分) ---
            elif split_mode == 'seq_per_dataset':
                seq_dict = defaultdict(list)
                for tsfm, item in items_list:
                    if group_by_parent:
                        group_id = _parent_key(item)
                    else:
                        group_id = item.get('sample_meta', {}).get('seq_id', -1)
                    seq_dict[group_id].append((tsfm, item))
                    
                unique_seqs = sorted(list(seq_dict.keys()))
                rs = np.random.RandomState(self.train_config.get('seed', 2025))
                rs.shuffle(unique_seqs) # 序列级别随机打乱
                
                # 🌟 [精细化 80/20 切分]
                n_seqs = len(unique_seqs)
                if n_seqs == 1:
                    # 只有 1 条序列时，优先进入训练集
                    train_seqs = set(unique_seqs)
                    test_seqs = set()
                else:
                    # 使用 round 四舍五入，确保最贴近 80% 的比例（至少保证训练集有 1 条）
                    split_idx = max(1, round(n_seqs * 0.8)) 
                    train_seqs = set(unique_seqs[:split_idx])
                    test_seqs = set(unique_seqs[split_idx:])

                # 🌟 [核心修改] 取消 train_targets/test_targets 的硬性限制
                # 让所有池子里的数据集都在内部进行均匀的 80/20 划分！
                for seq_id, seq_items in seq_dict.items():
                    # 1. 序列分到了训练组 -> 毫无保留进入训练集
                    if seq_id in train_seqs:
                        for tsfm, item in seq_items:
                            if max_train < 0 or train_counters[ds_name] < max_train:
                                self.train_samples.append(item)
                                train_counters[ds_name] += 1
                                
                    # 2. 序列分到了测试组 -> 毫无保留进入测试集
                    elif seq_id in test_seqs:
                        for tsfm, item in seq_items:
                            if max_test < 0 or test_counters[ds_name] < max_test:
                                self.test_samples_dict[tsfm].append(item)
                                test_counters[ds_name] += 1

            # --- 模式 1: Temporal Per Seq (同序列，末端给Test，其余全部充实进数据库) ---
            elif split_mode == 'temporal_per_seq':
                seq_dict = defaultdict(list)
                for tsfm, item in items_list:
                    if group_by_parent:
                        group_id = _parent_key(item)
                    else:
                        group_id = item.get('sample_meta', {}).get('seq_id', -1)
                    seq_dict[group_id].append((tsfm, item))
                    
                is_train_ds = ds_name in train_targets
                is_test_ds = ds_name in test_targets
                    
                # 可以在函数开头或 config 中定义该开关
                drop_last_window = self.train_config.get('drop_last_window', False) # 是否剔除最后一个窗口

                for seq_id, seq_items in seq_dict.items():
                    if not seq_items: continue
                    
                    if is_test_ds:
                        # 按照历史起始时间戳从小到大排序
                        seq_items.sort(key=lambda x: x[1].get('sample_meta', {}).get('hist_start', 0))

                        # 0. 可选策略：剔除最后一个窗口（通常最后一个窗口可能截断或包含边缘数据）
                        if drop_last_window and len(seq_items) > 1:
                            seq_items = seq_items[:-1]
                            
                        if not seq_items:
                            continue
                            
                        # 计算 80/20 切分点：
                        # 若启用 group_by_parent，则先按时间戳分桶，确保同一 parent 下同一时间步的不同通道不被切开。
                        if group_by_parent:
                            buckets = defaultdict(list)
                            for tsfm, item in seq_items:
                                h_start = item.get('sample_meta', {}).get('hist_start', -1)
                                buckets[h_start].append((tsfm, item))
                            unique_starts = sorted(buckets.keys())
                            n_total = len(unique_starts)
                            split_idx = int(n_total * 0.8)
                            if split_idx == n_total and n_total > 0:
                                split_idx = n_total - 1
                            train_starts = set(unique_starts[:split_idx])
                            test_starts = set(unique_starts[split_idx:])
                            train_part, test_part = [], []
                            for h_start in unique_starts:
                                if h_start in train_starts:
                                    train_part.extend(buckets[h_start])
                                elif h_start in test_starts:
                                    test_part.extend(buckets[h_start])
                        else:
                            n_total = len(seq_items)
                            split_idx = int(n_total * 0.8)
                            # 极端短序列保护：如果切分导致测试集为空但总长度大于0，强制至少留 1 个给测试集
                            if split_idx == n_total and n_total > 0:
                                split_idx = n_total - 1 
                            train_part = seq_items[:split_idx]
                            test_part = seq_items[split_idx:]
                        
                        # 1. 前 80% 窗口：倒进数据库(Train)
                        for tsfm, item in train_part:
                            if max_train < 0 or train_counters[ds_name] < max_train:
                                self.train_samples.append(item)
                                train_counters[ds_name] += 1
                                
                        # 2. 后 20% 窗口：独立作为 Test
                        for test_tsfm, test_item in test_part:
                            if max_test < 0 or test_counters[ds_name] < max_test:
                                self.test_samples_dict[test_tsfm].append(test_item)
                                test_counters[ds_name] += 1
                                
                    elif is_train_ds:
                        # 3. 仅属于 train_group：毫无保留，所有窗口全部倒进数据库(Train)
                        for tsfm, item in seq_items:
                            if max_train < 0 or train_counters[ds_name] < max_train:
                                self.train_samples.append(item)
                                train_counters[ds_name] += 1

        self.log(f"📊 数据加载完毕！载入的训练样本总数: {len(self.train_samples)}")
        # ====================================================================
        # 🌟 打印数据质量严格筛查报告
        # ====================================================================
        self.log("\n" + "="*60)
        self.log("🛡️ 数据质量严格筛查统计报告 (NaN/Inf/极值过滤)")
        self.log("="*60)
        self.log(f"📥 总计读取原始样本数: {total_raw}")
        total_dropped = sum(self.data_filter_stats.values())
        self.log(f"✅ 成功保留高质量样本: {total_raw - total_dropped}")
        self.log(f"❌ 总计剔除不合格样本: {total_dropped}")
        if total_dropped > 0:
            self.log("-" * 60)
            self.log("剔除原因明细:")
            for reason, cnt in self.data_filter_stats.items():
                self.log(f"  🔻 {reason}: {cnt} 个样本")
        self.log("="*60 + "\n")
        
        self.log(f"📊 划分完毕！进入训练集的样本数: {len(self.train_samples)}")

        # ====================================================================
        # 🌟 GiftEval / LOTSA 同源跨集率审计
        # ====================================================================
        def _audit_parent(sample):
            meta = sample.get('sample_meta', {}) or {}
            parent = str(meta.get('parent_item_id', '')).strip()
            if parent:
                return parent
            item_id = str(meta.get('item_id', '')).strip()
            if item_id:
                return re.sub(r'_dim\d+$', '', item_id)
            return ""

        train_parent_by_ds = defaultdict(set)
        test_parent_by_ds = defaultdict(set)
        for s in self.train_samples:
            ds = str(s.get('dataset', '')).lower()
            if ds.startswith('lotsa_') or ds.startswith('ge_'):
                p = _audit_parent(s)
                if p:
                    train_parent_by_ds[ds].add(p)

        for samples in self.test_samples_dict.values():
            for s in samples:
                ds = str(s.get('dataset', '')).lower()
                if ds.startswith('lotsa_') or ds.startswith('ge_'):
                    p = _audit_parent(s)
                    if p:
                        test_parent_by_ds[ds].add(p)

        if train_parent_by_ds or test_parent_by_ds:
            self.log("🧪 [Leakage Audit] LOTSA/GiftEval 同源跨集率:")
            for ds in sorted(set(list(train_parent_by_ds.keys()) + list(test_parent_by_ds.keys()))):
                train_p = train_parent_by_ds.get(ds, set())
                test_p = test_parent_by_ds.get(ds, set())
                overlap = len(train_p & test_p)
                denom = max(len(test_p), 1)
                overlap_ratio = overlap / denom
                self.log(f"   - {ds}: overlap={overlap}/{len(test_p)} ({overlap_ratio:.2%})")
        self._scan_and_adapt_lengths()
        if len(self.train_samples) > 0:
            self.log("🏗️ 构建向量数据库...")
        self._scan_and_adapt_lengths()
        if len(self.train_samples) > 0:
            self.log("🏗️ 构建向量数据库...")
            self._ingest_to_db(self.train_samples)
        else:
            if not self.debug: raise ValueError(f"❌ 训练集为空！")

    def _ingest_to_db(self, samples):
        batch_h, batch_r, batch_m, batch_l_r = [], [], [], []
        ctx_len = self.db_config.get('context_len')
        pred_len = self.model_config.get('pred_len', 96)
        
        for item in tqdm(samples, desc="Indexing"):
            h = torch.from_numpy(item['history_norm']).unsqueeze(0)
            r = torch.from_numpy(item['scaled_residual']).unsqueeze(0)
            l_r = torch.from_numpy(item['local_residual_norm']).unsqueeze(0)
            
            h = torch.nn.functional.pad(h, (ctx_len - h.shape[1], 0)) if h.shape[1] < ctx_len else h[:, -ctx_len:]
            r = torch.nn.functional.pad(r, (0, pred_len - r.shape[1])) if r.shape[1] < pred_len else r[:, :pred_len]
            l_r = torch.nn.functional.pad(l_r, (ctx_len - l_r.shape[1], 0)) if l_r.shape[1] < ctx_len else l_r[:, -ctx_len:]
            
            batch_h.append(h); batch_r.append(r); batch_l_r.append(l_r) 
            batch_m.append({'dataset_name': item['dataset'], 'source_model': item.get('source_model'), 'freq': item.get('freq'), 'domain': item.get('domain')})
            
            if len(batch_h) >= 256:
                self.db.add_batch(history=torch.cat(batch_h), residual=torch.cat(batch_r), metas=batch_m, local_residuals=torch.cat(batch_l_r))
                batch_h, batch_r, batch_l_r, batch_m = [], [], [], []
        
        if batch_h: self.db.add_batch(history=torch.cat(batch_h), residual=torch.cat(batch_r), metas=batch_m, local_residuals=torch.cat(batch_l_r))

    def _get_valid_lens_from_padding(self, tensor, tol=1e-6):
        B, L = tensor.shape
        non_zero_mask = torch.abs(tensor) > tol
        indices = torch.arange(L, device=tensor.device).unsqueeze(0).expand(B, L)
        valid_indices = torch.where(non_zero_mask, indices, -1)
        valid_lens = valid_indices.max(dim=1).values + 1
        return valid_lens

    @torch.no_grad()
    def _compute_metrics_gpu(self, preds, targets, histories, valid_lens):
        # 🌟 1. 提升至 float64 (double) 精度计算，防止单样本的内部长时序累加溢出
        preds = torch.clamp(preds.double(), min=-1e9, max=1e9)
        targets = targets.double()
        histories = histories.double()
        B, L = preds.shape
        
        mask = torch.arange(L, device=preds.device).expand(B, L) < valid_lens.unsqueeze(1)
        valid_elements = mask.sum(dim=1).double()
        
        mse_num = (((preds - targets) ** 2) * mask.double()).sum(dim=1)
        
        abs_p, abs_t = torch.abs(preds), torch.abs(targets)
        denom = abs_p + abs_t
        valid_smape_mask = (denom > 1e-5) & mask
        # 使用 200.0 确保浮点计算
        smape_num = (200.0 * torch.abs(preds - targets) / denom).masked_fill(~valid_smape_mask, 0.0).sum(dim=1)
        smape_denom = valid_smape_mask.sum(dim=1).double()
        
        mae_per_sample = (torch.abs(preds - targets) * mask.double()).sum(dim=1) / (valid_lens.double() + 1e-8)
        naive_denom = torch.mean(torch.abs(histories[:, 1:] - histories[:, :-1]), dim=1)
        valid_mase_mask = naive_denom > 1e-5
        mase_num = mae_per_sample * valid_mase_mask.double() / (naive_denom + 1e-8)
        
        return mse_num, valid_elements, smape_num, smape_denom, mase_num, valid_mase_mask.double()

    def _evaluate_loader(self, loader, model=None):
        if model: model.eval()
        all_res = defaultdict(list)
        use_vibe = self.model_config.get('use_vibe_features', True)

        with torch.no_grad():
            for id, batch in enumerate(loader):
                t_hist = batch['history'].to(self.device).squeeze(1).float()
                t_emb = batch['sample_embedding'].to(self.device).squeeze(1).float()
                c_embs = batch['retrieved_context'].to(self.device).float()
                c_res = batch['retrieved_residual'].to(self.device).float()
                target_res_scaled = batch['target_residual'].to(self.device).squeeze(1).float()
                truth = batch['truth'].to(self.device).squeeze(1).float()
                scale = batch['scale'].to(self.device).view(-1, 1).float()

                t_err_emb = batch.get('sample_err_embedding')
                c_err_embs = batch.get('retrieved_err_context')
                if t_err_emb is not None: t_err_emb = t_err_emb.to(self.device).squeeze(1).float()
                if c_err_embs is not None: c_err_embs = c_err_embs.to(self.device).float()

                target_local_res = batch.get('local_residual')
                if target_local_res is not None: 
                    target_local_res = target_local_res.to(self.device).float()
                    # 🌟 [修复] 强制去掉中间的通道维度
                    if target_local_res.dim() == 3: target_local_res = target_local_res.squeeze(1)

                valid_lens = batch['valid_len'].to(self.device)
                base_pred = truth - (target_res_scaled * scale)
                B = base_pred.shape[0]

                # ========================================================
                # 🌟 1. 计算当前样本的 Local Validation sMAPE (用于排序门控)
                # ========================================================
                if target_local_res is not None:
                    local_truth = t_hist
                    local_pred = t_hist - target_local_res
                    local_denom = torch.abs(local_pred) + torch.abs(local_truth)
                    local_mask = torch.abs(local_truth) > 1e-5
                    valid_local_mask = (local_denom > 1e-5) & local_mask
                    l_smape = (200.0 * torch.abs(local_pred - local_truth) / local_denom).masked_fill(~valid_local_mask, 0.0)
                    local_smape = l_smape.sum(dim=1) / valid_local_mask.sum(dim=1).clamp(min=1).double()
                else:
                    local_smape = torch.zeros(B, device=self.device, dtype=torch.float64)

                # ========================================================
                # 🌟 2. 预测未来残差并截断到 L_min
                # ========================================================
                if model is None:
                    final_pred = base_pred
                    L_min = base_pred.shape[1]
                else:
                    kwargs = {}
                    ts_physics_features = batch.get('ts_physics_features')
                    if ts_physics_features is not None:
                        kwargs['ts_physics_features'] = ts_physics_features.to(self.device).float()

                    if use_vibe:
                        recent_hist = t_hist[:, -24:] if t_hist.shape[1] >= 24 else t_hist
                        if recent_hist.shape[1] > 1:
                            hist_vol = torch.std(recent_hist, dim=1, keepdim=True) + 1e-5
                            hist_mad = torch.mean(torch.abs(recent_hist[:, 1:] - recent_hist[:, :-1]), dim=1, keepdim=True) + 1e-5
                        else:
                            hist_vol = torch.zeros((B, 1), device=self.device) + 1e-5
                            hist_mad = torch.zeros((B, 1), device=self.device) + 1e-5
                        kwargs['vibe_features'] = torch.cat([hist_vol, hist_mad], dim=1).float()

                    pred_norm, info = self.model(
                        t_emb, c_embs, c_res, t_hist, target_res_scaled,
                        t_err_emb=t_err_emb, c_err_embs=c_err_embs, target_local_res=target_local_res, **kwargs
                    )

                    pred_corr_raw = pred_norm * scale
                    L_min = min(pred_corr_raw.shape[1], base_pred.shape[1])
                    final_pred = base_pred[:, :L_min] + pred_corr_raw[:, :L_min]

                # ========================================================
                # 🌟 3. 计算 Base 和 Corrected 的 sMAPE
                # ========================================================
                truth_trunc = truth[:, :L_min]
                base_pred_trunc = base_pred[:, :L_min]
                abs_t = torch.abs(truth_trunc)
                mask_future = torch.arange(L_min, device=self.device).expand(B, L_min) < valid_lens.unsqueeze(1)

                # Base (如果不修正，模型会产生的原始误差)
                abs_p_base = torch.abs(base_pred_trunc)
                denom_base = abs_p_base + abs_t
                valid_base_mask = (denom_base > 1e-5) & mask_future
                base_smape_num = (200.0 * torch.abs(base_pred_trunc - truth_trunc) / denom_base).masked_fill(~valid_base_mask, 0.0).sum(dim=1)
                smape_den = valid_base_mask.sum(dim=1).double()

                # Corrected (如果全力修正，模型产生的误差)
                abs_p_corr = torch.abs(final_pred)
                denom_corr = abs_p_corr + abs_t
                valid_corr_mask = (denom_corr > 1e-5) & mask_future
                corr_smape_num = (200.0 * torch.abs(final_pred - truth_trunc) / denom_corr).masked_fill(~valid_corr_mask, 0.0).sum(dim=1)

                all_res['base_smape_num'].append(base_smape_num)
                all_res['corr_smape_num'].append(corr_smape_num)
                all_res['smape_den'].append(smape_den)
                all_res['local_smape'].append(local_smape)

        return {k: torch.cat(v).cpu().numpy() for k, v in all_res.items()}

    def build_loaders(self):
        pred_len = self.model_config.get('pred_len', 96)
        filter_freq = bool(self.train_config.get('filter_by_freq', 0))
        filter_domain = bool(self.train_config.get('filter_by_domain', 0))
        scope_mode = self.train_config.get('retrieval_scope', 'cross_dataset')
        if scope_mode == 'cross_dataset': strategy = CrossDatasetScopeStrategy()
        elif scope_mode == 'same_model': strategy = SameModelScopeStrategy()
        else: strategy = GlobalScopeStrategy()
            
        retrieval_alpha = self.train_config.get('retrieval_alpha', 1.0)
        retrieval_beta = self.train_config.get('retrieval_beta', 1.0) 
        pseudo_ratio = self.train_config.get('pseudo_ratio', 0.0)
        pseudo_strength = self.train_config.get('pseudo_strength', 0)

        if pseudo_ratio > 0 and pseudo_strength > 0 and getattr(self, 'tsfm_predictor', None) is None:
            self.tsfm_predictor = self._make_tsfm_predictor()
        elif not hasattr(self, 'tsfm_predictor'):
            self.tsfm_predictor = None

        if self.train_ds is None and len(self.train_samples) > 0:
            self.train_ds = CorrectionDataset(
                self.db, self.train_samples, self.db_config['context_len'], 
                self.model_config.get('top_k', 5), pred_len=pred_len, retrieval_strategy='encoder',
                scope_strategy=strategy, pseudo_ratio=pseudo_ratio, pseudo_aug_strength=pseudo_strength,
                tsfm_predictor=self.tsfm_predictor, shuffle_retrieved_order=self.train_config.get('shuffle_retrieved_order', False),
                retrieval_alpha=retrieval_alpha, retrieval_beta=retrieval_beta, filter_by_freq=filter_freq, filter_by_domain=filter_domain
            )
            
        if getattr(self, 'train_loader', None) is None and self.train_ds is not None:
            hard_q_train = self.train_config.get('hard_quantile_train', 100.0)
            train_q_threshold = 100.0 - hard_q_train
            valid_indices = [i for i, q_val in enumerate(self.train_ds.smape_quantiles) if q_val >= train_q_threshold]
            filtered_train_ds = Subset(self.train_ds, valid_indices)
            self.train_loader = DataLoader(filtered_train_ds, batch_size=self.train_config.get('batch_size', 32), shuffle=True, num_workers=0)

        if getattr(self, 'val_loaders', None) is None:
            self.val_loaders = {}
            for tsfm_name, samples in self.test_samples_dict.items():
                if not samples: continue
                ds = CorrectionDataset(
                    self.db, samples, self.db_config['context_len'], self.model_config.get('top_k', 5), pred_len=pred_len,
                    retrieval_strategy='encoder', scope_strategy=strategy, precompute_embeddings=True, 
                    pseudo_ratio=pseudo_ratio, pseudo_aug_strength=pseudo_strength, tsfm_predictor=self.tsfm_predictor,
                    retrieval_alpha=retrieval_alpha, retrieval_beta=retrieval_beta, filter_by_freq=filter_freq, filter_by_domain=filter_domain
                )
                self.val_loaders[tsfm_name] = DataLoader(ds, batch_size=self.train_config.get('batch_size', 32), shuffle=False)

    def run(self):
        self.build_loaders()
        train_loader = getattr(self, 'train_loader', [])
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        
        gating_strategy = self.train_config.get('gating_strategy', 'none')
        
        # 🌟 只有 scan_100 模式需要预计算阈值
        if gating_strategy == 'scan_100':
            # ========================================================
            # 🌟 新增：预计算训练集 Local Validation sMAPE 分位数阈值
            # ========================================================
            self.log(f"📊 正在计算训练集 Local sMAPE 以生成 1~100 百分位门控阈值...")
            train_local_smapes = []
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Calc Train Thresholds"):
                    t_hist = batch['history'].to(self.device).squeeze(1).float()
                    target_local_res = batch.get('local_residual')
                    if target_local_res is not None:
                        target_local_res = target_local_res.to(self.device).float()
                        # 🌟 [修复] 强制去掉中间的通道维度，防止生成 (B, B, 512)
                        if target_local_res.dim() == 3: target_local_res = target_local_res.squeeze(1)
                        
                        local_truth = t_hist
                        local_pred = t_hist - target_local_res
                        denom = torch.abs(local_pred) + torch.abs(local_truth)
                        local_mask = torch.abs(local_truth) > 1e-5
                        valid_mask = (denom > 1e-5) & local_mask
                        l_smape = (200.0 * torch.abs(local_pred - local_truth) / denom).masked_fill(~valid_mask, 0.0)
                        l_smape_mean = l_smape.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1).double()
                        train_local_smapes.extend(l_smape_mean.cpu().tolist())

            train_local_smapes = np.array(train_local_smapes)
            self.thresholds_dict = {}
            for x in range(1, 101):
                # x 表示 Top X% (x=10 意味着只允许最难的前 10% 样本被修正，阈值取第 90 分位数)
                # x=100 意味着允许 100% 的样本修正，阈值是 0
                self.thresholds_dict[x] = np.percentile(train_local_smapes, 100 - x)
                
            # # 准备专门记录 100 个门控日志的 CSV
            # output_dir = self.train_config.get('output_dir', 'checkpoints')
            # best_ckpt_path = os.path.join(output_dir, "best_model.pth")
            # csv_path = os.path.join(output_dir, "metrics_gated_100.csv")
            
            # sorted_models = sorted(self.val_loaders.keys())
            # headers = ["Epoch", "Train_Loss"]
            # for x in range(1, 101):
            #     for m in sorted_models:
            #         headers.extend([f"{m}_Imp_Top{x}%"])
                    
            # if not os.path.exists(csv_path):
            #     with open(csv_path, 'w', newline='') as f:
            #         csv.writer(f).writerow(headers)
            # ========================================================
        # 🌟 初始化结果保存机制 (格式对齐 metrics.csv)
        # ========================================================
        output_dir = self.train_config.get('output_dir', 'checkpoints')
        best_ckpt_path = os.path.join(output_dir, "best_model.pth")
        csv_path = os.path.join(output_dir, "metrics.csv")
        show_train_metrics = self.train_config.get('show_train_metrics', 0) == 1
        sorted_models = sorted(self.val_loaders.keys()) if self.val_loaders else []
        headers = ["Epoch", "Train_Loss"]
        # 🌟 [新增] 如果开启了展示训练集，在表头优先加入这两列
        if show_train_metrics:
            headers.extend(["Train_sMAPE", "Train_sMAPE_Imp"])
        # 动态生成表头：如果是 scan_100 则生成 1~100 的宽表，否则生成基础窄表
        if gating_strategy == 'scan_100':
            for x in range(1, 101):
                for m in sorted_models:
                    # 严格对齐 metrics.csv 的命名规范格式
                    headers.extend([f"{m}_sMAPE_Top{x}", f"{m}_sMAPE_Imp_Top{x}"])
        else:
            for m in sorted_models:
                headers.extend([f"{m}_sMAPE", f"{m}_sMAPE_Imp"])
                
        # 如果是第一次运行或者文件不存在，则写入表头
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
        # ========================================================
        # 🌟 动态构建损失函数 (Loss Factory)
        # ========================================================
        loss_type = self.train_config.get('loss_type', 'hybrid').lower()
        self.log(f"📉 使用损失函数: {loss_type.upper()}")
        
        # 必须使用 reduction='none' 保证我们可以手动应用 Valid Mask
        if loss_type == 'huber':
            criterion = nn.HuberLoss(delta=self.train_config.get('huber_delta', 1.0), reduction='none')
        elif loss_type == 'mse':
            criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            criterion = nn.L1Loss(reduction='none')
        else:
            criterion = HybridFrequencyLoss(
                alpha=self.train_config.get('freq_alpha', 1.0),
                beta=self.train_config.get('freq_beta', 0.2),
                gamma=self.train_config.get('freq_gamma', 0.5),
                delta=self.train_config.get('huber_delta', 1.0)
            )
        self.criterion = criterion.to(self.device)
        
        use_vibe = self.model_config.get('use_vibe_features', True)
        
        output_dir = self.train_config.get('output_dir', 'checkpoints')
        best_ckpt_path = os.path.join(output_dir, "best_model.pth")
        csv_path = os.path.join(output_dir, "metrics.csv")
        
        sorted_models = sorted(self.val_loaders.keys())
        raw_hard_q_tests = self.train_config.get('hard_quantile_test', [100.0])
        if not isinstance(raw_hard_q_tests, list): raw_hard_q_tests = [raw_hard_q_tests]
        # primary_q_val = raw_hard_q_tests[0] 
        # primary_q_val = max(raw_hard_q_tests)
        display_q_tests = sorted(list(set(raw_hard_q_tests))) 
        
        # self.log(f"📊 正在计算 Baseline 全量性能... (以支持极速路由模拟)")
        # baseline_results = {}
        # for m_name in sorted_models:
        #     b_res = self._evaluate_loader(self.val_loaders[m_name], model=None)
        #     baseline_results[m_name] = b_res
        #     b_smape = b_res['smape_num'].sum() / max(b_res['smape_den'].sum(), 1e-8)
        #     b_mase = b_res['mase_num'].sum() / max(b_res['mase_den'].sum(), 1e-8)
        #     self.log(f"   🔹 {m_name}: 全量 Base sMAPE={b_smape:.2f} | Base MASE={b_mase:.3f}")
        
        # headers = ["Epoch", "Train_Loss"]
        # for q_val in display_q_tests:
        #     for m in sorted_models:
        #         headers.extend([f"{m}_sMAPE_Top{q_val}", f"{m}_sMAPE_Imp_Top{q_val}", f"{m}_MASE_Top{q_val}", f"{m}_MASE_Imp_Top{q_val}"])
            
        # if not os.path.exists(csv_path):
        #     with open(csv_path, 'w', newline='') as f:
        #         csv.writer(f).writerow(headers)
                
        # patience_counter = 0
        # epochs = self.train_config.get('epochs', 100)
        # min_save_epoch = 3
        
        # # 🌟 [核心修改 1] 将保存基准改为“最大增益 (Imp)”，由于增益越大越好，初始值设为负无穷
        # best_imp_metric = -float('inf') 
        # ========================================================
        # 🌟 极速 Baseline 性能计算 (仅基于 sMAPE)
        # ========================================================
        self.log(f"📊 正在计算 Baseline 全量性能... (以支持极速路由模拟)")
        baseline_results = {}
        for m_name in sorted_models:
            b_res = self._evaluate_loader(self.val_loaders[m_name], model=None)
            baseline_results[m_name] = b_res
            # 注意：新的 loader 返回的是 base_smape_num
            b_smape = b_res['base_smape_num'].sum() / max(b_res['smape_den'].sum(), 1e-8)
            self.log(f"   🔹 {m_name}: 全量 Base sMAPE={b_smape:.2f}")
                
        patience_counter = 0
        epochs = self.train_config.get('epochs', 100)
        min_save_epoch = 3
        
        # 将保存基准设为“最大增益 (Imp)”，初始值设为负无穷
        best_imp_metric = -float('inf')
        self.log(f"🚀 开始训练 (Epochs: {epochs}, Start saving at Ep {min_save_epoch})")
        
        for epoch in range(epochs):
            train_loss_acc = 0.0
            valid_batch_count = 0
            
            # ==========================================
            # 训练阶段 (保持不变)
            # ==========================================
            if optimizer and train_loader: 
                self.model.train()
                pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False)
                for id, batch in enumerate(pbar):
                    t_hist = batch['history'].to(self.device).squeeze(1).float()
                    t_emb = batch['sample_embedding'].to(self.device).squeeze(1).float()
                    c_embs = batch['retrieved_context'].to(self.device).float()
                    c_res = batch['retrieved_residual'].to(self.device).float()
                    target_res_scaled = batch['target_residual'].to(self.device).squeeze(1).float()
                    
                    t_err_emb = batch.get('sample_err_embedding')
                    c_err_embs = batch.get('retrieved_err_context')
                    if t_err_emb is not None: t_err_emb = t_err_emb.to(self.device).squeeze(1).float()
                    if c_err_embs is not None: c_err_embs = c_err_embs.to(self.device).float()
                    
                    target_local_res = batch.get('local_residual')
                    if target_local_res is not None: 
                        target_local_res = target_local_res.to(self.device).float()
                        # 🌟 [修复] 强制去掉中间的通道维度
                        if target_local_res.dim() == 3: target_local_res = target_local_res.squeeze(1)
                    
                    valid_lens = batch['valid_len'].to(self.device)

                    # ============================================================
                    # 🚨 [DEBUG 探照灯] 模型输入前 X光检查
                    # ============================================================
                    debug_tensors = {
                        "t_hist": t_hist, "t_emb": t_emb, 
                        "c_embs": c_embs, "c_res": c_res, 
                        "target_res": target_res_scaled,
                        "t_err_emb": t_err_emb, "c_err_embs": c_err_embs
                    }
                    for name, tsr in debug_tensors.items():
                        if tsr is not None and (torch.isnan(tsr).any() or torch.isinf(tsr).any()):
                            print(f"\n🚨 [FATAL] 输入张量 '{name}' 包含 NaN/Inf!")
                            print(f"   Max: {tsr.max().item() if not torch.isnan(tsr).all() else 'NaN'}, Min: {tsr.min().item() if not torch.isnan(tsr).all() else 'NaN'}")
                            raise ValueError("训练被迫中止：输入数据已污染")
                    # ============================================================

                    # if optimizer: optimizer.zero_grad()

                    if optimizer: optimizer.zero_grad()
                    with autocast(device_type=self.device_type, enabled=(self.device_type=='cuda')):
                        label_norm = target_res_scaled
                        kwargs = {}
                        
                        # ============================================================
                        # 🌟 [新增] 动态读取并传入门控特征
                        # ============================================================
                        ts_physics_features = batch.get('ts_physics_features')
                        if ts_physics_features is not None:
                            kwargs['ts_physics_features'] = ts_physics_features.to(self.device).float()
                        
                        if use_vibe:
                            recent_hist = t_hist[:, -24:] if t_hist.shape[1] >= 24 else t_hist
                            if recent_hist.shape[1] > 1:
                                hist_vol = torch.std(recent_hist, dim=1, keepdim=True) + 1e-5 
                                hist_mad = torch.mean(torch.abs(recent_hist[:, 1:] - recent_hist[:, :-1]), dim=1, keepdim=True) + 1e-5
                            else:
                                hist_vol = torch.zeros((t_hist.shape[0], 1), device=self.device) + 1e-5
                                hist_mad = torch.zeros((t_hist.shape[0], 1), device=self.device) + 1e-5
                            kwargs['vibe_features'] = torch.cat([hist_vol, hist_mad], dim=1).float()

                        pred_norm, info = self.model(
                            t_emb, c_embs, c_res, t_hist, target_res_scaled,
                            t_err_emb=t_err_emb, c_err_embs=c_err_embs, target_local_res=target_local_res, **kwargs
                        )
                        pred_out = info.get('pred_res_normalized', pred_norm)
                        raw_pred_out = info.get('raw_pred_res_normalized', pred_out)

                        B, L = pred_out.shape
                        mask = torch.arange(L, device=self.device).expand(B, L) < valid_lens.unsqueeze(1)
                        
                        pred_clean = pred_out * mask.float()
                        raw_pred_clean = raw_pred_out * mask.float()
                        label_clean = label_norm * mask.float()
                        
                        # 1. 基础主损失
                        loss_matrix = self.criterion(pred_clean, label_clean)
                        main_loss = (loss_matrix * mask.float()).sum() / (mask.sum() + 1e-8)
                        
                        gating_strategy = self.train_config.get('gating_strategy', 'none')
                        loss = main_loss
                        
                        # 2. 仅在 learnable 模式下计算复杂的辅助门控与幽灵梯度
                        if gating_strategy == 'learnable':
                            raw_loss_matrix = self.criterion(raw_pred_clean, label_clean)
                            ghost_loss = (raw_loss_matrix * mask.float()).sum() / (mask.sum() + 1e-8)
                            loss = main_loss + 0.5 * ghost_loss
                            
                            gate_logits = info.get('gate_logits')
                            if gate_logits is not None:
                                with torch.no_grad():
                                    base_err = torch.abs(label_clean).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                                    corr_err = torch.abs(raw_pred_clean - label_clean).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                                    target_worth = (base_err > corr_err).float().unsqueeze(-1)
                                    
                                pos_weight = torch.tensor([5.0], device=self.device)
                                gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(gate_logits, target_worth, pos_weight=pos_weight)
                                loss = loss + self.train_config.get('gate_loss_weight', 0.5) * gate_loss
                                info['gate_loss'] = gate_loss
                    if loss is None or torch.isnan(loss) or torch.isinf(loss): continue
                        
                    if optimizer:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        
                    train_loss_acc += loss.item()
                    valid_batch_count += 1
                    
                    postfix = {'loss': f"{loss.item():.4f}"}
                    if 'gate_prob' in info: postfix['g_prob'] = f"{info['gate_prob'].mean().item():.2f}"
                    if 'gate_loss' in info: postfix['g_loss'] = f"{info['gate_loss'].item():.4f}"
                    pbar.set_postfix(postfix)
                
                avg_train_loss = train_loss_acc / max(1, valid_batch_count)
            else:
                avg_train_loss = 0.0
            
            # ==========================================
            # 🌟 推理与验证阶段 (多策略路由)
            # ==========================================
            self.model.eval()
            self.log(f"\nEp {epoch+1} | Loss: {avg_train_loss:.4f}")
            row = {"Epoch": epoch+1, "Train_Loss": avg_train_loss}
            
            # 1. 跑一次全量无删减推断，获得所有测试集样本的修正后误差
            model_eval_results = {}
            for m_name in sorted_models:
                model_eval_results[m_name] = self._evaluate_loader(self.val_loaders[m_name], model=self.model)
                
            best_epoch_imp = -float('inf')
            
            # ==========================================
            # 🌟 [核心修改] 提前计算训练集表现 (支持采样/全量/跳过)
            # ==========================================
            train_eval_mode = self.train_config.get('train_eval_mode', 'full') # 'none', 'sample', 'full'
            train_eval_samples = self.train_config.get('train_eval_samples', 1000)

            if show_train_metrics or gating_strategy == 'scan_100':
                if train_eval_mode == 'none':
                    # 模式 1：跳过验证，伪造全 0 的返回值，防止后续计算崩溃
                    train_eval_res = {
                        'base_smape_num': np.array([0.0]),
                        'corr_smape_num': np.array([0.0]),
                        'smape_den': np.array([1e-8]), # 防止除零
                        'local_smape': np.array([0.0])
                    }
                else:
                    eval_loader = self.train_loader
                    # 模式 2：随机采样验证 (利用 Subset 构建轻量级 Loader)
                    if train_eval_mode == 'sample' and train_eval_samples > 0 and train_eval_samples < len(self.train_ds):
                        # 随机抽取指定数量的索引
                        indices = np.random.choice(len(self.train_ds), train_eval_samples, replace=False).tolist()
                        sampled_ds = Subset(self.train_ds, indices)
                        # 生成仅包含采样子集的 DataLoader
                        eval_loader = DataLoader(
                            sampled_ds, 
                            batch_size=self.train_loader.batch_size, 
                            shuffle=False, 
                            num_workers=self.train_loader.num_workers
                        )
                    # 模式 3：全量验证 (默认执行 eval_loader = self.train_loader)
                    
                    # 执行推断
                    train_eval_res = self._evaluate_loader(eval_loader, model=self.model)
                
            if show_train_metrics:
                # 提取指标
                if train_eval_mode == 'none':
                    tr_base, tr_curr, tr_imp = 0.0, 0.0, 0.0
                else:
                    b_num_tr = np.sum(train_eval_res['base_smape_num'], dtype=np.float64)
                    c_num_tr = np.sum(train_eval_res['corr_smape_num'], dtype=np.float64)
                    den_tr = np.sum(train_eval_res['smape_den'], dtype=np.float64)
                    
                    tr_base = b_num_tr / max(den_tr, 1e-8)
                    tr_curr = c_num_tr / max(den_tr, 1e-8)
                    tr_imp = (tr_base - tr_curr) / max(tr_base, 1e-8) * 100
                
                # 打印日志与记录
                mode_str = f"Sample={train_eval_samples}" if train_eval_mode == 'sample' else train_eval_mode.capitalize()
                self.log(f"   🔸 [Train Set | {mode_str}] Imp: {tr_imp:+.4f}% (sMAPE: {tr_base:.2f} -> {tr_curr:.2f})")
                row["Train_sMAPE"] = tr_curr
                row["Train_sMAPE_Imp"] = tr_imp

            # ==========================================
            # 🚀 路线 A: 扫描 1~100 虚拟分位数阈值 (严谨双集验证)
            # ==========================================
            if gating_strategy == 'scan_100':
                best_train_imp = -float('inf')
                best_x, best_thresh = None, None
                
                for x in range(1, 101):
                    thresh = self.thresholds_dict[x]
                    l_smape_tr = train_eval_res['local_smape'] # 👈 直接复用上面的计算结果
                    gate_mask_tr = l_smape_tr >= thresh
                    final_num_tr = np.where(gate_mask_tr, train_eval_res['corr_smape_num'], train_eval_res['base_smape_num'])
                    
                    b_num = np.sum(train_eval_res['base_smape_num'], dtype=np.float64)
                    c_num = np.sum(final_num_tr, dtype=np.float64)
                    den = np.sum(train_eval_res['smape_den'], dtype=np.float64)
                    
                    t_imp = (b_num/max(den, 1e-8) - c_num/max(den, 1e-8)) / max(b_num/max(den, 1e-8), 1e-8) * 100
                    if t_imp > best_train_imp:
                        best_train_imp = t_imp
                        best_x, best_thresh = x, thresh
                
                # 步骤 2：在【测试集】上扫描并记录所有表现
                test_imp_at_best_x = 0.0 
                for x in range(1, 101):
                    thresh = self.thresholds_dict[x]
                    total_base_num, total_corr_num, total_den = 0.0, 0.0, 0.0
                    
                    for m_name in sorted_models:
                        c_res = model_eval_results[m_name]
                        gate_mask = c_res['local_smape'] >= thresh
                        final_num = np.where(gate_mask, c_res['corr_smape_num'], c_res['base_smape_num'])
                        
                        b_num = np.sum(c_res['base_smape_num'], dtype=np.float64)
                        c_num = np.sum(final_num, dtype=np.float64)
                        den = np.sum(c_res['smape_den'], dtype=np.float64)
                        
                        total_base_num += b_num
                        total_corr_num += c_num
                        total_den += den
                        
                        m_base = b_num / max(den, 1e-8)
                        m_curr = c_num / max(den, 1e-8)
                        m_imp = (m_base - m_curr) / max(m_base, 1e-8) * 100
                        
                        row[f"{m_name}_sMAPE_Top{x}"] = m_curr
                        row[f"{m_name}_sMAPE_Imp_Top{x}"] = m_imp

                    global_base = total_base_num / max(total_den, 1e-8)
                    global_curr = total_corr_num / max(total_den, 1e-8)
                    global_imp = (global_base - global_curr) / max(global_base, 1e-8) * 100
                    
                    if x == best_x:
                        test_imp_at_best_x = global_imp
                
                best_epoch_imp = test_imp_at_best_x
                self.log(f"🏆 [Scan_100] 根据训练集选取门控: Top {best_x}% 难样本 (阈值 >= {best_thresh:.2f}%)")
                self.log(f"   -> 训练集虚拟门控 Imp: {best_train_imp:+.4f}% | 真实测试集 Imp: {best_epoch_imp:+.4f}%")

            # ==========================================
            # 🚀 路线 B: 常规直接评估 ('none', 'learnable', 'static')
            # ==========================================
            else:
                total_base_num, total_corr_num, total_den = 0.0, 0.0, 0.0
                
                for m_name in sorted_models:
                    c_res = model_eval_results[m_name]
                    b_num = np.sum(c_res['base_smape_num'], dtype=np.float64)
                    c_num = np.sum(c_res['corr_smape_num'], dtype=np.float64) 
                    den = np.sum(c_res['smape_den'], dtype=np.float64)
                    
                    total_base_num += b_num
                    total_corr_num += c_num
                    total_den += den
                    
                    m_base = b_num / max(den, 1e-8)
                    m_curr = c_num / max(den, 1e-8)
                    m_imp = (m_base - m_curr) / max(m_base, 1e-8) * 100
                    
                    row[f"{m_name}_sMAPE"] = m_curr
                    row[f"{m_name}_sMAPE_Imp"] = m_imp
                    self.log(f"   🔹 [Test] {m_name} Imp: {m_imp:+.4f}% (sMAPE: {m_curr:.2f})")
                    
                global_base = total_base_num / max(total_den, 1e-8)
                global_curr = total_corr_num / max(total_den, 1e-8)
                best_epoch_imp = (global_base - global_curr) / max(global_base, 1e-8) * 100
                self.log(f"🏆 全局平均测试集 Imp: {best_epoch_imp:+.4f}%")
                
            # ==========================================
            # 💾 文件写入：静默追加到 CSV
            # ==========================================
            with open(csv_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=headers).writerow(row)
                
            # ==========================================
            # 4. 最佳模型保存与早停机制 (共享最佳指标)
            # ==========================================
            current_ep = epoch + 1
            if current_ep < min_save_epoch:
                self.log(f"⏳ Warmup Phase: Best Imp = {best_epoch_imp:+.4f}% (Not tracked yet)")
            else:
                if best_epoch_imp > best_imp_metric + 1e-4:
                    self.log(f"🔥 新纪录！最佳真实 Imp 突破: {best_epoch_imp:+.4f}% > {best_imp_metric:+.4f}%")
                    best_imp_metric = best_epoch_imp
                    patience_counter = 0
                    try: torch.save(self.model.state_dict(), best_ckpt_path)
                    except Exception as e: self.log(f"❌ Save Failed: {e}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience: 
                        self.log("🛑 连续多轮未突破最佳拦截增益，Early Stopping 触发。")
                        break
            
            # 5. 调整学习率
            if scheduler:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau): 
                    scheduler.step(-best_epoch_imp) 
                else: 
                    scheduler.step()
