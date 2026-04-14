# Copyright (c) 2023, Salesforce, Inc.
# Copyright (c) 2025 fireball0213, LAMDA, Nanjing University
# SPDX-License-Identifier: Apache-2

import os
import math
from functools import cached_property
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Optional
from gluonts.model.forecast import SampleForecast, QuantileForecast
import json
import numpy as np
import pandas as pd
import datasets
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
import pyarrow.compute as pc
from toolz import compose

# ====================== 常量定义 ======================

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6, "H": 48, "Q": 8, "D": 14, "M": 18, "W": 13, "U": 8, "T": 8,
}


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT: return 1
        elif self == Term.MEDIUM: return 10
        elif self == Term.LONG: return 15


# ====================== 核心工具函数 ======================

def clean_frequency(freq: str) -> str:
    """清洗频率字符串，确保与 pd.Period 兼容。"""
    if freq in ['MS', '1MS']: return 'M'
    if freq in ['YS', 'AS', '1YS', '1AS']: return 'A'
    if freq in ['QS', '1QS']: return 'Q'
    
    if freq == '1H': return 'H'
    if freq == '1min': return 'min'
    if freq == '1D': return 'D'
    
    return freq

def itemize_start(data_entry: DataEntry, freq: str = None) -> DataEntry:
    """提取 start 并转为 pd.Period"""
    start = data_entry["start"]
    if hasattr(start, "item"):
        start = start.item()
    
    if isinstance(start, (int, float)):
        if start > 3e11: ts = pd.to_datetime(start, unit='ns')
        else: ts = pd.to_datetime(start, unit='s')
    else:
        ts = pd.Timestamp(start)

    if freq:
        data_entry["start"] = pd.Period(ts, freq=freq)
    else:
        data_entry["start"] = pd.Period(ts)
        
    return data_entry


def maybe_reconvert_freq(freq: str) -> str:
    deprecated_map = {
        "Y": "A", "YE": "A", "QE": "Q", "ME": "M",
        "h": "H", "min": "T", "s": "S", "us": "U",
    }
    if freq in deprecated_map:
        return deprecated_map[freq]
    return freq


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(self, data_it: Iterable[DataEntry], is_train: bool = False) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry

# ====================== 辅助类: 自适应测试集 ======================

class AdaptiveTestData:
    """
    模拟 GluonTS TestData 对象的行为，但内部使用自适应窗口逻辑。
    必须提供 .input 和 .label 属性，且支持 len()。
    """
    def __init__(self, dataset_instance, max_window):
        self.dataset = dataset_instance
        self.max_window = max_window
        # 预计算所有切片，确保 len() 可用
        self._cache = self._generate_all()

    def _generate_all(self):
        pairs = []
        pred_len = self.dataset.prediction_length
        dist = pred_len 
        
        for entry in self.dataset.gluonts_dataset:
            full_target = entry["target"]
            full_len = full_target.shape[-1]
            
            curr_end = full_len
            count = 0
            
            # 自适应切片循环
            while count < self.max_window:
                split_idx = curr_end - pred_len
                
                # 剩余数据不足以构成一个完整的 Label，停止
                if split_idx < 0:
                    break
                
                input_entry = entry.copy()
                input_entry["target"] = full_target[..., :split_idx]
                
                label_entry = entry.copy()
                label_entry["target"] = full_target[..., split_idx:curr_end]
                
                pairs.append((input_entry, label_entry))
                
                curr_end -= dist
                count += 1
        return pairs

    def __iter__(self):
        return iter(self._cache)
    
    def __len__(self):
        return len(self._cache)

    @property
    def input(self):
        """返回仅包含 input_entry 的列表，供 predictor 使用"""
        return [p[0] for p in self._cache]

    @property
    def label(self):
        """[新增] 返回仅包含 label_entry 的列表，供 evaluate_forecasts 使用"""
        return [p[1] for p in self._cache]


# ====================== 主 Dataset 类 ======================

class Dataset:
    def __init__(
            self,
            name: str,
            term: Term | str = Term.SHORT,
            to_univariate: bool = False,
            storage_env_var: str = "Project_Path",
            force_univariate: bool = False
    ):
        load_dotenv()
        
        self.force_univariate = force_univariate
        self.to_univariate = to_univariate
        self.term = Term(term)
        self.name = name

        project_path = os.getenv(storage_env_var)
        if project_path:
            storage_path = Path(project_path) / "Datasets/processed_datasets"
        else:
            storage_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "processed_datasets"

        target_path = storage_path / name
        if not target_path.exists():
            alt_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "Datasets/processed_datasets" / name
            if alt_path.exists(): target_path = alt_path

        if not target_path.exists():
            if os.path.exists(name): target_path = Path(name)
            else: raise FileNotFoundError(f"❌ 找不到数据集目录: {target_path}")

        # 智能搜索子目录
        is_dataset_root = (target_path / "dataset_info.json").exists() or \
                          (target_path / "state.json").exists()
        final_path = target_path

        if not is_dataset_root and target_path.is_dir():
            found_sub = False
            for sub_dir in target_path.iterdir():
                if sub_dir.is_dir() and ((sub_dir / "dataset_info.json").exists() or \
                                         (sub_dir / "state.json").exists()):
                    final_path = sub_dir
                    found_sub = True
                    break 
            if not found_sub:
                print(f"⚠️ 警告: 未找到标准 HuggingFace 结构，尝试根目录...")

        try:
            self.hf_dataset = datasets.load_from_disk(str(final_path)).with_format("numpy")
        except Exception as e:
            raise FileNotFoundError(f"❌ 加载失败 {final_path}: {e}")

        if self.force_univariate:
            self.has_past_feat = False
        else:
            self.has_past_feat = "past_feat_dynamic_real" in self.hf_dataset.column_names

        raw_freq = self.hf_dataset[0]["freq"]
        self._clean_freq = clean_frequency(raw_freq)
    
    @cached_property
    def prediction_length(self) -> int:
        freq_str = norm_freq_str(to_offset(self._clean_freq).name)
        freq_str = maybe_reconvert_freq(freq_str)
        
        if freq_str not in PRED_LENGTH_MAP and freq_str not in M4_PRED_LENGTH_MAP:
            fallback = self._clean_freq[0]
            if fallback in PRED_LENGTH_MAP: freq_str = fallback

        mapping = M4_PRED_LENGTH_MAP if "m4" in self.name.lower() else PRED_LENGTH_MAP
        base_len = mapping.get(freq_str, 48)
        
        return self.term.multiplier * base_len

    @property
    def freq(self) -> str:
        return self._clean_freq

    @property
    def target_dim(self) -> int:
        if self.force_univariate: return 1
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if target.ndim > 1 else 1

    @property
    def past_feat_dynamic_real_dim(self) -> int:
        if self.force_univariate or not self.has_past_feat: return 0
        first = self.hf_dataset[0]
        past_feat_dynamic_real = first["past_feat_dynamic_real"]
        return past_feat_dynamic_real.shape[0] if past_feat_dynamic_real.ndim > 1 else 1

    @cached_property
    def windows(self) -> int:
        return MAX_WINDOW

    @cached_property
    def _min_series_length(self) -> int:
        if self.force_univariate:
            sample = self.hf_dataset[0]["target"]
            return sample.shape[-1] 
        column = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(column, 0, 1)))
        else:
            lengths = pc.list_value_length(column)
        return int(min(lengths.to_numpy()))
    @classmethod
    def from_memory(cls, entries, freq="H", prediction_length=96):
        """
        [新增] 从内存数据直接构造 Dataset 实例，供 TSFM Predictor 使用。
        entries: List[dict], 形如 [{'target': np.array, 'start': pd.Period, 'item_id': '...'}]
        """
        instance = cls.__new__(cls)
        # 伪造必要的属性，绕过磁盘读取逻辑
        instance.gluonts_dataset = entries
        instance.test_data = entries
        instance.freq = freq
        instance.prediction_length = prediction_length
        # 如果原始 Dataset 类有其他强依赖的属性，可以在这里补上 (例如 target_dim 等)
        instance.target_dim = 1 if len(entries[0]['target'].shape) == 1 else entries[0]['target'].shape[1]
        return instance
    # --------- 核心转换逻辑 ---------

    @cached_property
    def gluonts_dataset(self):
        """转换为 GluonTS 数据集格式: 清洗频率 + 转 Period + 长度初筛"""
        final_data = []
        target_freq = self.freq
        required_len = self.prediction_length # 只要够 1 个窗口长度就可以保留
        
        skipped_count = 0
        
        if self.force_univariate:
            for row in self.hf_dataset:
                raw_target = np.array(row["target"], dtype=np.float32)
                
                # 过滤极短序列
                seq_len = raw_target.shape[-1]
                if seq_len < required_len:
                    skipped_count += 1
                    continue

                raw_start = row["start"].item() if hasattr(row["start"], "item") else row["start"]
                if isinstance(raw_start, (int, float)):
                    if raw_start > 3e11: ts = pd.to_datetime(raw_start, unit='ns')
                    else: ts = pd.to_datetime(raw_start, unit='s')
                else:
                    ts = pd.Timestamp(raw_start)
                
                period_start = pd.Period(ts, freq=target_freq)
                base_id = str(row["item_id"])

                if raw_target.ndim > 1:
                    C = raw_target.shape[0]
                    for i in range(C):
                        final_data.append({
                            "start": period_start,
                            "target": raw_target[i],
                            "item_id": f"{base_id}_dim{i}",
                            "parent_item_id": base_id,
                            "channel_id": i,
                            "freq": target_freq
                        })
                else:
                    final_data.append({
                        "start": period_start,
                        "target": raw_target,
                        "item_id": base_id,
                        "parent_item_id": base_id,
                        "channel_id": 0,
                        "freq": target_freq
                    })
            
            if skipped_count > 0:
                print(f"⚠️ [Dataset] 剔除了 {skipped_count} 条无法支持单个预测窗口的极短序列。")
            return final_data

        else:
            def filter_short(entry):
                t = entry['target']
                l = t.shape[-1] if hasattr(t, 'shape') else len(t)
                return l >= required_len

            def itemize_start_with_freq(entry):
                return itemize_start(entry, freq=target_freq)

            filtered_ds = filter(filter_short, self.hf_dataset)
            process = ProcessDataEntry(target_freq, one_dim_target=self.target_dim == 1)
            ds = Map(compose(process, itemize_start_with_freq), filtered_ds)
            if self.to_univariate and self.target_dim > 1:
                ds = MultivariateToUnivariate("target").apply(ds)
            return ds

    # --------- GluonTS split 接口 (重写版) ---------

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1))
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        return validation_dataset

    @property
    def test_data(self):
        """
        返回自定义的 AdaptiveTestData 对象，
        它拥有 .input 和 .label 属性，且支持 len() 操作。
        """
        return AdaptiveTestData(self, max_window=self.windows)


# ====================== GluonTS <-> NumPy 工具 ======================
# (保持不变)

def gluonts_to_numpy(input_dataset: List[SampleForecast]):
    data_list: List[np.ndarray] = []
    for forecast in input_dataset:
        data_list.append(forecast.samples.T)
    data_array = np.stack(data_list, axis=0)
    return data_array

def load_forecasts_from_npy(samples_path: str, meta_path: str, freq: str) -> List[SampleForecast]:
    samples = np.load(samples_path)
    with open(meta_path, "r") as fp: meta = json.load(fp)
    if len(meta) != samples.shape[0]: raise ValueError("样本数与元数据条目数不匹配")
    forecasts: List[SampleForecast] = []
    for idx, info in enumerate(meta):
        item_id = info["item_id"]
        start_val = info["start_date"]
        start_date = pd.Period(start_val, freq=freq)
        sample_arr = samples[idx]
        sf = SampleForecast(samples=sample_arr, start_date=start_date, item_id=item_id)
        forecasts.append(sf)
    return forecasts

def load_gluonts_pred(base_path: str, model_name: str, model_cl_name: str, dataset_name: str, pred_len, channels, windows, verobse=False):
    model_folder = os.path.join(base_path, model_name)
    model_cl_folder = os.path.join(model_folder, model_cl_name)
    samples_path = os.path.join(model_cl_folder, f"{dataset_name}_samples.npy")
    meta_path = os.path.join(model_cl_folder, f"{dataset_name}_meta.json")
    samples = np.load(samples_path)
    with open(meta_path, "r") as fp: meta = json.load(fp)
    entries = meta.get("entries", None)
    performance = meta.get("performance", None) or {}
    runtime_seconds = performance.get("runtime_seconds", None)
    median_forecast = np.median(samples, axis=1)
    if median_forecast.ndim == 2:
        try:
            median_forecast = median_forecast.reshape(-1, channels, windows, pred_len)
            median_forecast = median_forecast.transpose(0, 2, 3, 1)
            median_forecast = median_forecast.reshape(-1, pred_len, channels)
        except ValueError:
            median_forecast = median_forecast.reshape(-1, pred_len, 1)
    freq = dataset_name.split("_")[-2] if "_" in dataset_name else "H"
    pred_dataset: List[SampleForecast] = []
    n_series = median_forecast.shape[0]
    for idx in range(n_series):
        if entries is None or idx >= len(entries): break
        info = entries[idx]
        item_id = str(info["item_id"])
        start = pd.Period(info["start_date"], freq=freq)
        target_array = median_forecast[idx]
        target = target_array.T
        forecast = SampleForecast(samples=target, start_date=start, item_id=item_id)
        pred_dataset.append(forecast)
    return pred_dataset, samples, runtime_seconds

def numpy_to_gluonts(data_array, template_dataset):
    forecasts: List[SampleForecast] = []
    N, T, C = data_array.shape
    for i in range(min(N, len(template_dataset))):
        base_sample = template_dataset[i]
        item_id = base_sample.item_id
        start_date = base_sample.start_date
        sample_array = data_array[i]
        if C > 1: sample_array = sample_array.reshape(1, T, C)
        else: sample_array = sample_array.reshape(1, T)
        forecast = SampleForecast(samples=sample_array.astype(np.float32), start_date=start_date, item_id=item_id)
        forecasts.append(forecast)
    return forecasts
