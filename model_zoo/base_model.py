import os
import csv
import json
import time
import warnings
import random
# 在文件头部
from utils.path_utils import get_correction_data_dir
import pickle
from pathlib import Path


from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datetime import datetime
import pickle
from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MASE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

from utils.data import Dataset
from utils.debug import debug_check_input_nan, debug_print_test_input, debug_forecasts

from Model_Path.model_zoo_config import MULTIVAR_TSFM_PREFIXES
from selector.selector_config import Selector_zoo_details

warnings.filterwarnings("ignore")
load_dotenv()  # 加载环境变量

# ====================== 全局配置与常量 ======================

# 数据集属性（domain、变量维度、freq 等）
# dataset_properties_map = json.load(open("Dataset_Path/dataset_properties.json", encoding="utf-8"))
# 确保此路径存在，或者根据项目结构调整
dataset_properties_map = json.load(open("Datasets/processed_datasets/dataset_properties.json", encoding="utf-8"))
# 评估指标集合
metrics = [
    MASE(),
    SMAPE(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1 * i for i in range(1, 10)]
    ),
]

# 数据集重命名
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class BaseModel:
    def __init__(self, model_name, args, output_dir=None):
        self.args = args
        self.model_name = model_name
        if output_dir is None:
            self.output_dir = args.output_dir
        self.batch_size = args.batch_size

        self.get_save_path()
        print('Save Path: ', self.csv_file_path)

        self.done_datasets = []
        if self.args.skip_saved:
            if os.path.exists(self.csv_file_path):
                df_res = pd.read_csv(self.csv_file_path)
                if "dataset" in df_res.columns:
                    self.done_datasets = df_res["dataset"].values

                    print(f"Done {len(self.done_datasets)} datasets")
            else:
                print(f"[skip_saved] 结果文件不存在，忽略已完成数据集检测：{self.csv_file_path}")

    def get_save_path(self):

        os.makedirs(self.output_dir, exist_ok=True)

        if self.args.run_mode == "zoo":
            if self.args.fix_context_len:
                self.model_cl_name = f"cl_{self.args.context_len}"
            else:
                self.model_cl_name = "cl_original"
            self.output_dir = os.path.join(self.output_dir, self.model_cl_name)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            self.csv_file_path = os.path.join(self.output_dir, "all_results.csv")

        elif self.args.run_mode == "select":
            cfg = Selector_zoo_details.get(self.model_name, None)
            if cfg is None:
                raise ValueError(f"⚠️ 未知 selector 模型 {self.model_name}，请在 selector_config.py 中补充配置")

            tpl = cfg["csv_name_tpl"]

            # ⭐ 这里统一组织 format 所需的字段：
            filename = tpl.format(
                current_zoo_num=self.args.current_zoo_num,
                zoo_total_num=self.args.zoo_total_num,
                ensemble_size=self.args.ensemble_size,
                seed=getattr(self.args, "seed", None),
                real_order_metric=getattr(self.args, "real_order_metric", None),
            )

            self.csv_file_path = os.path.join(self.output_dir, filename)

        else:
            raise ValueError('⚠️ 未知运行模式，仅支持 zoo / select')

        header = [
            "dataset",
            "model",
            "MASE",
            "sMAPE",
            "CRPS",
            "domain",
            "num_variates",
            "model_order"
        ]

        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

    def get_predictor(self, dataset, batch_size):
        """
        子类实现：加载模型，返回 predictor 对象
        """
        raise NotImplementedError("子类必须实现 get_predictor 方法")

    def _build_ds_meta(self, ds_name, term):
        """统一解析 ds_name，返回 ds_key, ds_freq, ds_config, dataset_name"""
        if "/" in ds_name:
            ds_key_raw, ds_freq = ds_name.split("/")
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
        else:
            ds_key_raw = ds_name
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"
        dataset_name = f"{ds_key}_{ds_freq}_{term}"
        return ds_key, ds_freq, ds_config, dataset_name

    def _should_force_univariate(self):
        """
        根据模型类型决定是否开启 force_univariate 模式。
        如果模型不支持多变量（不在白名单中）且不是 select 模式，则强制转为单变量。
        """
        if self.args.run_mode == "select":
            return False
            
        prefix = self.model_name.split("_")[0].lower()
        if prefix in MULTIVAR_TSFM_PREFIXES:
            return False
            
        return True

    def _make_forecasts(self, dataset, dataset_name, ds_config, fixed_model_order, debug_mode):
        """
        统一的预测入口：
        - 选择器：返回 (forecasts, model_order)
        - 非选择器：内部处理 OOM 重试、debug 打印、NaN 检查和噪声注入
        """
        model_order = None
        batch_size = self.batch_size

        if self.args.run_mode == "zoo":
            while True:
                try:
                    predictor = self.get_predictor(dataset, batch_size)
                    test_input = dataset.test_data.input

                    if debug_mode:
                        # 打印数据格式
                        debug_print_test_input(dataset)
                        debug_check_input_nan(test_input)

                    input_data = test_input
                    forecasts = list(
                        tqdm(
                            predictor.predict(input_data),
                            total=len(dataset.test_data.input),
                            desc=f"Predicting {ds_config}",
                        )
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"⚠️ OOM at batch_size {batch_size}, "
                        f"reducing to {batch_size // 2}"
                    )
                    batch_size //= 2

        elif self.args.run_mode == "select":
            predictor = self.get_predictor(dataset, batch_size)
            forecast_iter, model_order = predictor.predict(
                dataset.test_data.input, dataset_name, fixed_model_order
            )
            forecasts = list(forecast_iter)

        else:
            raise ValueError('⚠️ 未知运行模式，仅支持 zoo / select')

        if debug_mode:
            debug_forecasts(forecasts)

        return forecasts, model_order

    # ==============================================================
    # 主运行流程
    # ==============================================================

    def run(self):
        total_time = 0
        total_memory = 0
        max_memory = 0

        if self.model_name == "Random_Select":
            # 提前固定，防止不同数据集使用的模型顺序不一致
            fixed_model_order = list(range(self.args.current_zoo_num))
            random.shuffle(fixed_model_order)
        else:
            fixed_model_order = None

        print(f"🚀 Running {self.model_name}", )

        if len(self.done_datasets) > 0 and self.args.skip_saved:
            print(f"✅  Skipping...✅  Done with {len(self.done_datasets)} datasets. ")
        self.all_data_configs = []
        for ds_name in self.args.all_datasets:
            terms = ["short", "medium", "long"]
            for term in terms:
                # 中长 term 只对指定数据集生效
                if (term in ["medium", "long"]) and (ds_name not in self.args.med_long_datasets.split()):
                    continue

                # 统一构造 ds_key / ds_freq / ds_config / dataset_name
                ds_key, ds_freq, ds_config, dataset_name = self._build_ds_meta(ds_name, term)
                self.all_data_configs.append(ds_config)

                if ds_config in self.done_datasets and getattr(self.args, "skip_saved", False):
                    print(f"{ds_config}.", end=" ✅  ")
                    continue
                else:
                    print(f"\n🚀 Dataset: [{ds_config}]",
                          f"Model: {self.model_name}",
                          'GPU:', os.environ.get('CUDA_VISIBLE_DEVICES', 'None'),
                          'Batch_size:', self.batch_size,
                          'num_workers:', self.args.num_workers
                          )
                
                # ---------- 1. 决定数据加载模式 ----------
                # 根据模型是否支持多变量，决定是否开启 force_univariate
                force_univariate = self._should_force_univariate()
                
                # ---------- 2. 实例化 Dataset (使用 data.py 的新接口) ----------
                dataset = Dataset(
                    name=ds_name, 
                    term=term, 
                    force_univariate=force_univariate
                )
                
                start_time = time.time()

                # ---------- 3. 执行预测 ----------
                forecasts, model_order = self._make_forecasts(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    ds_config=ds_config,
                    fixed_model_order=fixed_model_order,
                    debug_mode=self.args.debug_mode,
                )
                
                # ---------- 4. 评估指标 ----------
                res = evaluate_forecasts(
                    forecasts=forecasts,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=1024,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(dataset.freq),
                )

                # ===================== 记录耗时和显存 =====================
                end_time = time.time()
                elapsed = end_time - start_time
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                memory_used = reserved + allocated

                max_memory = max(max_memory, memory_used)
                total_memory += memory_used
                total_time += elapsed

                print(f"time cost 🧭 {elapsed:.2f}s",
                      f"memory-use {memory_used:.0f} MB", end=' ')

                if self.args.save_pred:
                    self.save_results(res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used, dataset, model_order)
                
                # =========================================================
                # [关键修正] 导出校正数据
                # =========================================================
                if getattr(self.args, "save_for_correction", False):
                    # A. 计算点预测 (Point Predictions) 用于计算残差
                    preds = []
                    for f in forecasts:
                        if hasattr(f, "samples"): # SampleForecast
                            preds.append(np.mean(f.samples, axis=0))
                        elif hasattr(f, "mean"): # DistributionForecast / QuantileForecast
                            preds.append(f.mean)
                        elif hasattr(f, "forecast_array"):
                             preds.append(np.mean(f.forecast_array, axis=0))
                        else:
                            try:
                                preds.append(f.mean)
                            except:
                                raise ValueError(f"无法从 Forecast 对象中提取点预测值: {type(f)}")
                    
                    # B. 获取上下文长度
                    ctx_len = getattr(self.args, 'context_len', 512)

                    # C. 调用封装好的导出函数
                    self._export_correction_data(
                        test_dataset=dataset.test_data, 
                        forecasts=forecasts, 
                        preds=preds, 
                        dataset_name=dataset_name, 
                        ds_config=ds_config,
                        context_len=ctx_len
                    )

        # ===================== 运行结束后：统计总体性能并检查结果文件 =====================
        num_ds = len(self.all_data_configs)
        if num_ds - len(self.done_datasets) > 0 and self.args.save_pred:
            average_time = total_time / max(num_ds, 1)
            average_memory = total_memory / max(num_ds, 1)

            print(f"\n🧭 已运行{num_ds}个数据集，total_time:", f"{total_time:.2f}s", "average_time:", f"{average_time:.2f}s",
                  "max_memory:", f"{max_memory:.0f} MB", "average_memory:", f"{average_memory:.0f} MB \n", )

            time_save_filename = "results/runtime-TSFM.csv"

            if self.args.fix_context_len:
                context_tag = self.args.context_len
            else:
                context_tag = "original"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.isfile(time_save_filename)

            row = {
                "model_name": self.model_name,
                "context_length": context_tag,
                "dataset_num": num_ds,
                "total_time_s": round(total_time, 0),
                "average_time_s": round(average_time, 2),
                "average_memory_MB": round(average_memory, 0),
                "timestamp": timestamp,
            }

            with open(time_save_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    # ==============================================================
    # 结果保存逻辑
    # ==============================================================
    def save_results(self, res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used, dataset=None, model_order=None):

        formatted_model_order = '[' + " ".join(map(str, model_order)) + ']' if model_order is not None else ""

        row = [
            ds_config,
            self.model_name,
            res["MASE[0.5]"][0],
            res["sMAPE[0.5]"][0],
            res["mean_weighted_sum_quantile_loss"][0],
            dataset_properties_map[ds_key]["domain"],
            dataset_properties_map[ds_key]["num_variates"],
            formatted_model_order
        ]

        with open(self.csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)


        if res is not None:
            print(
                f"Saved metrics:[",
                f"MASE: {res['MASE[0.5]'][0]:.2f}",
                f"sMAPE: {res['sMAPE[0.5]'][0]:.2f}",
                f"CRPS: {res['mean_weighted_sum_quantile_loss'][0]:.2f}"
                f"] to {self.csv_file_path}")
        else:
            print(f"{self.model_name} No evaluation results.")

        # 保存预测结果到npy和json
        if self.args.run_mode == "zoo":
            # 1) 样本数组：shape = (num_series, num_samples, pred_len, num_channels)
            if hasattr(forecasts[0], "samples"):
                arrs = [fc.samples for fc in forecasts]
            elif hasattr(forecasts[0], "forecast_array"):
                arrs = [fc.forecast_array for fc in forecasts]
            else:
                print(f"forecasts[0] attributes: {dir(forecasts[0])}")
                raise ValueError("forecasts[0] does not have 'samples' or 'forecast_array' attribute")
            samples = np.stack(arrs, axis=0)

            samples_path = os.path.join(self.output_dir, f"{dataset_name}_samples.npy")

            np.save(samples_path, samples)

            # 2) 保存元数据 + 性能指标
            meta = {
                "performance": {
                    "runtime_seconds": elapsed,
                    "memory_use_mb": memory_used,
                    "batch_size": self.batch_size,
                },
                "entries": [
                    {
                        "item_id": fc.item_id,
                        "start_date": str(fc.start_date)
                    }
                    for fc in forecasts
                ]
            }
            meta_path = os.path.join(self.output_dir, f"{dataset_name}_meta.json")
            with open(meta_path, "w") as fp:
                json.dump(meta, fp)
            print(f"👉 预测结果npy和json元数据保存到 {self.output_dir}")

    # def _export_correction_data(self, test_dataset, forecasts, preds, dataset_name, ds_config, context_len):
    #     """
    #     [增强版] 导出清洗后的校正数据
    #     1. 剔除过短序列 (防止 excessive zero-padding)
    #     2. 修复 NaN (线性插值)
    #     3. 剔除 Inf 和 全常数序列
    #     """
    #     # === 配置清洗阈值 ===
    #     # 最小历史长度：至少要有预测长度的 1/2 或固定值 (如 48)，避免极短序列
    #     MIN_HIST_LEN = max(48, int(forecasts[0].prediction_length * 0.5)) 
    #     MAX_NAN_RATIO = 0.2  # 允许的最大缺失值比例
        
    #     print(f"💾 [Data Export] 正在导出并清洗数据... (Min Len: {MIN_HIST_LEN})")
        
    #     timestamps = []
    #     histories = []
    #     truths = []
    #     clean_preds = []
        
    #     # 统计计数器
    #     stats = {
    #         "total": 0, "saved": 0,
    #         "drop_short": 0, "drop_nan": 0, "drop_inf": 0, "drop_constant": 0
    #     }
        
    #     # 对齐遍历
    #     # 注意：preds 是之前计算好的点预测列表
    #     iter_data = zip(test_dataset, forecasts, preds)
        
    #     for item, forecast, pred_val in iter_data:
    #         stats["total"] += 1
            
    #         # --- 1. 提取原始数据 ---
    #         ts = forecast.start_date
    #         if hasattr(ts, 'to_timestamp'):
    #             ts = ts.to_timestamp()
                
    #         if isinstance(item, tuple):
    #             input_entry, label_entry = item
    #             raw_hist = input_entry["target"]
    #             raw_gt = label_entry["target"]
    #         else:
    #             full_target = item["target"]
    #             p_len = forecast.prediction_length
    #             raw_gt = full_target[-p_len:]
    #             raw_hist = full_target[:-p_len]
            
    #         # 转换为 float32 numpy 以便处理
    #         hist = np.array(raw_hist, dtype=np.float32)
    #         gt = np.array(raw_gt, dtype=np.float32)
            
    #         # --- 2. [清洗] NaN 处理 (线性插值) ---
    #         def fill_nan_linear(arr):
    #             mask = np.isnan(arr)
    #             if not mask.any(): return arr
    #             # 如果 NaN 太多，返回 None 标记为丢弃
    #             if mask.mean() > MAX_NAN_RATIO: return None
                
    #             # 简单线性插值
    #             x = np.arange(len(arr))
    #             arr[mask] = np.interp(x[mask], x[~mask], arr[~mask])
    #             return arr

    #         hist = fill_nan_linear(hist)
    #         gt = fill_nan_linear(gt)
            
    #         if hist is None or gt is None:
    #             stats["drop_nan"] += 1
    #             continue

    #         # --- 3. [清洗] 长度检查 (关键!) ---
    #         # 如果有效历史长度太短，Dataset.py 会补大量 0，导致脏数据 -> 剔除
    #         if len(hist) < MIN_HIST_LEN:
    #             stats["drop_short"] += 1
    #             continue
                
    #         # --- 4. [清洗] Inf / 极端值检查 ---
    #         if np.isinf(hist).any() or np.isinf(gt).any() or np.abs(hist).max() > 1e9:
    #             stats["drop_inf"] += 1
    #             continue
                
    #         # --- 5. [清洗] 常数序列检查 ---
    #         # 历史全是同一个值 (方差为0)，通常是死线，无法学习 -> 剔除
    #         if np.std(hist) < 1e-6:
    #             stats["drop_constant"] += 1
    #             continue

    #         # --- 6. 数据截断 (保留最近的 context_len) ---
    #         # 注意：只有在通过长度检查后才截断，保证保留的数据是真实的
    #         if len(hist) > context_len:
    #             hist = hist[-context_len:]
                
    #         # --- 7. 加入结果列表 ---
    #         timestamps.append(ts)
    #         histories.append(hist)
    #         truths.append(gt)
    #         clean_preds.append(pred_val)
    #         stats["saved"] += 1

    #     # 长度再次对齐 (理论上是一致的，防御性编程)
    #     min_len = min(len(truths), len(clean_preds))
    #     histories = histories[:min_len]
    #     truths = truths[:min_len]
    #     clean_preds = clean_preds[:min_len]
    #     timestamps = timestamps[:min_len]
        
    #     # 计算残差
    #     residuals = []
    #     for t, p in zip(truths, clean_preds):
    #         if isinstance(p, np.ndarray) and isinstance(t, np.ndarray):
    #             if p.size == t.size: p = p.reshape(t.shape)
    #             elif p.ndim > t.ndim: p = p.mean(axis=0).reshape(t.shape)
    #         res = t - p
    #         residuals.append(res)

    #     # 打印清洗报告
    #     print(f"📊 [Cleaning Report] {dataset_name}")
    #     print(f"   Total: {stats['total']} -> Saved: {stats['saved']}")
    #     print(f"   Dropped: Short={stats['drop_short']}, NaN={stats['drop_nan']}, "
    #           f"Inf={stats['drop_inf']}, Const={stats['drop_constant']}")

    #     if stats['saved'] == 0:
    #         print(f"⚠️ 警告: 数据集 {dataset_name} 所有样本均被过滤，将跳过保存。")
    #         return

    #     # 构造并保存
    #     data_dump = {
    #         "dataset_name": dataset_name,
    #         "source": dataset_name,
    #         "config": ds_config,
    #         "timestamps": timestamps,
    #         "histories": np.array(histories, dtype=object),
    #         "truths": np.array(truths, dtype=object),
    #         "preds": clean_preds,
    #         "residuals": np.array(residuals, dtype=object)
    #     }
        
    #     output_root = getattr(self.args, 'output_root', 'correction_datasets')
    #     model_id = f"{self.args.model_family}_{self.args.model_size}" if hasattr(self.args, 'model_family') else self.model_name
        
    #     save_dir = get_correction_data_dir(
    #         output_root=output_root,
    #         model_id=model_id,
    #         context_len=context_len,
    #         fix_context=False
    #     )
        
    #     file_name = f"{dataset_name}_correction_data.pkl"
    #     save_path = os.path.join(save_dir, file_name)
        
    #     try:
    #         os.makedirs(save_dir, exist_ok=True)
    #         with open(save_path, "wb") as f:
    #             pickle.dump(data_dump, f)
    #         print(f"✅ 已保存清洗后的数据: {save_path}")
    #     except Exception as e:
    #         print(f"❌ 保存失败: {e}")
    def _export_correction_data(self, test_dataset, forecasts, preds, dataset_name, ds_config, context_len, 
                                local_preds=None, local_truths=None):
        """
        [增强版 V2] 导出清洗后的校正数据，支持“局部残差”数据。
        """
        # === 配置清洗阈值 ===
        MIN_HIST_LEN = max(48, int(forecasts[0].prediction_length * 0.5)) 
        MAX_NAN_RATIO = 0.2
        
        print(f"💾 [Data Export V2] 正在导出并清洗数据 (含局部残差)... (Min Len: {MIN_HIST_LEN})")
        
        # 主数据容器
        data_storage = {
            "timestamps": [],
            "histories": [],
            "truths": [],     # 未来真值
            "preds": [],      # 未来预测
            "local_residuals": [], # [新增] 局部残差指纹
        }
        
        # === [修复点] 初始化所有用到的统计键 ===
        stats = {
            "total": 0, 
            "saved": 0, 
            "drop": 0,          # 通用丢弃计数
            "drop_short": 0,    # 长度不足
            "drop_nan": 0,      # 含过多NaN
            "drop_inf": 0,      # 含无穷大
            "drop_constant": 0  # 常数序列(方差为0)
        }
        
        # 确保 local 数据存在
        has_local = (local_preds is not None) and (local_truths is not None)
        if not has_local:
            print("⚠️ 警告: 未提供 local_preds/local_truths，将无法生成局部残差指纹！")

        # 迭代器构建
        main_iter = zip(test_dataset, forecasts, preds)
        
        for idx, (item, forecast, pred_val) in enumerate(main_iter):
            stats["total"] += 1
            
            # --- 1. 提取基础数据 ---
            ts = forecast.start_date
            if hasattr(ts, 'to_timestamp'): ts = ts.to_timestamp()
                
            if isinstance(item, tuple):
                input_entry, label_entry = item
                raw_hist = input_entry["target"]
                raw_gt = label_entry["target"]
            else:
                full_target = item["target"]
                p_len = forecast.prediction_length
                raw_gt = full_target[-p_len:]
                raw_hist = full_target[:-p_len]
            
            hist = np.array(raw_hist, dtype=np.float32)
            gt = np.array(raw_gt, dtype=np.float32)
            
            # --- 2. 处理 Local 数据 (修复维度不匹配问题) ---
            loc_res = None
            if has_local:
                l_pred = local_preds[idx]
                l_true = local_truths[idx]
                
                if l_pred is None or l_true is None:
                    stats["drop"] += 1; continue 
                
                if not isinstance(l_pred, np.ndarray): l_pred = np.array(l_pred)
                if not isinstance(l_true, np.ndarray): l_true = np.array(l_true)

                # A. 处理多样本维度
                if l_pred.ndim > l_true.ndim: 
                    l_pred = l_pred.mean(axis=0)

                # B. 长度截断对齐
                if l_pred.shape[0] > l_true.shape[0]:
                    l_pred = l_pred[:l_true.shape[0]]
                elif l_pred.shape[0] < l_true.shape[0]:
                    stats["drop"] += 1; continue
                
                # C. 形状严格对齐
                if l_pred.shape != l_true.shape:
                    l_pred = l_pred.reshape(l_true.shape)
                
                loc_res = l_true - l_pred
                
                if np.isnan(loc_res).any() or np.isinf(loc_res).any():
                     stats["drop"] += 1; continue

            # --- 3. 基础清洗逻辑 ---
            def fill_nan_linear(arr):
                mask = np.isnan(arr)
                if not mask.any(): return arr
                if mask.mean() > MAX_NAN_RATIO: return None
                x = np.arange(len(arr))
                arr[mask] = np.interp(x[mask], x[~mask], arr[~mask])
                return arr

            hist = fill_nan_linear(hist)
            gt = fill_nan_linear(gt)
            
            if hist is None or gt is None:
                stats["drop_nan"] += 1
                continue

            # --- [清洗] 长度检查 ---
            if len(hist) < MIN_HIST_LEN:
                stats["drop_short"] += 1
                continue
                
            # --- [清洗] Inf / 极端值检查 ---
            if np.isinf(hist).any() or np.isinf(gt).any() or np.abs(hist).max() > 1e9:
                stats["drop_inf"] += 1
                continue
                
            # --- [清洗] 常数序列检查 ---
            if np.std(hist) < 1e-6:
                stats["drop_constant"] += 1
                continue

            # --- 4. 截断与保存 ---
            if len(hist) > context_len:
                hist = hist[-context_len:]
                
            data_storage["timestamps"].append(ts)
            data_storage["histories"].append(hist)
            data_storage["truths"].append(gt)
            data_storage["preds"].append(pred_val)
            if has_local:
                data_storage["local_residuals"].append(loc_res)
                
            stats["saved"] += 1

        # 计算主任务残差
        residuals = []
        for t, p in zip(data_storage["truths"], data_storage["preds"]):
            if isinstance(p, np.ndarray) and isinstance(t, np.ndarray):
                if p.size == t.size: p = p.reshape(t.shape)
                elif p.ndim > t.ndim: p = p.mean(axis=0).reshape(t.shape)
            residuals.append(t - p)

        print(f"📊 [Report] {dataset_name}: Saved {stats['saved']}/{stats['total']}")
        # 打印详细丢弃原因
        print(f"   Dropped: Short={stats['drop_short']}, NaN={stats['drop_nan']}, "
              f"Inf={stats['drop_inf']}, Const={stats['drop_constant']}, LocalErr={stats['drop']}")

        if stats['saved'] == 0: return

        # 构造 Dump 对象
        data_dump = {
            "dataset_name": dataset_name,
            "source": dataset_name,
            "config": ds_config,
            "timestamps": data_storage["timestamps"],
            "histories": np.array(data_storage["histories"], dtype=object),
            "truths": np.array(data_storage["truths"], dtype=object),
            "preds": data_storage["preds"],
            "residuals": np.array(residuals, dtype=object),
            "local_residuals": np.array(data_storage["local_residuals"], dtype=object) if has_local else None
        }
        
        output_root = getattr(self.args, 'output_root', 'correction_datasets')
        model_id = f"{self.args.model_family}_{self.args.model_size}" if hasattr(self.args, 'model_family') else self.model_name
        
        save_dir = get_correction_data_dir(output_root=output_root, model_id=model_id, context_len=context_len, fix_context=False)
        file_name = f"{dataset_name}_correction_data.pkl"
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, file_name), "wb") as f:
                pickle.dump(data_dump, f)
            print(f"✅ 已保存 (含局部特征): {file_name}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")