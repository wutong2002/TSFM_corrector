import os
import warnings
import torch
from torch import cuda
from typing import List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from model_zoo.base_model import BaseModel
from utils.data import gluonts_to_numpy, numpy_to_gluonts, load_gluonts_pred


class Baseline_Select_Model(BaseModel):
    def __init__(self, args, model_name, Model_zoo_current):
        self.args = args
        self.model_name = model_name
        self.Model_sizes = Model_zoo_current
        self.output_dir = os.path.join("results", self.model_name)
        super().__init__(self.model_name, args, self.output_dir)
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        # 生成模型ID到模型缩写的映射
        self.abbr_to_id = {
            model_info["abbreviation"]: model_info["id"]
            for family in self.Model_sizes.values()
            for model_info in family.values()
        }

        if self.args.fix_context_len:
            self.model_cl_name = f"cl_{self.args.context_len}"
        else:
            self.model_cl_name = "cl_original"

    def get_predictor(self, dataset, batch_size):
        """创建预测器，子类需指定选择策略"""
        self.args.test_pred_len = dataset.prediction_length
        return Baseline_Select_Predictor(
            args=self.args,
            Model_sizes=self.Model_sizes,
            prediction_length=dataset.prediction_length,
            channels=dataset.target_dim,
            windows=dataset.windows,
            model_cl_name=self.model_cl_name,
            select_strategy=self._get_select_strategy(dataset)  # 子类指定选择策略
        )

    def _get_select_strategy(self, dataset):
        """子类实现此方法返回选择策略函数"""
        raise NotImplementedError


class Baseline_Select_Predictor:
    """通用预测器，支持多种选择策略"""

    def __init__(
            self, args, Model_sizes,
            prediction_length: int,
            channels, windows, model_cl_name,
            select_strategy: callable  # 选择策略函数
    ):
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.prediction_length = prediction_length
        self.args = args
        self.Model_sizes = Model_sizes
        self.channels = channels
        self.windows = windows
        self.model_cl_name = model_cl_name
        self.select_strategy = select_strategy

        self.id_to_abbr = {
            model_info["id"]: model_info["abbreviation"]
            for family in self.Model_sizes.values()
            for model_info in family.values()
        }

    def predict(self, test_data_input: List[dict], dataset_name, model_order: Optional[List[int]] = None) -> List:
        """执行预测，支持传入预定义的model_order或使用选择策略生成"""
        preds_dict, dataset_pred_per_model_gluonts = self._load_predictions(dataset_name)

        if model_order is None:
            # 如果没有传入model_order，则使用选择策略生成
            model_order, ensemble_size = self.select_strategy(dataset_name)
        else:
            _, ensemble_size = self.select_strategy(dataset_name)

        # 集成预测
        preds = self._create_ensemble(preds_dict, model_order, ensemble_size)

        if self.args.debug_mode:
            print(f"Predictions shape: {np.shape(preds)}")

        # 转换预测结果为gluonts格式
        forecasts = numpy_to_gluonts(preds, dataset_pred_per_model_gluonts)

        return forecasts, model_order

    def _load_predictions(self, dataset_name):
        """加载所有TSFM的预测数据"""
        preds_dict = {}
        for family, size_dict in self.Model_sizes.items():
            for size_name in size_dict.keys():
                model_name = f"{family}_{size_name}"
                model_idx = size_dict[size_name]['id']

                dataset_pred_per_model_gluonts, pred_npy, runtime_seconds = load_gluonts_pred(
                    "results", model_name, self.model_cl_name, dataset_name, self.prediction_length,
                    self.channels, self.windows
                )
                dataset_pred_per_model = gluonts_to_numpy(dataset_pred_per_model_gluonts)

                if dataset_pred_per_model.shape[-1] == 1 and self.channels > 1:
                    dataset_pred_per_model = dataset_pred_per_model.reshape(
                        -1, self.channels, dataset_pred_per_model.shape[1]
                    ).transpose(0, 2, 1)

                preds_dict[model_idx] = dataset_pred_per_model

        return preds_dict, dataset_pred_per_model_gluonts

    def _create_ensemble(self, preds_dict, model_order, ensemble_size):
        """集成预测，简单平均法"""
        ensemble_preds = []
        print(f"Selected Top-{ensemble_size} Models : [", end=' ')

        for model_idx in range(ensemble_size):
            channel_select_model_id = model_order[model_idx]
            select_model_name = self.id_to_abbr.get(channel_select_model_id)
            print(f"{select_model_name}", end=" ")

            cur_pred_full = preds_dict[channel_select_model_id]
            ensemble_size_pred = []

            for cur_channel in range(self.channels):
                cur_pred = cur_pred_full[:, :, cur_channel]  # (N, pred_len)
                ensemble_size_pred.append(cur_pred)

            ensemble_size_preds = np.stack(ensemble_size_pred, axis=-1)  # (N, pred_len, C)
            ensemble_preds.append(ensemble_size_preds)

        print(']')
        ensemble_preds_lst = np.stack(ensemble_preds, axis=-1)
        preds = np.mean(ensemble_preds_lst, axis=-1)

        return preds


class All_Select_Model(Baseline_Select_Model):
    """全部集成模型"""

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None):
            model_order = list(range(self.args.current_zoo_num - 1, -1, -1)) #默认为Recent模式
            ensemble_size = self.args.current_zoo_num  # 全部模型都参与集成
            return model_order, ensemble_size

        return select_strategy


class Random_Select_Model(Baseline_Select_Model):
    """随机选择模型，且同时用于所有数据集"""

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None):  # 选择策略在base_model中实现
            ensemble_size = self.args.ensemble_size
            return None, ensemble_size

        return select_strategy


class Recent_Select_Model(Baseline_Select_Model):
    """选择近期模型"""

    def _get_select_strategy(self, dataset):
        def select_strategy(dataset_name=None):
            model_order = list(range(self.args.current_zoo_num - 1, -1, -1))  # 自然数的倒序列表,越新的越靠前
            ensemble_size = self.args.ensemble_size
            return model_order, ensemble_size

        return select_strategy


class Real_Select_Model(Baseline_Select_Model):
    """真实顺序，作为order的真实标记"""

    def __init__(self, args, model_name, Model_zoo_current):
        super().__init__(args, model_name, Model_zoo_current)
        self._dataset_fixed_orders = None
        self.real_order_metric = getattr(args, "real_order_metric", "sMAPE")

    # ---------- 读取 baseline 结果，构建每个 ds_config 的真实模型顺序 ----------
    def _build_dataset_fixed_orders(self):
        """读取 baseline all_results.csv，按 real_metric 为每个数据集计算模型顺序"""
        if self._dataset_fixed_orders is not None:
            return self._dataset_fixed_orders

        results_dir = Path("results")

        all_frames = []

        # 只遍历当前 zoo 中的模型（Model_zoo_current）
        for family, sizes_dict in self.Model_sizes.items():
            for size_name, model_info in sizes_dict.items():
                model_folder = f"{family}_{size_name}"
                model_cl_name = self.model_cl_name
                csv_path = results_dir / model_folder / model_cl_name / "all_results.csv"
                print(f"[Real_Select] 读取 baseline 结果: {csv_path}")
                if not csv_path.exists():
                    print(f"[Real_Select] baseline 文件缺失: {csv_path}")
                    warnings.warn(f"[Real_Select] baseline 文件缺失: {csv_path}")
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"[Real_Select] 无法读取 {csv_path}: {e}")
                    warnings.warn(f"[Real_Select] 无法读取 {csv_path}: {e}")
                    continue

                if self.real_order_metric not in df.columns:
                    print(f"[Real_Select] {csv_path} 中没有列 '{self.real_order_metric}'，跳过该模型")
                    warnings.warn(f"[Real_Select] {csv_path} 中没有列 '{self.real_order_metric}'，跳过该模型")
                    continue

                model_abbr = model_info["abbreviation"]

                tmp = df[["dataset", self.real_order_metric]].copy()
                tmp["model_abbr"] = model_abbr
                all_frames.append(tmp)

        if not all_frames:
            raise RuntimeError("[Real_Select] 没有成功加载任何 baseline all_results.csv，无法构建真实顺序。")

        baseline_df = pd.concat(all_frames, ignore_index=True)

        dataset_fixed_orders = {}

        for ds_config, ds_group in baseline_df.groupby("dataset"):
            # 每个 ds_config 上，按模型聚合该 metric 的均值
            metric_by_model = (
                ds_group.groupby("model_abbr")[self.real_order_metric]
                .mean()
                .dropna()
            )
            if metric_by_model.empty:
                continue

            # metric 越小越好，从小到大排序
            ordered_abbrs = list(metric_by_model.sort_values(ascending=True).index)

            # 用模型 id 表示（id 已按 release 时间编码）
            model_ids = []
            for abbr in ordered_abbrs:
                if abbr not in self.abbr_to_id:
                    warnings.warn(f"[Real_Select] 模型缩写 {abbr} 不在当前 zoo 的 abbr_to_id 中，跳过")
                    continue
                model_ids.append(int(self.abbr_to_id[abbr]))

            if model_ids:
                dataset_fixed_orders[ds_config] = model_ids

        if not dataset_fixed_orders:
            raise RuntimeError("[Real_Select] 所有数据集都没有有效真实顺序，请检查 baseline 结果。")

        self._dataset_fixed_orders = dataset_fixed_orders
        return dataset_fixed_orders

    # ----------------- 选择策略 -----------------
    def _get_select_strategy(self, dataset):
        """
        当 real_order_metric == 'sMAPE' 时，使用按 sMAPE 计算出来的真实排序。
        """
        dataset_fixed_orders = self._build_dataset_fixed_orders()

        def select_strategy(dataset_name=None):
            if dataset_name is None:
                raise ValueError("[Real_Select] 需要 dataset_name 来恢复 ds_config。")

            try:
                ds_key, ds_freq, term = dataset_name.rsplit("_", 2)
            except ValueError:
                raise ValueError(f"[Real_Select] 非预期的 dataset_name 格式: {dataset_name}")

            ds_config = f"{ds_key}/{ds_freq}/{term}"

            if ds_config not in dataset_fixed_orders:
                raise ValueError(
                    f"[Real_Select] ds_config={ds_config} 在 baseline 结果中没有真实顺序，"
                    f"请确认 {ds_config} 的 all_results.csv 已生成。"
                )

            fixed_model_order = dataset_fixed_orders[ds_config]

            print(
                f"[Real_Select] sort_metric={self.args.real_order_metric}) for dataset_name={dataset_name}: "
                f"{fixed_model_order}"
            )

            ensemble_size = self.args.ensemble_size
            return fixed_model_order, ensemble_size

        return select_strategy
