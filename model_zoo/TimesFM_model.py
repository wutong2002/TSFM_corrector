import os
import json
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from model_zoo.TSFM_src.timesfm.configs import ForecastConfig
from model_zoo.TSFM_src.timesfm.timesfm_2p5 import timesfm_2p5_torch
from model_zoo.base_model import BaseModel
from utils.missing import fill_missing
# ====================== GluonTS 日志过滤 ======================

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)

# ====================== 模型封装层 ======================

class TimesFMModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)
        self.model_size = model_name.split("_")[-1]
        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):

        # ===== 1) 根据模型版本构建 TimesFm 实例 =====
        if self.model_size == '2.5':
            # 在线调用
            # tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
            # tfm.load_checkpoint()

            # 本地加载
            tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_local_path,
                local_files_only=True,
            )
        else:
            raise ValueError(f"Unsupported model size: {self.model_size}")

        return TimesFmPredictor(
            self.args,
            tfm,
            batch_size,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            model_size=self.model_size
        )


class TimesFmPredictor:

    def __init__(
            self,
            args,
            tfm,
            batch_size,
            prediction_length: int,
            ds_freq: str,
            model_size: str,
    ):

        self.args = args
        self.tfm = tfm
        self.batch_size = batch_size
        self.tfm.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.model_size=model_size
        self.prediction_length = prediction_length

        if self.model_size == '2.5':
            self.quantiles = list(np.arange(1, 10) / 10.0)
            context_info = (
                self.args.context_len if self.args.fix_context_len else "original"
            )
            print(
                f"[TimesFM-2.5] context_len={context_info}, "
                f"freq_used=Fasle, "
                f"impute_missing=True "
            )
        else:
            raise ValueError(f"Unsupported model size: {self.model_size}")

    def predict(self, test_data_input: List[dict], batch_size: int = 1024) -> List[QuantileForecast]:

        forecasts: List[QuantileForecast] = []
        idx_offset = 0
        for batch in tqdm(
                batcher(test_data_input, batch_size=self.batch_size),
                total=len(test_data_input) // self.batch_size,
                desc="TimesFM Predict"):

            context = []
            # --------------------------------------------------
            # ① 逐样本：NaN 清洗 + context 截取（nan & context 处理）
            # --------------------------------------------------
            for entry in batch:
                raw = np.array(entry["target"], dtype=float)

                raw = fill_missing(
                    raw,
                    all_nan_strategy_1d="linspace",
                    interp_kind_1d="nearest",
                    add_noise_1d=True,
                    noise_ratio_1d=0.01,
                )

                # --- context 截取 ---
                if self.args.fix_context_len:
                    arr = raw[-self.args.context_len:]
                else:
                    arr = raw
                context.append(arr)

            # --------------------------------------------------
            # ② TimesFM 前向预测：freq + context -> quantile 预测（pred_len 由 horizon_len 控制）
            # --------------------------------------------------
            # 1）根据 batch 中最长序列长度，按 patch 大小 p 对齐
            max_context = 0
            for arr in context:
                if max_context < arr.shape[0]:
                    max_context = arr.shape[0]

            # 补齐为 p 的倍数
            p = getattr(getattr(self.tfm, "model", None), "p", None)
            if p is not None:
                max_context = ((max_context + p - 1) // p) * p

            # 如果指定了 fix_context_len，则再用 context_len 做一个上限裁剪，
            if self.args.fix_context_len:
                max_context = min(self.args.context_len, max_context)

            # 避免过长
            max_context = min(15360, max_context)

            # 2）compile：在这里动态编译预测配置
            self.tfm.compile(
                forecast_config=ForecastConfig(
                    max_context=max_context,
                    max_horizon=self.prediction_length,
                    infer_is_positive=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                    return_backcast=False,
                    normalize_inputs=True,
                    # per_core_batch_size 与外部 batch_size 对齐
                    per_core_batch_size=self.batch_size,
                ),
            )

            # 3）forecast：显式传入 horizon = prediction_length
            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length,
                inputs=context,
            )

            full_preds = full_preds[:, 0: self.prediction_length, 1:]

            batch_forecast = full_preds.transpose((0, 2, 1))  # [B, Q, H]

            # --------------------------------------------------
            # ③ 逐条构造 QuantileForecast，并检查 NaN
            # --------------------------------------------------
            for i_in_batch, (arr, ts) in enumerate(zip(batch_forecast, batch)):
                if np.isnan(arr).any():
                    global_idx = idx_offset + i_in_batch
                    raise ValueError(f"[NaN DEBUG] TimesFM 第 {global_idx}-th 序列的预测含 NaN")

                start = ts["start"] + len(ts["target"])

                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=arr,
                        forecast_keys=list(map(str, self.quantiles)),
                        start_date=start,
                    )
                )

            idx_offset += len(batch)

        return forecasts
