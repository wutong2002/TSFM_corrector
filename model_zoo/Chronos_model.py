import os
import time
import math
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from chronos import BaseChronosPipeline, ForecastType
from gluonts.model.forecast import QuantileForecast, SampleForecast

from model_zoo.base_model import BaseModel
from model_zoo.TSFM_src.chronos_utils import SeriesDataset, identity_collate



# ====================== GluonTS 日志过滤 ======================

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# ====================== 模型封装层 ======================

class ChronosModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.module_name = module_name
        self.model_name = model_name
        self.args = args
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        predictor = ChronosPredictor(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            num_samples=20,
            prediction_length=dataset.prediction_length, )
        return predictor


class ChronosPredictor:

    def __init__(
            self,
            config,
            batch_size,
            model_path,
            num_samples: int,
            prediction_length: int,
            *args,
            **kwargs,
    ):
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map="auto",
            *args,
            **kwargs,
        )
        self.pipeline.device = self.device

        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.config = config
        self.batch_size = batch_size

        context_info = (
            self.config.context_len
            if getattr(self.config, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[Chronos] context_len={context_info}, "
            f"freq_used=Fasle, "
            f"impute_missing=False"
        )

    def predict(self, test_data_input: List[dict], batch_size: int = 32) -> List:
        pipeline = self.pipeline
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if pipeline.forecast_type == ForecastType.SAMPLES
            else {}
        )

        # ===== 1) DataLoader 构造：在 CPU 端并行预取数据 =====
        loader = DataLoader(
            SeriesDataset(test_data_input),
            batch_size=self.batch_size,
            num_workers=self.config.num_workers or 4,
            pin_memory=True,
            collate_fn=identity_collate,
            prefetch_factor=2,
        )

        forecast_outputs = []

        # ===== 2) 逐 batch 预测 =====
        with torch.no_grad():
            # for batch in tqdm(loader, total=math.ceil(len(test_data_input) / self.batch_size)):
            for batch in tqdm(
                batcher(test_data_input, batch_size=self.batch_size),
                total=len(test_data_input) // self.batch_size,
                desc="Chronos Predict"):

                if self.config.fix_context_len:  # 使用固定长度的 context
                    context = [torch.as_tensor(entry["target"][-self.config.context_len:]).to(self.device, non_blocking=True) for entry in batch]
                else:
                    context = [torch.as_tensor(entry["target"]).to(self.device, non_blocking=True) for entry in batch]  # 在 GPU 上做非阻塞拷贝，避免于 CPU 端等待

                preds_tensor = pipeline.predict(
                    context,
                    prediction_length=self.prediction_length,
                    **predict_kwargs, )

                if isinstance(preds_tensor, torch.Tensor):
                    preds_np = preds_tensor.cpu().numpy()
                else:
                    preds_np = preds_tensor
                forecast_outputs.append(preds_np)

        forecast_outputs = np.concatenate(forecast_outputs)

        # ===== 3) 封装为 GluonTS Forecast 对象 =====
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])

            if pipeline.forecast_type == ForecastType.SAMPLES:
                forecasts.append(
                    SampleForecast(samples=item, start_date=forecast_start_date)
                )
            elif pipeline.forecast_type == ForecastType.QUANTILES:
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, pipeline.quantiles)),
                        # forecast_keys=['0.5'],
                        start_date=forecast_start_date,
                    )
                )

        return forecasts
