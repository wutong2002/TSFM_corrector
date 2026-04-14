import os
import sys
import time
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast

from model_zoo.base_model import BaseModel


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"          # 屏蔽 TF 部分 INFO 日志
os.environ["TABPFN_FORCE_DEVICE"] = "cuda"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

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

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.simplefilter("ignore", RuntimeWarning)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
np.seterr(all="ignore")


# ====================== 模型封装层 ======================

class SundialModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)
        self.model_local_path = model_local_path
        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        # ===== 1) context 处理：固定 / 默认上下文长度 =====
        if self.args.fix_context_len:
            context_length = self.args.context_len
        else:
            context_length = 2880  # 默认设定

        print(
            f"[Sundial] context_len={context_length}, "
            f"freq_used=Fasle, "
            f"impute_missing=True"
        )

        # ===== 2) 构建 Predictor（pred_len 由 prediction_length 控制） =====
        predictor = SundialPredictor(
            num_samples=100,
            prediction_length=dataset.prediction_length,
            model_local_path=self.model_local_path,
            device_map="cuda:0",
            batch_size=batch_size,
            context_length=context_length,
        )

        return predictor




class SundialPredictor:
    """
        - predict:
        1) 依据 context_length 截断每条序列（context 处理）
        2) 对 NaN 做 LastValueImputation 填补（缺失值处理）
        3) 调用 self.model.generate 生成未来序列（pred_len 处理）
        4) 封装为 GluonTS SampleForecast
    """
    def __init__(
            self,
            num_samples: int,
            prediction_length: int,
            device_map,
            model_local_path,
            batch_size,
            context_length: int = 2880,
    ):
        self.device = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_local_path, trust_remote_code=True)
        self.model.to(self.device)
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.context_length = context_length

    def left_pad_and_stack_1D(self, tensors):
        """
        将若干 1D tensor 左侧补 NaN 对齐后堆叠成 [B, T]：
        - 先找出该 batch 最长长度
        - 其他样本左侧 padding NaN，使得末尾对齐
        """
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(
                size=(max_len - len(c),), fill_value=torch.nan, device=c.device
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def prepare_and_validate_context(self, context):
        """
        将 list[Tensor] 或 Tensor 统一处理为 2D Tensor [B, T]：
        - list -> left_pad_and_stack_1D
        - 保证最终是二维 [batch, time_steps]
        """
        if isinstance(context, list):
            context = self.left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(self,test_data_input,):

        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size), total=len(test_data_input) // self.batch_size,
                          desc="Sundial Predict"):
            context = [torch.tensor(entry["target"]) for entry in batch]
            batch_x = self.prepare_and_validate_context(context).to(self.device)

            # ===== 1) context 处理：依据 context_length 截断每条序列 =====
            if batch_x.shape[-1] > self.context_length:
                batch_x = batch_x[..., -self.context_length:]

            # ===== 2) 缺失值 / NaN 处理：LastValueImputation =====
            if torch.isnan(batch_x).any():
                batch_x = batch_x.cpu().numpy()

                imputed_rows = []
                for i in range(batch_x.shape[0]):
                    row = batch_x[i]
                    imputed_row = LastValueImputation()(row)
                    imputed_rows.append(imputed_row)
                batch_x = torch.tensor( np.vstack(imputed_rows),device=self.device,)

            # ===== 3) 进行生成预测（pred_len 处理） =====
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model.generate(batch_x, max_new_tokens=self.prediction_length, revin=True, num_samples=self.num_samples)
            forecast_outputs.append(outputs.detach().cpu().numpy())
        forecast_outputs = np.concatenate(forecast_outputs)

        # ===== 4) 封装为 GluonTS SampleForecast 列表 =====
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts
