import os

from model_zoo.base_model import BaseModel
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ====================== 模型封装层 ======================

class MoiraiModel(BaseModel):

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        # ===== 1) 处理 context
        if self.args.fix_context_len:
            context_length = self.args.context_len
        else:
            context_length = 4000 # Moirai 的默认设定

        print(
            f"[Moirai] context_len={context_length}, "
            f"freq_used=Fasle, "
            f"impute_missing=False"
        )

        # ===== 2) 初始化 MoiraiForecast：设定基础 context / pred_len 等 =====
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(self.model_local_path),
            prediction_length=1,           # 后面会根据 dataset 覆盖
            context_length=context_length,
            patch_size=32,
            num_samples=100,
            target_dim=1,                  # 后续再根据 dataset.target_dim 覆盖
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        # ===== 3) 设置 pred_len / 维度：根据 dataset 动态覆盖 hparams =====
        model.hparams.prediction_length = dataset.prediction_length
        model.hparams.target_dim = dataset.target_dim
        model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim

        # ===== 4) 无需处理 freq / NaN，直接交给 Moirai 的 GluonTS predictor
        predictor = model.create_predictor(batch_size=batch_size)
        return predictor
