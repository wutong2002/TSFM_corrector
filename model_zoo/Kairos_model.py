# -*- coding: utf-8 -*-
"""
Kairos Model Implementation (Local Source Code Version)
"""
import os
import sys
import torch
import numpy as np
from tqdm.auto import tqdm
from gluonts.model.forecast import SampleForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing
try:
    # 尝试导入项目根目录下的 tsfm 包
    from tsfm.model.kairos.modeling_kairos import KairosModel as NativeKairosModel
    from tsfm.model.kairos.configuration_kairos import KairosConfig
except ImportError as e:
    print("\n❌ 无法导入本地 Kairos 源代码！")
    print("请确认 'tsfm' 文件夹位于项目根目录。")
    raise e

class KairosModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        # 1. 显式设置 output_dir
        self.output_dir = os.path.join(args.output_dir, model_name)
        
        # 2. 调用父类初始化
        super().__init__(model_name, args, self.output_dir)
        
        self.model_local_path = model_local_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_predictor(self, dataset, batch_size):
        return KairosPredictor(
            self.model_local_path, 
            dataset.prediction_length, 
            self.device, 
            batch_size,
            self.args,
            # generate = True
        )

class KairosPredictor:
    def __init__(self, model_path, pred_len, device, batch_size, args):
        self.device = device
        self.prediction_length = pred_len
        self.batch_size = batch_size
        self.args = args
        
        print(f"[Kairos] Loading weights from: {model_path}")
        
        try:
            config = KairosConfig.from_pretrained(model_path)
            # 使用本地源码类加载
            self.model = NativeKairosModel.from_pretrained(
                model_path, 
                config=config,
                ignore_mismatched_sizes=True 
            ).to(self.device)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e

        self.model.eval()

    def predict(self, test_data_input):
        forecasts = []
        max_context = self.args.context_len if self.args.fix_context_len else 2048

        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size), 
                          total=len(test_data_input)//self.batch_size,
                          desc="Kairos Predict"):
            
            batch_seqs = []
            for item in batch:
                seq = np.array(item["target"], dtype=float)
                seq = fill_missing(seq, all_nan_strategy_1d="zero", interp_kind_1d="linear")
                if len(seq) > max_context:
                    seq = seq[-max_context:]
                batch_seqs.append(seq)

            max_len = max(len(s) for s in batch_seqs)
            padded_batch = []
            for s in batch_seqs:
                pad_len = max_len - len(s)
                if pad_len > 0:
                    padded_s = np.pad(s, (pad_len, 0), mode='constant', constant_values=0)
                else:
                    padded_s = s
                padded_batch.append(padded_s)
            
            seqs_tensor = torch.tensor(np.array(padded_batch), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                output = self.model(
                    past_target=seqs_tensor, 
                    prediction_length=self.prediction_length, 
                    generation=True,         
                    infer_is_positive=True,   
                    force_flip_invariance=True 
                )
                
                if isinstance(output, dict) and 'prediction_outputs' in output:
                    preds = output['prediction_outputs'].cpu().numpy()
                else:
                    preds = output.cpu().numpy()

            # 封装结果
            for i, p in enumerate(preds):
                if p.shape[-1] > self.prediction_length:
                    p = p[..., :self.prediction_length]
                
                samples = p  
                
                if samples.ndim == 1:
                    samples = samples.reshape(1, -1)

                start_date = batch[i]["start"] + len(batch[i]["target"])
                
                forecasts.append(SampleForecast(
                    samples=samples, 
                    start_date=start_date, 
                    item_id="kairos"
                ))
        
        return forecasts