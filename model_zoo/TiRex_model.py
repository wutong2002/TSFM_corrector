import os
import torch
import numpy as np
from tqdm.auto import tqdm
from gluonts.model.forecast import SampleForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing
try:
    from tirex import load_model
except ImportError:
    print("⚠️ Warning: 'tirex' not found. Please install it via pip install tirex-ts")

class TiRexModel(BaseModel):
    def __init__(self, args, module_name, model_name, model_local_path):
        self.output_dir = os.path.join(args.output_dir, model_name)
        super().__init__(model_name, args, self.output_dir)
        self.model_local_path = model_local_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_predictor(self, dataset, batch_size):
        print("dataset prediction_length:", dataset.prediction_length)
        return TiRexPredictor(
            self.model_local_path, 
            dataset.prediction_length, 
            self.device, 
            batch_size,
            self.args
        )

class TiRexPredictor:
    def __init__(self, model_path, pred_len, device, batch_size, args):
        self.device = device
        self.prediction_length = pred_len
        self.batch_size = batch_size
        self.args = args
        
        if os.path.isdir(model_path):
            print(f"[TiRex] '{model_path}' is a directory. Searching for checkpoint file...")
            # 优先寻找常见的文件名
            found = False
            for potential_name in ["checkpoint.ckpt", "model.pt", "pytorch_model.bin", "model.pth"]:
                p = os.path.join(model_path, potential_name)
                if os.path.exists(p):
                    model_path = p
                    found = True
                    break
            
            # 如果没找到常见名，寻找任意 .ckpt 或 .pt 文件
            if not found:
                for f in os.listdir(model_path):
                    if f.endswith(('.ckpt', '.pt', '.pth', '.bin')):
                        model_path = os.path.join(model_path, f)
                        found = True
                        break
            
            if not found:
                raise FileNotFoundError(f"Could not find a checkpoint file (.ckpt/.pt) in {model_path}")

        print(f"[TiRex] Loading actual weight file: {model_path}")
        
        # TiRex 加载 (确保 device 转为 string)
        self.model = load_model(model_path, device=str(device))
        self.model.eval()

    def predict(self, test_data_input):
        forecasts = []
        max_context = self.args.context_len if self.args.fix_context_len else 2048

        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size), 
                          total=len(test_data_input)//self.batch_size,
                          desc="TiRex Predict"):
            
            batch_seqs = []
            for item in batch:
                seq = np.array(item["target"], dtype=float)
                # 缺失值处理
                seq = fill_missing(seq, all_nan_strategy_1d="zero", interp_kind_1d="linear")
                # 截断
                if len(seq) > max_context:
                    seq = seq[-max_context:]
                batch_seqs.append(seq)

            # Padding
            max_len = max(len(s) for s in batch_seqs)
            padded_batch = []
            for s in batch_seqs:
                pad_len = max_len - len(s)
                if pad_len > 0:
                    padded_s = np.pad(s, (pad_len, 0), mode='constant', constant_values=0)
                else:
                    padded_s = s
                padded_batch.append(padded_s)
            
            # 转 Tensor
            input_tensor = torch.tensor(np.array(padded_batch), dtype=torch.float32).to(self.device)

            # 推理
            with torch.no_grad():
                output = self.model.forecast(
                    context=input_tensor, 
                    prediction_length=self.prediction_length
                )
                # print("output shape:", output[0].shape)
                # 处理返回值
                if isinstance(output, tuple):
                    preds = output[0].cpu().numpy()
                elif isinstance(output, torch.Tensor):
                    preds = output.cpu().numpy()
                else:
                    preds = output

            # 封装结果
            for i, p in enumerate(preds):
                if p.ndim == 2:

                    samples = p.T  
                elif p.ndim == 1:

                    samples = p.reshape(1, -1)
                else:

                    samples = p


                if samples.shape[-1] != self.prediction_length:

                    if samples.shape[0] == self.prediction_length:
                        samples = samples.T
                
                start_date = batch[i]["start"] + len(batch[i]["target"])
                forecasts.append(SampleForecast(
                    samples=samples, 
                    start_date=start_date, 
                    item_id="tirex"
                ))
        
        return forecasts