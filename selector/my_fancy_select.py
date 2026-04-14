import numpy as np
import torch
import importlib
import copy  
from tqdm import tqdm
from selector.baseline_select import Baseline_Select_Model
from Model_Path.model_zoo_config import Model_zoo_details

class MyFancySelectModel(Baseline_Select_Model):
    """
    Instance-wise Validation Selector

    """

    def _get_select_strategy(self, dataset):
        test_data_input = list(dataset.test_data.input)
        validate_sample_size = min(5, len(test_data_input))
        val_samples = test_data_input[:validate_sample_size]
        
        pred_len = dataset.prediction_length
        val_inputs = []
        val_targets = []

        for item in val_samples:
            target = np.array(item["target"], dtype=float)
            if len(target) > pred_len:
                # 切分：用前面的做输入，最后的 pred_len 做真值验证
                val_inputs.append({
                    "target": target[:-pred_len],
                    "start": item["start"],
                    "feat_static_cat": item.get("feat_static_cat"),
                    "item_id": item.get("item_id")
                })
                val_targets.append(target[-pred_len:])

        candidates = [
            ("chronos", "bolt_tiny"), 
            ("moirai", "small"),
            ("timesfm", "2.5"),
            ("kairos", "10m"),
            ("tirex", "base"),
        ]

        print(f"\n[FancySelect] 正在为数据集 {dataset.name} 挑选最佳模型...")
        
        best_score = float('inf')
        winner_abbr = None

        if len(val_inputs) == 0:
            print("⚠️ 数据太短，无法验证，默认选择第一个模型。")
            fam, size = candidates[0]
            if fam in Model_zoo_details and size in Model_zoo_details[fam]:
                winner_abbr = Model_zoo_details[fam][size]["abbreviation"]
        else:
            class DummyData:
                def __init__(self):
                    self.prediction_length = pred_len
                    self.target_dim = 1
                    self.freq = "1h"
                    self.past_feat_dynamic_real_dim = 0
                    self.feat_dynamic_real_dim = 0
                    self.feat_static_cat_dim = 0
            
            dummy_dataset = DummyData()

            for fam, size in candidates:
                try:
                    if fam not in Model_zoo_details: continue
                    cfg = Model_zoo_details[fam][size]
                    
                    # 动态导入模型类
                    ModelModule = importlib.import_module(cfg["model_module"])
                    ModelClass = getattr(ModelModule, cfg["model_class"])

                    child_args = copy.copy(self.args)
                    child_args.run_mode = "zoo" 
                    tmp_model = ModelClass(
                        child_args,  
                        module_name=cfg["module_name"],
                        model_name=f"{fam}_{size}", 
                        model_local_path=cfg["model_local_path"]
                    )
                    
                    # 获取预测器
                    predictor = tmp_model.get_predictor(dummy_dataset, batch_size=len(val_inputs))
                    predictor.prediction_length = pred_len 
                    
                    # 预测
                    forecasts = list(predictor.predict(val_inputs))
                    
                    # 计算 MAE (兼容不同输出格式)
                    total_mae = 0
                    for fc, true_val in zip(forecasts, val_targets):
                        if hasattr(fc, "samples"):
                            pred_mean = fc.samples.mean(axis=0)
                        elif hasattr(fc, "mean"):
                            pred_mean = fc.mean
                        elif hasattr(fc, "quantile"):
                            # Chronos 可能输出分位数
                            pred_mean = fc.quantile(0.5)
                        else:
                            # 兜底
                            pred_mean = np.array(fc) 
                            
                        total_mae += np.mean(np.abs(pred_mean - true_val))
                    
                    avg_score = total_mae / len(forecasts)
                    print(f"   🏃 {fam}-{size} \t MAE: {avg_score:.4f}")
                    
                    if avg_score < best_score:
                        best_score = avg_score
                        winner_abbr = cfg["abbreviation"]
                        
                    # 释放显存
                    del tmp_model, predictor
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    # 捕获错误但不中断，继续测试下一个模型
                    print(f"   ⚠️ 跳过 {fam}-{size}: {e}")

        # === 4. 确定赢家 ID ===
        winner_id = 0 # 默认 fallback
        if winner_abbr and winner_abbr in self.abbr_to_id:
            winner_id = self.abbr_to_id[winner_abbr]
            print(f"[FancySelect] 🏆 冠军是: {winner_abbr} (ID: {winner_id})\n")
        else:
            # 兜底逻辑
            if candidates:
                fam, size = candidates[0]
                if fam in Model_zoo_details and size in Model_zoo_details[fam]:
                    def_abbr = Model_zoo_details[fam][size]["abbreviation"]
                    if def_abbr in self.abbr_to_id:
                        winner_id = self.abbr_to_id[def_abbr]
            print(f"[FancySelect] ⚠️ 使用默认模型 ID: {winner_id}\n")
        def select_strategy(dataset_name=None):
            return [winner_id], 1

        return select_strategy