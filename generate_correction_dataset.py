import os
import sys
import pickle
import numpy as np
import importlib
import torch
import glob
from types import SimpleNamespace
from tqdm import tqdm
from pathlib import Path
import json
from utils.missing import fill_missing

# =========================================================================
# 1. 环境配置与导入
# =========================================================================
sys.path.append(os.getcwd())

try:
    from Model_Path.model_zoo_config import Model_zoo_details
except ImportError:
    print("❌ 错误: 无法导入 Model_Path.model_zoo_config")
    exit(1)

try:
    from utils.data import Dataset, Term
except ImportError:
    print("❌ 错误: 无法导入 utils.data.Dataset")
    exit(1)

# =========================================================================
# 2. 全局配置 (CONFIG)
# =========================================================================
CONFIG = {
    # --- 任务控制 --- "moirai", "kairos", "timesfm",
    "TARGET_FAMILIES": [ "tirex", "chronos"], 
    "TARGET_VARIANTS": [], 

    # 🌟 [新增功能] 数据集前缀白名单 (例如 ["QB_"] 只跑QuitoBench)。留空 [] 则不限制，跑所有扫描到的数据。
    "TARGET_DATASET_PREFIXES": ["QB_"], 

    # 🌟 自由长度控制机制 🌟
    "USE_NEW_MECHANISM": False,                    
    "CUSTOM_HISTORY_LEN": 96,                     
    "NEW_OUTPUT_ROOT": "correction_datasets_double_res_0", 

    # --- 传统机制配置 (USE_NEW_MECHANISM = False 时生效) ---
    "RAW_DATA_ROOT": os.path.join("Datasets", "raw_data"),
    "OUTPUT_ROOT": "correction_datasets_double_res_0", 
    "CONTEXT_LEN": 512,           
    
    # --- 样本生成超参 ---
    "STRIDE": 32,                 
    "MAX_SAMPLES_PER_DS": 2000,   
    "MAX_WINDOWS_PER_SEQ": 50,    
    "BATCH_SIZE": 128,
    "GPU_ID": "0",
    "PROPERTIES_PATH": os.path.join("Datasets", "processed_datasets", "dataset_properties.json"),
}

# =========================================================================
# 3. 工具函数
# =========================================================================
def get_metadata_for_dataset(properties, dataset_name, gluon_freq):
    key = dataset_name
    info = properties.get(key) or properties.get(key.lower())
    domain = info.get("domain", "Generic") if info else "Generic"
    freq = info.get("frequency", gluon_freq) if info else gluon_freq
    return {"domain": domain, "freq": freq}

def load_dataset_properties(path):
    if not os.path.exists(path): return {}
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

def dynamic_load_model_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def build_complete_args():
    args = SimpleNamespace()
    args.output_dir = os.path.join("results", "inference_gen_temp") 
    args.output_root = CONFIG["OUTPUT_ROOT"] 
    args.run_mode = "zoo"                    
    args.save_pred = False
    args.save_for_correction = True
    args.skip_saved = False
    args.debug_mode = False
    args.context_len = CONFIG["CONTEXT_LEN"]
    args.fix_context_len = False
    args.batch_size = CONFIG["BATCH_SIZE"]
    args.num_workers = 0
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args

def get_all_raw_datasets(root_path):
    dataset_paths = []
    root = Path(root_path)
    if not root.exists(): return []
    for dirpath, _, filenames in os.walk(root):
        if ("dataset_info.json" in filenames) or ("state.json" in filenames) or any(f.endswith(".arrow") for f in filenames):
            dataset_paths.append(str(Path(dirpath)))
    return sorted(list(set(dataset_paths)))

def check_is_processed(model_name, dataset_name):
    ds_lower = dataset_name.lower()
    target_ge = f"ge_{ds_lower}.pkl"    
    target_lotsa = f"lotsa_{ds_lower}"  
    
    search_pattern = os.path.join(
        CONFIG["OUTPUT_ROOT"], model_name, dataset_name, "*", "*", "correction_data", "*" 
    )
    for file_path in glob.glob(search_pattern):
        filename = os.path.basename(file_path).lower()
        # [修改] 增加 `ds_lower in filename` 的兜底，确保对 QB_ 等新增前缀也能正确识别是否已处理
        if (target_ge in filename) or (target_lotsa in filename) or (ds_lower in filename): 
            return True
    return False

def generate_sliding_windows(dataset_obj, target_ctx_len, target_pred_len, dataset_name, model_name):
    input_records = [] 
    test_dataset_pairs = []
    sample_metas = [] 
    
    default_stride = CONFIG["STRIDE"]
    
    for seq_idx, entry in enumerate(dataset_obj.gluonts_dataset):
        full_seq = entry['target'] 
        seq_len = len(full_seq)
        
        min_req_len = target_ctx_len + target_pred_len
        if seq_len < min_req_len: 
            continue
            
        max_t = seq_len - target_pred_len
        window_space = max_t - target_ctx_len
        
        default_num_windows = (window_space // default_stride) + 1
        if default_num_windows < 5 and window_space > 0:
            calc_stride = max(1, window_space // 4)
            current_stride = min(default_stride, calc_stride)
        else:
            current_stride = default_stride

        start_times = list(range(target_ctx_len, max_t + 1, current_stride))
        if len(start_times) > CONFIG["MAX_WINDOWS_PER_SEQ"]:
             start_times = start_times[-CONFIG["MAX_WINDOWS_PER_SEQ"]:]
             
        for t in start_times:
            truth_window = full_seq[t : t + target_pred_len]
            history_window = full_seq[t - target_ctx_len : t]
            
            input_entry = entry.copy()
            input_entry['target'] = history_window
            label_entry = entry.copy()
            label_entry['target'] = truth_window
            
            raw_ch = entry.get("channel_id", -1)
            try:
                channel_id = int(raw_ch)
            except Exception:
                channel_id = -1

            meta = {
                "tsfm_name": model_name,
                "dataset_subset": dataset_name,
                "seq_id": seq_idx,
                "item_id": str(entry.get("item_id", f"seq_{seq_idx}")),
                "parent_item_id": str(entry.get("parent_item_id", entry.get("item_id", f"seq_{seq_idx}"))),
                "channel_id": channel_id,
                "hist_start": int(t - target_ctx_len),
                "hist_end": int(t)
            }
            
            input_records.append(input_entry)
            test_dataset_pairs.append((input_entry, label_entry))
            sample_metas.append(meta)
            
    if len(input_records) > CONFIG["MAX_SAMPLES_PER_DS"]:
        indices = np.random.choice(len(input_records), CONFIG["MAX_SAMPLES_PER_DS"], replace=False)
        input_records = [input_records[i] for i in indices]
        test_dataset_pairs = [test_dataset_pairs[i] for i in indices]
        sample_metas = [sample_metas[i] for i in indices] 
        
    return input_records, test_dataset_pairs, sample_metas

# =========================================================================
# 4. 核心处理流程
# =========================================================================
def process_dataset_with_model(model_instance, raw_dataset_path, dataset_properties):
    dataset_name = "_".join(raw_dataset_path.split(os.sep)[3:])
    if "LOTSA" in raw_dataset_path or "lotsa" in raw_dataset_path: save_name = "LOTSA_" + dataset_name
    elif "Gift_Eval" in raw_dataset_path or "gift_eval" in raw_dataset_path: save_name = "GE_" + dataset_name
    else: save_name = dataset_name

    print(f"\n➡️ 处理数据集: {save_name}")
    if check_is_processed(model_instance.model_name, save_name): return False

    try:
        ds = Dataset(name=raw_dataset_path, term="short", force_univariate=True)
    except Exception as e:
        print(f"   ⚠️ [Load Error]: {e}")
        return False

    ds_config = f"{save_name}/{ds.freq}/short"
    meta_info = get_metadata_for_dataset(dataset_properties, save_name, ds.freq)

    if CONFIG.get("USE_NEW_MECHANISM", False):
        target_ctx_len = CONFIG["CUSTOM_HISTORY_LEN"]
        target_pred_len = 512
        ds.prediction_length = target_pred_len 
    else:
        target_pred_len = ds.prediction_length
        target_ctx_len = CONFIG["CONTEXT_LEN"]

    model_family = getattr(model_instance.args, 'model_family', 'unknown').lower()
    is_mask_sensitive = any(fam in model_family for fam in ["moirai", "uni2ts"])

    model_id = f"{model_instance.args.model_family}_{model_instance.args.model_size}" if hasattr(model_instance.args, 'model_family') else model_instance.model_name

    raw_inputs, test_dataset_pairs, sample_metas = generate_sliding_windows(
        ds, target_ctx_len, target_pred_len, save_name, model_id
    )
    
    n_samples = len(raw_inputs)
    if n_samples == 0:
        print(f"   ⚠️ [Empty] 序列过短，无法满足要求，已跳过")
        return False

    print(f"   🔨 [Processing] {n_samples} samples | Context: {target_ctx_len} | Pred: {target_pred_len}")

    try:
        predictor = model_instance.get_predictor(ds, CONFIG["BATCH_SIZE"])
        
        processed_main_inputs = []
        for entry in raw_inputs:
            full_hist = np.array(entry['target'], dtype=np.float32)
            if is_mask_sensitive:
                main_ctx = full_hist
            else:
                is_data = ~np.isnan(full_hist) & (np.abs(full_hist) > 1e-6)
                valid_indices = np.where(is_data)[0]
                if len(valid_indices) > 0:
                    main_ctx = fill_missing(full_hist[valid_indices[0]:])
                else:
                    main_ctx = full_hist 
            new_entry = entry.copy()
            new_entry['target'] = main_ctx
            processed_main_inputs.append(new_entry)

        print("      1/2 Inferring Main Task (Future)...")
        main_forecasts = list(tqdm(predictor.predict(processed_main_inputs), total=n_samples, leave=False))
        
        main_preds = []
        for fcst in main_forecasts:
            if hasattr(fcst, "samples"): val = np.mean(fcst.samples, axis=0)
            elif hasattr(fcst, "mean"): val = fcst.mean
            elif hasattr(fcst, "forecast_array"): val = np.mean(fcst.forecast_array, axis=0)
            else: val = None
            main_preds.append(val)

        print("      2/2 Inferring Self-Validation Task (Local Error)...")
        
        local_inputs = []
        local_truths = []
        
        for entry in raw_inputs: 
            full_hist = np.array(entry['target'], dtype=np.float32)
            valid_mask = ~np.isnan(full_hist)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) <= target_pred_len: 
                local_inputs.append(None); local_truths.append(None); continue
            
            if is_mask_sensitive:
                l_val_len = target_pred_len  
                split_idx = valid_indices[-l_val_len]
                ctx = full_hist.copy()
                ctx[split_idx:] = np.nan 
                truth = full_hist[split_idx:][valid_mask[split_idx:]]
            else:
                is_data = ~np.isnan(full_hist) & (np.abs(full_hist) > 1e-6)
                valid_indices_in_raw = np.where(is_data)[0]
                
                if len(valid_indices_in_raw) <= target_pred_len: 
                    local_inputs.append(None); local_truths.append(None); continue

                start_idx = valid_indices_in_raw[0]
                effective_hist_filled = fill_missing(full_hist[start_idx:])
                
                if len(effective_hist_filled) <= target_pred_len:
                    local_inputs.append(None); local_truths.append(None); continue
                
                l_val_len = target_pred_len
                ctx = effective_hist_filled[:-l_val_len]
                truth = effective_hist_filled[-l_val_len:]

            new_entry = entry.copy()
            new_entry['target'] = ctx
            local_inputs.append(new_entry)
            local_truths.append(truth)
            
        valid_indices = [i for i, x in enumerate(local_inputs) if x is not None]
        valid_local_inputs = [local_inputs[i] for i in valid_indices]
        local_preds_map = {}
        
        if len(valid_local_inputs) > 0:
            local_forecasts = list(tqdm(predictor.predict(valid_local_inputs), total=len(valid_local_inputs), leave=False))
            for idx, fcst in zip(valid_indices, local_forecasts):
                if hasattr(fcst, "samples"): val = np.mean(fcst.samples, axis=0)
                elif hasattr(fcst, "mean"): val = fcst.mean
                elif hasattr(fcst, "forecast_array"): val = np.mean(fcst.forecast_array, axis=0)
                else: val = None
                
                l_truth = local_truths[idx]
                
                if val is not None and l_truth is not None:
                    valid_l = len(l_truth) 
                    loc_res = l_truth - val[:valid_l]
                    local_preds_map[idx] = loc_res
        
        local_preds_aligned = [local_preds_map.get(i, None) for i in range(n_samples)]

        timestamps, histories, truths, clean_preds, residuals, local_res_list = [], [], [], [], [], []
        saved_sample_metas = [] 
        
        f32_max = np.finfo(np.float32).max * 0.9
        
        def is_invalid(arr):
            if arr is None: return True  
            arr_np = np.array(arr, dtype=np.float64) 
            if len(arr_np) == 0: return True 
            return np.isnan(arr_np).any() or np.isinf(arr_np).any() or np.max(np.abs(arr_np)) > f32_max
        
        for idx, (item, forecast, pred_val, meta) in enumerate(zip(test_dataset_pairs, main_forecasts, main_preds, sample_metas)):
            hist = np.array(item[0]["target"], dtype=np.float64)
            gt = np.array(item[1]["target"], dtype=np.float64)
            
            p = pred_val
            if p.size == gt.size: p = p.reshape(gt.shape)
            elif p.ndim > gt.ndim: p = p.mean(axis=0).reshape(gt.shape)
            res = gt - p
            
            loc_res = None
            if len(local_preds_aligned) > idx and local_preds_aligned[idx] is not None:
                loc_res = np.array(local_preds_aligned[idx], dtype=np.float64)
            
            if is_invalid(hist) or is_invalid(gt) or is_invalid(p) or is_invalid(loc_res) or np.std(hist) < 1e-6:
                continue
                
            ts = forecast.start_date
            timestamps.append(ts.to_timestamp() if hasattr(ts, 'to_timestamp') else ts)
            
            histories.append(hist.astype(np.float32))
            truths.append(gt.astype(np.float32))
            clean_preds.append(p.astype(np.float32))
            residuals.append(res.astype(np.float32))
            local_res_list.append(loc_res.astype(np.float32))
            saved_sample_metas.append(meta) 

        if len(histories) == 0:
            print("   ⚠️ 所有数据未通过物理合规性校验，无内容可存。")
            return False

        data_dump = {
            "dataset_name": save_name,
            "source": save_name,
            "config": ds_config,
            "metadata": meta_info,
            "timestamps": timestamps,
            "histories": np.array(histories, dtype=object),
            "truths": np.array(truths, dtype=object),
            "preds": clean_preds,
            "residuals": np.array(residuals, dtype=object),
            "local_residuals": np.array(local_res_list, dtype=object),
            "sample_metadata": saved_sample_metas  
        }
        
        save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], model_id, save_name, str(ds.freq), "short", "correction_data")
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{save_name}_correction_data.pkl"
        
        with open(os.path.join(save_dir, file_name), "wb") as f:
            pickle.dump(data_dump, f)
            
        print(f"   ✅ 成功强力写入 {len(histories)} 条干净数据 -> {os.path.join(save_dir, file_name)}")
        return True

    except Exception as e:
        print(f"   ❌ [Error] {save_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# =========================================================================
# 5. 主入口
# =========================================================================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["GPU_ID"]
    
    if CONFIG.get("USE_NEW_MECHANISM", False):
        CONFIG["OUTPUT_ROOT"] = CONFIG["NEW_OUTPUT_ROOT"]
        print(f"\n🚀 [配置生效] 启用强制比例定长机制！")
        print(f"   - 历史长度强制设定为: {CONFIG['CUSTOM_HISTORY_LEN']}")
        print(f"   - 预测长度强制推导为: 512")
        print(f"   - 📁 数据将绝对隔离保存至: {CONFIG['OUTPUT_ROOT']}")
    else:
        print(f"\n🚀 [配置生效] 采用原生长度机制 (遵守原数据集 Pred_Len, 固定 Context={CONFIG['CONTEXT_LEN']})")
    
    print(f"📋 正在加载数据集属性: {CONFIG['PROPERTIES_PATH']}")
    all_properties = load_dataset_properties(CONFIG['PROPERTIES_PATH'])
    print(f"✅ 加载了 {len(all_properties)} 条属性记录")
    
    # 完全保留原生扫描路径逻辑
    print(f"🔍 正在扫描原始数据目录: {CONFIG['RAW_DATA_ROOT']} ...")
    all_raw_paths = get_all_raw_datasets(CONFIG['RAW_DATA_ROOT'])
    
    target_prefixes = CONFIG.get("TARGET_DATASET_PREFIXES", [])
    if target_prefixes:
        print(f"🎯 [开启过滤] 仅处理前缀包含 {target_prefixes} 的数据集...")
    
    print("="*60)

    for family, variants in Model_zoo_details.items():
        if CONFIG["TARGET_FAMILIES"] and family not in CONFIG["TARGET_FAMILIES"]: continue
        
        for variant, info in variants.items():
            if CONFIG["TARGET_VARIANTS"] and variant not in CONFIG["TARGET_VARIANTS"]: continue
            
            model_full_name = f"{family}_{variant}"
            print(f"\n🚀 [Model] {model_full_name}")
            
            try:
                ModelClass = dynamic_load_model_class(info["model_module"], info["model_class"])
                args = build_complete_args()
                args.model_family = family
                args.model_size = variant
                args.output_root = CONFIG["OUTPUT_ROOT"]
                
                model = ModelClass(
                    args=args, module_name=info["module_name"],
                    model_name=model_full_name, model_local_path=info["model_local_path"]
                )
                
                processed_count = 0
                skipped_count = 0
                
                pbar = tqdm(all_raw_paths, desc="Scanning Datasets")
                for raw_path in pbar:
                    ds_name = os.path.basename(raw_path)
                    
                    # 🌟 [极简前缀拦截] 只要文件夹名称不在此前缀列表中，直接略过
                    if target_prefixes and not any(ds_name.startswith(prefix) for prefix in target_prefixes):
                        continue
                    
                    if check_is_processed(model_full_name, ds_name):
                        skipped_count += 1
                        continue
                    
                    pbar.set_description(f"Processing {ds_name}")
                    success = process_dataset_with_model(model, raw_path, all_properties)
                    if success: processed_count += 1
                
                print(f"   🏁 模型 {model_full_name} 完成: 新增 {processed_count} 个, 跳过 {skipped_count} 个")
                    
            except Exception as e:
                print(f"❌ 模型初始化失败: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n🎉 所有数据集构建任务完成!")
