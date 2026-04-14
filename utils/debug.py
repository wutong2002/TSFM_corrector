import copy
import numpy as np
import itertools

def debug_check_input_nan(test_data_input):
    for i, entry in enumerate(test_data_input):
        tgt = np.asarray(entry["target"], dtype=float)
        if not np.all(np.isfinite(tgt)):
            n_nan = np.isnan(tgt).sum()
            n_posinf = np.isposinf(tgt).sum()
            n_neginf = np.isneginf(tgt).sum()
            print(
                f"[DATA NaN DEBUG] series {i} has non-finite values: "
                f"NaN={n_nan}, +Inf={n_posinf}, -Inf={n_neginf}, shape={tgt.shape}"
            )
            # 只看前几个就够了，不要全打印
            bad_idx = np.where(~np.isfinite(tgt))[0][:10]
            print("  bad indices (first 10):", bad_idx)
            break  # 先锁定一条


def debug_print_test_input(dataset, max_items: int = 1):
    """Inspect dataset.test_data.input for debugging."""
    test_input = dataset.test_data.input

    print("[DEBUG] type(dataset.test_data.input):", type(test_input))
    try:
        print("[DEBUG] len(dataset.test_data.input):", len(test_input))
    except TypeError:
        print("[DEBUG] len(...) not supported for this type")

    # 取出前 max_items 个样本
    if isinstance(test_input, (list, tuple)):
        samples = list(test_input[:max_items])
    else:
        samples = list(itertools.islice(test_input, max_items))

    for idx, item in enumerate(samples):
        print(f"\n[DEBUG] Sample #{idx} type:", type(item))
        if isinstance(item, dict):
            for k, v in item.items():
                if hasattr(v, "shape"):
                    print(f"  key={k}, type={type(v)}, shape={v.shape}, "
                          f"dtype={getattr(v, 'dtype', None)}")
                else:
                    # 标量/简单类型打印值，其余打印类型
                    if isinstance(v, (int, float, str)):
                        print(f"  key={k}, type={type(v)}, value={v}")
                    else:
                        print(f"  key={k}, type={type(v)}, value=...")
        else:
            if hasattr(item, "shape"):
                print(f"  shape={item.shape}, dtype={getattr(item, 'dtype', None)}")
            else:
                print("  value:", item)


def debug_forecasts(forecasts):
    if len(forecasts) == 0:
        print("\n[ERROR] forecasts 是空列表，无法评估！")
        raise ValueError("预测列表 forecasts 为空，无法进行后续处理。")

    # 检查 forecasts[0] 的结构（SampleForecast / QuantileForecast）
    fc0 = forecasts[0]
    if hasattr(fc0, "samples"):
        print(f"SampleForecasts shape: {len(forecasts)} * ", fc0.samples.shape)
    elif hasattr(fc0, "forecast_array"):
        print(
            f"QuantileForecasts shape: {len(forecasts)} * ",
            fc0.forecast_array.shape,
        )
    else:
        print(f"\n[DEBUG] 预测数量: {len(forecasts)}")
        print(f"[DEBUG] forecasts[0] 类型: {type(fc0)}")
        if hasattr(fc0, "__dict__"):
            print(f"[DEBUG] forecasts[0] 属性列表:")
            for attr_name, attr_value in fc0.__dict__.items():
                print(
                    f"  - {attr_name}: 类型 {type(attr_value)}, "
                    f"内容预览: {str(attr_value)[:200]}"
                )
        raise ValueError(
            "forecasts[0] does not have 'samples' or 'forecast_array' attribute"
        )