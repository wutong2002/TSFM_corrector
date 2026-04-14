import os

def get_model_id(family: str, size: str) -> str:
    """
    统一生成模型唯一标识符 (Model ID)
    例如: chronos + tiny -> chronos_tiny
    """
    return f"{family}_{size}"

def get_experiment_dir(output_root: str, model_id: str, context_len: int, fix_context: bool) -> str:
    """
    获取基础实验的主输出目录
    结构: {root}/{model_id}/{experiment_tag}
    例如: results/chronos_tiny/cl_original
    """
    # 逻辑与 base_model.py 保持一致
    exp_tag = f"cl_{context_len}" if fix_context else "cl_original"
    return os.path.join(output_root, model_id, exp_tag)

def get_correction_data_dir(output_root: str, model_id: str, context_len: int, fix_context: bool = False) -> str:
    """
    [核心] 获取残差校正数据的存储目录 (学件库路径)
    结构: {root}/{model_id}/{experiment_tag}/correction_data/{model_id}
    """
    base_exp_dir = get_experiment_dir(output_root, model_id, context_len, fix_context)
    # 依然保留您之前生成数据的结构，以免旧数据失效
    # 如果想更简洁，可以去掉最后这层 model_id，但为了兼容现状我们保留它
    return os.path.join(base_exp_dir, "correction_data")

def get_corrector_checkpoint_dir(output_root: str, model_id: str, corrector_arch: str) -> str:
    """
    获取校正器模型权重的保存目录
    结构: results/corrector_checkpoints/{model_id}/{arch}
    """
    # 默认存在 results/corrector_checkpoints 下
    return os.path.join(output_root, "corrector_checkpoints", model_id, corrector_arch)

def get_inference_output_dir(output_root: str, model_id: str, tag: str = "default") -> str:
    """
    获取联合推理结果的保存目录
    """
    return os.path.join(output_root, "correction_inference", model_id, tag)