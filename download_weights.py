import os
from huggingface_hub import snapshot_download

def download_timesfm():
    # 1. 定义目标路径
    repo_id = "google/timesfm-1.0-200m"
    local_dir = os.path.join("checkpoints", "timesfm-1.0-200m")
    
    print(f"🚀 正在下载 {repo_id} 到 {local_dir} ...")
    
    # 2. 执行下载 (使用 snapshot_download 确保所有相关文件都被下载)
    # allow_patterns 确保我们只下载 pytorch 权重和配置文件，忽略不需要的 flax 权重
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 确保下载的是真实文件而不是链接
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.ckpt"],
        ignore_patterns=["*.msgpack", "*.h5"] # 忽略 flax 权重节省空间
    )
    
    print(f"✅ 下载完成！权重已保存在: {os.path.abspath(local_dir)}")
    print("📝 请检查目录下是否有 'config.json' 和 'model.safetensors' (或 pytorch_model.bin)")

if __name__ == "__main__":
    download_timesfm()