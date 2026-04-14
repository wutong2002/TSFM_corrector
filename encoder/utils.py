import os
import glob

def find_best_units_checkpoint(search_dir="checkpoints", pattern="units_x*_pretrain_checkpoint.pth"):
    """
    自动搜索 UniTS 权重文件
    策略: 优先找 x128, 然后 x64, 优先 pretrain
    """
    # 支持相对路径和绝对路径
    if not os.path.isabs(search_dir):
        search_dir = os.path.abspath(search_dir)
        
    print(f"🔍 正在目录中搜索 UniTS 权重: {search_dir}")
    
    candidates = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.startswith("units_") and file.endswith(".pth") and "pretrain" in file:
                candidates.append(os.path.join(root, file))
    
    if not candidates:
        return None
    
    # 排序策略：优先选择 x128，其次 x64
    def sort_key(path):
        score = 0
        if "x128" in path: score += 10
        elif "x64" in path: score += 5
        if "pretrain" in path: score += 2
        return -score 
        
    candidates.sort(key=sort_key)
    best_ckpt = candidates[0]
    print(f"✅ 自动锁定最佳权重: {os.path.basename(best_ckpt)}")
    return best_ckpt