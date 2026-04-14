import torch
import numpy as np
from database.dataset import CorrectionDataset
from database.manager import SchoolwareDB
from encoder.statistical import AdvancedStatisticalEncoder
from retriever.engine import ExactCosineRetriever

def debug_pseudo_samples():
    """
    调试伪样本构造过程
    """
    print("="*60)
    print("伪样本构造调试")
    print("="*60)
    
    # 创建一个简单的样本
    sample = {
        'history': np.random.rand(512) * 100,  # 历史数据，尺度较大
        'residual': np.random.rand(96) * 10,   # 残差，尺度较小
        'truth': np.random.rand(96) * 100,      # 真实值
        'scaled_residual': None  # 让dataset自动计算
    }
    
    print(f"\n原始样本统计:")
    print(f"  history: mean={np.mean(sample['history']):.4f}, std={np.std(sample['history']):.4f}")
    print(f"  residual: mean={np.mean(sample['residual']):.4f}, std={np.std(sample['residual']):.4f}")
    
    # 计算归一化尺度（模拟dataset的逻辑）
    scale = np.mean(np.abs(sample['history'])) + 1e-6
    scaled_res = np.clip(sample['residual'] / scale, -20, 20)
    
    print(f"\n归一化计算:")
    print(f"  scale (mean abs history): {scale:.4f}")
    print(f"  scaled_res: mean={np.mean(scaled_res):.4f}, std={np.std(scaled_res):.4f}")
    print(f"  反归一化验证: mean={np.mean(scaled_res * scale):.4f} (应接近原始残差均值)")
    
    # 模拟伪样本构造
    print(f"\n伪样本构造模拟 (pseudo_strength=0):")
    
    # 转换为tensor
    target_emb = torch.randn(1, 128)  # 假设embedding维度是128
    target_res = torch.tensor(scaled_res, dtype=torch.float32).unsqueeze(0)  # 归一化的残差
    
    num_pseudo = 3
    
    # 复制形成基础
    pseudo_embs = target_emb.repeat(num_pseudo, 1)
    pseudo_res = target_res.repeat(num_pseudo, 1)
    
    print(f"  原始 target_res: shape={target_res.shape}, mean={target_res.mean().item():.4f}, std={target_res.std().item():.4f}")
    print(f"  伪样本 pseudo_res: shape={pseudo_res.shape}, mean={pseudo_res.mean().item():.4f}, std={pseudo_res.std().item():.4f}")
    
    # 问题：伪样本的残差是归一化的，但模型期望原始残差
    print(f"\n⚠️  问题发现:")
    print(f"  伪样本的残差是归一化的 (mean={pseudo_res.mean().item():.4f})")
    print(f"  但模型的value_proj期望的是原始残差")
    print(f"  这会导致模型学习到错误的尺度")
    
    # 正确的做法：应该使用原始残差构造伪样本
    print(f"\n✅ 正确做法:")
    print(f"  1. 使用原始残差构造伪样本")
    print(f"  2. 或者在模型中明确处理归一化的残差")
    
    # 验证：如果使用原始残差构造伪样本
    target_res_raw = torch.tensor(sample['residual'], dtype=torch.float32).unsqueeze(0)
    pseudo_res_raw = target_res_raw.repeat(num_pseudo, 1)
    
    print(f"\n使用原始残差构造伪样本:")
    print(f"  原始 target_res_raw: mean={target_res_raw.mean().item():.4f}, std={target_res_raw.std().item():.4f}")
    print(f"  伪样本 pseudo_res_raw: mean={pseudo_res_raw.mean().item():.4f}, std={pseudo_res_raw.std().item():.4f}")
    
    return scale, scaled_res, sample['residual']

def debug_model_forward():
    """
    调试模型前向传播
    """
    from corrector.corrector_model import DeepTransformerCorrector
    
    print("\n" + "="*60)
    print("模型前向传播调试")
    print("="*60)
    
    # 创建模型
    model_config = {
        'embed_dim': 128,
        'hidden_dim': 256,
        'pred_len': 96,
        'num_heads': 8,
        'dropout': 0.1,
        'num_layers': 3,
        'dim_feedforward': 1024,
        'normalize': True
    }
    
    model = DeepTransformerCorrector(model_config)
    model.eval()
    
    # 创建输入
    batch_size = 2
    top_k = 5
    
    target_emb = torch.randn(batch_size, 128)
    retrieved_embs = torch.randn(batch_size, top_k, 128)
    retrieved_residuals = torch.randn(batch_size, top_k, 96) * 10  # 原始残差，尺度较大
    history = torch.randn(batch_size, 512) * 100  # 历史数据
    
    print(f"\n输入统计:")
    print(f"  retrieved_residuals: mean={retrieved_residuals.mean().item():.4f}, std={retrieved_residuals.std().item():.4f}")
    print(f"  history: mean={history.mean().item():.4f}, std={history.std().item():.4f}")
    
    # 前向传播
    with torch.no_grad():
        pred_raw, info = model(target_emb, retrieved_embs, retrieved_residuals, history)
        pred_norm = info.get('pred_res_normalized', pred_raw)
    
    print(f"\n输出统计:")
    print(f"  pred_raw (反归一化后): mean={pred_raw.mean().item():.4f}, std={pred_raw.std().item():.4f}")
    print(f"  pred_norm (归一化后): mean={pred_norm.mean().item():.4f}, std={pred_norm.std().item():.4f}")
    
    # 验证归一化
    label_norm, stats = model.normalize_residual(retrieved_residuals[:, 0, :], history)
    print(f"\n归一化验证:")
    print(f"  标签归一化后: mean={label_norm.mean().item():.4f}, std={label_norm.std().item():.4f}")
    print(f"  使用的scale: mean={stats['std'].mean().item():.4f}")
    
    # 反归一化验证
    denorm_pred = model.denormalize_residual(pred_norm, stats)
    print(f"\n反归一化验证:")
    print(f"  pred_norm * scale: mean={(pred_norm * stats['std']).mean().item():.4f}")
    print(f"  denorm_pred: mean={denorm_pred.mean().item():.4f}")
    print(f"  pred_raw: mean={pred_raw.mean().item():.4f} (应接近denorm_pred)")

if __name__ == "__main__":
    scale, scaled_res, raw_res = debug_pseudo_samples()
    debug_model_forward()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"\n问题根源:")
    print(f"  1. 伪样本使用归一化的残差构造")
    print(f"  2. 模型期望输入原始残差")
    print(f"  3. 这导致模型学习到错误的尺度")
    print(f"\n解决方案:")
    print(f"  1. 在dataset中使用原始残差构造伪样本")
    print(f"  2. 或者在模型中明确处理归一化的残差")
