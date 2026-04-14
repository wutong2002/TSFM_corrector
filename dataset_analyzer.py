#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析脚本

功能：
- 复刻train_corrector.py中的数据构造逻辑
- 进行全方位的数据集分析
- 生成可视化图表
- 找出影响训练效果的潜在因素

使用方法：
python dataset_analyzer.py
"""

# 导入必要的库
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
sys.path.append('.')

# 配置参数
CONFIG = {
    'data_dir': 'correction_datasets/chronos_bolt_tiny/cl_original/correction_data',  # 数据目录
    'context_len': 512,  # 上下文长度
    'allow_missing_values': 1,  # 是否允许缺失值
    'max_samples_per_dataset': 2000,  # 单数据集样本上限
    'train_datasets_list': ['train'],  # 训练集列表
    'test_datasets_list': ['test'],  # 测试集列表
    'disable_pbar': False
}

class DatasetAnalyzer:
    """数据集分析器"""
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        self.target_len = config['context_len']
        self.allow_missing = config['allow_missing_values']
        self.max_samples_per_dataset = config['max_samples_per_dataset']
        self.train_list = [d.lower() for d in config['train_datasets_list']]
        self.test_list = [d.lower() for d in config['test_datasets_list']]
        
        # 数据质量参数
        self.max_missing_ratio = 0.20
        self.max_consecutive_nan = 12
        
        # 加载数据
        self.all_samples = self._load_dump_files()
        self.train_samples, self.test_samples = self._split_samples()
        
        print(f"总样本数: {len(self.all_samples)}")
        print(f"训练集样本数: {len(self.train_samples)}")
        print(f"测试集样本数: {len(self.test_samples)}")
    
    def _load_dump_files(self):
        """加载dump文件"""
        samples = []
        if not os.path.exists(self.data_dir):
            print(f"错误：数据目录 {self.data_dir} 不存在")
            return []
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        for f in tqdm(files, desc="加载PKL文件"):
            try:
                with open(os.path.join(self.data_dir, f), 'rb') as fp:
                    data = pickle.load(fp)
                
                dataset_name = data.get('dataset_name', f.split('_correction_data')[0])
                times = data.get('times', data.get('timestamps', []))
                histories = data.get('histories', [])
                residuals = data.get('residuals', [])
                truths = data.get('truths', [])
                
                count = min(len(histories), len(residuals))
                for i in range(count):
                    samples.append({
                        'timestamp': times[i] if i < len(times) else None,
                        'history': histories[i],
                        'residual': residuals[i],
                        'truth': truths[i] if i < len(truths) else None,
                        'source': f,
                        'dataset': dataset_name
                    })
            except Exception as e:
                print(f"加载文件 {f} 失败: {e}")
                continue
        
        return samples
    
    def _split_samples(self):
        """分割训练集和测试集"""
        train_samples = []
        test_samples = []
        dataset_counters = defaultdict(int)
        role_cache = {}
        
        for item in tqdm(self.all_samples, desc="分割样本"):
            ds_name = item.get('dataset', 'unknown').lower()
            
            if ds_name not in role_cache:
                _is_train = any(allowed in ds_name for allowed in self.train_list)
                _is_test = any(allowed in ds_name for allowed in self.test_list)
                role_cache[ds_name] = (_is_train, _is_test)
            
            is_train, is_test = role_cache[ds_name]
            
            if not is_train and not is_test:
                continue
                
            if is_train and self.max_samples_per_dataset > 0:
                if dataset_counters[ds_name] >= self.max_samples_per_dataset:
                    continue
                    
            # 检查数据质量
            valid, _ = self._check_sample_quality(item)
            if not valid:
                continue
                
            if is_train:
                train_samples.append(item)
                dataset_counters[ds_name] += 1
            elif is_test:
                test_samples.append(item)
        
        return train_samples, test_samples
    
    def _check_sample_quality(self, item):
        """检查样本质量"""
        h_raw = np.array(item.get('history', []), dtype=np.float32)
        r_raw = np.array(item.get('residual', []), dtype=np.float32)
        t_raw = np.array(item.get('truth', []), dtype=np.float32)
        
        valid_h, _ = self._check_data_quality(h_raw)
        valid_r, _ = self._check_data_quality(r_raw)
        valid_t, _ = self._check_data_quality(t_raw)
        
        if self.allow_missing:
            return (valid_h and valid_r and valid_t), 'valid'
        else:
            return (self._is_clean(h_raw) and self._is_clean(r_raw) and self._is_clean(t_raw)), 'valid'
    
    def _check_data_quality(self, arr):
        """检查数据质量"""
        if len(arr) == 0:
            return False, 'empty'
        
        is_bad = ~np.isfinite(arr)
        bad_count = np.sum(is_bad)
        
        if bad_count == 0:
            return True, 'clean'
        
        ratio = bad_count / len(arr)
        if ratio > self.max_missing_ratio:
            return False, f'high_missing_ratio({ratio:.2f})'
        
        # 检查连续NaN
        padded = np.concatenate(([False], is_bad, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) > 0:
            max_len = np.max(ends - starts)
            if max_len > self.max_consecutive_nan:
                return False, f'long_consecutive_missing({max_len})'
        
        return True, 'valid_missing'
    
    def _is_clean(self, arr):
        """检查数据是否完全干净"""
        return np.isfinite(arr).all()

class DataQualityAnalyzer:
    """数据质量分析器"""
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def analyze(self):
        """执行全面分析"""
        print("\n" + "="*50)
        print("=== 数据质量分析 ===")
        print("="*50)
        self._basic_statistics()
        self._missing_values_analysis()
        self._outlier_analysis()
        self._length_analysis()
        self._distribution_analysis()
    
    def _basic_statistics(self):
        """基本统计信息"""
        print("\n--- 基本统计信息 ---")
        datasets = set([s['dataset'] for s in self.analyzer.all_samples])
        print(f"数据集数量: {len(datasets)}")
        print(f"数据集列表: {sorted(datasets)}")
        
        # 按数据集统计
        dataset_counts = Counter([s['dataset'] for s in self.analyzer.all_samples])
        print("\n各数据集样本数:")
        for ds, count in dataset_counts.most_common():
            print(f"  {ds}: {count}")
    
    def _missing_values_analysis(self):
        """缺失值分析"""
        print("\n--- 缺失值分析 ---")
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            total_nan = 0
            total_inf = 0
            sample_count = 0
            
            for sample in samples:
                has_issue = False
                
                # 检查history
                hist = np.array(sample.get('history', []))
                nan_hist = np.sum(np.isnan(hist))
                inf_hist = np.sum(np.isinf(hist))
                total_nan += nan_hist
                total_inf += inf_hist
                
                # 检查residual
                res = np.array(sample.get('residual', []))
                nan_res = np.sum(np.isnan(res))
                inf_res = np.sum(np.isinf(res))
                total_nan += nan_res
                total_inf += inf_res
                
                # 检查truth
                truth = np.array(sample.get('truth', []))
                nan_truth = np.sum(np.isnan(truth))
                inf_truth = np.sum(np.isinf(truth))
                total_nan += nan_truth
                total_inf += inf_truth
                
                if nan_hist > 0 or inf_hist > 0 or nan_res > 0 or inf_res > 0 or nan_truth > 0 or inf_truth > 0:
                    sample_count += 1
            
            total_values = sum(len(np.array(s.get('history', []))) + len(np.array(s.get('residual', []))) + len(np.array(s.get('truth', []))) for s in samples)
            nan_ratio = (total_nan / total_values) * 100 if total_values > 0 else 0
            inf_ratio = (total_inf / total_values) * 100 if total_values > 0 else 0
            
            print(f"\n{set_name} (共 {len(samples)} 个样本):")
            if len(samples) > 0:
                print(f"  有问题样本数: {sample_count} ({sample_count/len(samples)*100:.2f}%)")
            else:
                print(f"  有问题样本数: {sample_count} (0.00%)")
            print(f"  总NaN值: {total_nan} ({nan_ratio:.4f}%)")
            print(f"  总Inf值: {total_inf} ({inf_ratio:.4f}%)")
    
    def _outlier_analysis(self):
        """异常值分析"""
        print("\n--- 异常值分析 ---")
        
        def detect_outliers(arr):
            if len(arr) < 4 or not np.isfinite(arr).any():
                return [], 0
            
            arr_clean = arr[np.isfinite(arr)]
            if len(arr_clean) < 4:
                return [], 0
            
            Q1 = np.percentile(arr_clean, 25)
            Q3 = np.percentile(arr_clean, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = arr_clean[(arr_clean < lower_bound) | (arr_clean > upper_bound)]
            return outliers, len(outliers)/len(arr_clean)*100
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            all_outliers = []
            total_outlier_ratio = 0
            sample_count = 0
            
            for sample in samples:
                # 检查residual（最关键的特征）
                res = np.array(sample.get('residual', []))
                outliers, ratio = detect_outliers(res)
                if len(outliers) > 0:
                    all_outliers.extend(outliers)
                    total_outlier_ratio += ratio
                    sample_count += 1
            
            avg_outlier_ratio = total_outlier_ratio / len(samples) if len(samples) > 0 else 0
        print(f"\n{set_name}:")
        if len(samples) > 0:
            print(f"  含异常值样本数: {sample_count} ({sample_count/len(samples)*100:.2f}%)")
        else:
            print(f"  含异常值样本数: {sample_count} (0.00%)")
        print(f"  平均异常值比例: {avg_outlier_ratio:.2f}%")
        print(f"  总异常值数: {len(all_outliers)}")
        if all_outliers:
            print(f"  异常值范围: [{min(all_outliers):.2f}, {max(all_outliers):.2f}]")
    
    def _length_analysis(self):
        """序列长度分析"""
        print("\n--- 序列长度分析 ---")
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            lengths = {
                'history': [len(np.array(s.get('history', []))) for s in samples],
                'residual': [len(np.array(s.get('residual', []))) for s in samples],
                'truth': [len(np.array(s.get('truth', []))) for s in samples]
            }
            
            print(f"\n{set_name}序列长度统计:")
            for field, lens in lengths.items():
                if lens:  # 确保列表不为空
                    print(f"  {field}: 均值={np.mean(lens):.2f}, 中位数={np.median(lens):.2f}, 最小={np.min(lens)}, 最大={np.max(lens)}")
    
    def _distribution_analysis(self):
        """分布分析"""
        print("\n--- 分布分析 ---")
        
        def get_stats(arr):
            arr_clean = arr[np.isfinite(arr)]
            if len(arr_clean) == 0:
                return None
            return {
                'mean': np.mean(arr_clean),
                'std': np.std(arr_clean),
                'min': np.min(arr_clean),
                'max': np.max(arr_clean),
                'median': np.median(arr_clean)
            }
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            print(f"\n{set_name}特征分布统计:")
            
            # 收集所有residual值（最关键的特征）
            all_residuals = []
            for sample in samples:
                res = np.array(sample.get('residual', []))
                all_residuals.extend(res[np.isfinite(res)].tolist())
            
            all_residuals = np.array(all_residuals)
            if len(all_residuals) > 0:
                stats = get_stats(all_residuals)
                if stats:
                    print(f"  Residual统计: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
                    print(f"                最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}")
                    print(f"                中位数={stats['median']:.4f}")

class DataVisualizer:
    """数据可视化器"""
    def __init__(self, analyzer):
        self.analyzer = analyzer
        # 创建输出目录
        self.output_dir = 'analysis_plots'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize(self):
        """执行所有可视化"""
        print("\n" + "="*50)
        print("=== 数据可视化 ===")
        print("="*50)
        
        self._plot_dataset_distribution()
        self._plot_missing_values()
        self._plot_residual_distribution()
        self._plot_sequence_lengths()
        self._plot_sample_examples()
        
        print(f"\n所有图表已保存至: {self.output_dir}")
    
    def _plot_dataset_distribution(self):
        """绘制数据集分布"""
        print("\n--- 数据集分布 ---")
        
        # 统计各数据集样本数
        train_counts = Counter([s['dataset'] for s in self.analyzer.train_samples])
        test_counts = Counter([s['dataset'] for s in self.analyzer.test_samples])
        
        # 合并数据集列表
        all_datasets = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
        
        train_values = [train_counts.get(ds, 0) for ds in all_datasets]
        test_values = [test_counts.get(ds, 0) for ds in all_datasets]
        
        # 绘制图表
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.35
        x = np.arange(len(all_datasets))
        
        ax.bar(x - bar_width/2, train_values, bar_width, label='训练集', color='skyblue')
        ax.bar(x + bar_width/2, test_values, bar_width, label='测试集', color='salmon')
        
        ax.set_xlabel('数据集')
        ax.set_ylabel('样本数')
        ax.set_title('各数据集样本分布')
        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dataset_distribution.png'), dpi=300)
        plt.show()
    
    def _plot_missing_values(self):
        """绘制缺失值分布"""
        print("\n--- 缺失值分布 ---")
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            nan_ratios = []
            
            for sample in samples:
                res = np.array(sample.get('residual', []))
                if len(res) > 0:
                    nan_ratio = np.sum(np.isnan(res)) / len(res)
                    nan_ratios.append(nan_ratio)
            
            # 绘制直方图
            plt.figure(figsize=(10, 5))
            plt.hist(nan_ratios, bins=50, alpha=0.7, color='skyblue')
            plt.xlabel('NaN比例')
            plt.ylabel('样本数')
            plt.title(f'{set_name}残差NaN比例分布')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, f'{set_name}_missing_values.png'), dpi=300)
            plt.show()
    
    def _plot_residual_distribution(self):
        """绘制残差分布"""
        print("\n--- 残差分布 ---")
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            all_residuals = []
            for sample in samples:
                res = np.array(sample.get('residual', []))
                all_residuals.extend(res[np.isfinite(res)].tolist())
            
            all_residuals = np.array(all_residuals)
            if len(all_residuals) > 0:
                # 限制显示范围，避免极端值影响
                q1 = np.percentile(all_residuals, 1)
                q99 = np.percentile(all_residuals, 99)
                filtered_residuals = all_residuals[(all_residuals >= q1) & (all_residuals <= q99)]
                
                # 绘制直方图和核密度估计
                plt.figure(figsize=(10, 5))
                sns.histplot(filtered_residuals, bins=100, kde=True, alpha=0.7, color='purple')
                plt.xlabel('残差值')
                plt.ylabel('频率')
                plt.title(f'{set_name}残差分布 (显示1%-99%分位数)')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, f'{set_name}_residual_distribution.png'), dpi=300)
                plt.show()
    
    def _plot_sequence_lengths(self):
        """绘制序列长度分布"""
        print("\n--- 序列长度分布 ---")
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            lengths = {
                'history': [len(np.array(s.get('history', []))) for s in samples],
                'residual': [len(np.array(s.get('residual', []))) for s in samples],
                'truth': [len(np.array(s.get('truth', []))) for s in samples]
            }
            
            # 绘制箱线图
            plt.figure(figsize=(12, 6))
            
            # 准备数据
            data = [lengths[field] for field in ['history', 'residual', 'truth']]
            labels = ['History', 'Residual', 'Truth']
            
            plt.boxplot(data, labels=labels, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red'))
            
            plt.xlabel('特征')
            plt.ylabel('序列长度')
            plt.title(f'{set_name}序列长度分布')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, f'{set_name}_sequence_lengths.png'), dpi=300)
            plt.show()
    
    def _plot_sample_examples(self):
        """绘制样本示例"""
        print("\n--- 样本示例 ---")
        
        # 随机选择几个样本进行可视化
        import random
        random.seed(2025)
        
        for set_name, samples in [('训练集', self.analyzer.train_samples), ('测试集', self.analyzer.test_samples)]:
            if len(samples) == 0:
                continue
                
            # 随机选择3个样本
            selected_samples = random.sample(samples, min(3, len(samples)))
            
            for i, sample in enumerate(selected_samples):
                plt.figure(figsize=(12, 8))
                
                # 获取数据
                history = np.array(sample.get('history', []))
                residual = np.array(sample.get('residual', []))
                truth = np.array(sample.get('truth', [])) if sample.get('truth') is not None else None
                
                # 绘制history
                plt.subplot(3, 1, 1)
                plt.plot(history, label='History', color='blue')
                plt.title(f'{set_name} 样本 {i+1} - History')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # 绘制residual
                plt.subplot(3, 1, 2)
                plt.plot(residual, label='Residual', color='green')
                plt.title(f'{set_name} 样本 {i+1} - Residual')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # 绘制truth
                if truth is not None:
                    plt.subplot(3, 1, 3)
                    plt.plot(truth, label='Truth', color='red')
                    plt.title(f'{set_name} 样本 {i+1} - Truth')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{set_name}_sample_{i+1}.png'), dpi=300)
                plt.show()

class ComprehensiveAnalyzer:
    """综合分析器"""
    def __init__(self, config):
        self.analyzer = DatasetAnalyzer(config)
        self.quality_analyzer = DataQualityAnalyzer(self.analyzer)
        self.visualizer = DataVisualizer(self.analyzer)
    
    def run_analysis(self):
        """运行全面分析"""
        print("🚀 开始数据集分析任务")
        print(f"📂 数据源: {self.analyzer.data_dir}")
        print(f"⚙️ 缺失值填充: {self.analyzer.allow_missing}")
        print(f"⚖️ 单数据集样本上限: {self.analyzer.max_samples_per_dataset}")
        
        # 执行质量分析
        self.quality_analyzer.analyze()
        
        # 执行可视化
        self.visualizer.visualize()
        
        # 生成综合报告
        self._generate_report()
    
    def _generate_report(self):
        """生成综合报告"""
        print("\n" + "="*50)
        print("=== 综合分析报告 ===")
        print("="*50)
        
        print("\n1. 数据基本情况")
        print(f"   - 总样本数: {len(self.analyzer.all_samples)}")
        print(f"   - 训练集样本数: {len(self.analyzer.train_samples)}")
        print(f"   - 测试集样本数: {len(self.analyzer.test_samples)}")
        print(f"   - 数据集数量: {len(set([s['dataset'] for s in self.analyzer.all_samples]))}")
        
        print("\n2. 数据质量评估")
        # 计算整体质量指标
        all_train_residuals = []
        for sample in self.analyzer.train_samples:
            res = np.array(sample.get('residual', []))
            all_train_residuals.extend(res[np.isfinite(res)].tolist())
        
        all_train_residuals = np.array(all_train_residuals)
        if len(all_train_residuals) > 0:
            print(f"   - 训练集残差标准差: {np.std(all_train_residuals):.4f}")
            print(f"   - 训练集残差范围: [{np.min(all_train_residuals):.4f}, {np.max(all_train_residuals):.4f}]")
        
        print("\n3. 潜在问题识别")
        # 检查是否存在可能影响训练的问题
        if len(self.analyzer.train_samples) < 100:
            print("   ⚠️ 警告: 训练集样本数量过少")
        
        # 检查残差分布是否合理
        if len(all_train_residuals) > 0:
            std_res = np.std(all_train_residuals)
            if std_res > 10:
                print("   ⚠️ 警告: 残差标准差过大，可能存在异常值")
            if np.abs(np.mean(all_train_residuals)) > 1:
                print("   ⚠️ 警告: 残差均值偏离零，可能存在系统偏差")
        
        print("\n4. 建议改进措施")
        print("   - 检查数据预处理流程，确保异常值得到合理处理")
        print("   - 考虑对残差进行标准化或归一化")
        print("   - 验证数据分割策略，确保训练集和测试集分布一致")
        print("   - 考虑增加数据增强或正则化技术")
        
        print("\n" + "="*50)
        print("分析完成！")
        print("="*50)

if __name__ == "__main__":
    # 创建综合分析器
    comprehensive_analyzer = ComprehensiveAnalyzer(CONFIG)
    
    # 运行分析
    comprehensive_analyzer.run_analysis()
