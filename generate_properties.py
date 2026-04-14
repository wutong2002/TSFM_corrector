import os
import json
from datasets import load_from_disk
from tqdm import tqdm

# [配置]
PROCESSED_DIR = "Datasets/processed_datasets"
OUTPUT_JSON = "Datasets/processed_datasets/dataset_properties.json"

def get_domain_from_name(name):
    """
    根据数据集名称推断领域
    逻辑涵盖 GE, LOTSA 以及 QuitoBench 数据集的命名规则
    """
    name = name.lower()
    
    # === QuitoBench (应用流量/云计算工作负载) ===
    # 基于文献：来自支付宝生产平台的应用流量数据，涵盖九个业务领域
    if name.startswith("qb_") or "quito" in name:
        return "Application Traffic"
        
    # === Traffic (交通) ===
    if any(k in name for k in ["traffic", "pems", "kdd", "subway", "metro", "taxi", "rideshare", "vehicle", "loop"]):
        return "Traffic"
    
    # === Nature (自然/气象/环境) ===
    if any(k in name for k in ["weather", "rain", "climate", "nature", "air", "wind", "temperature", "river", "hydrology"]):
        return "Nature"
    
    # === Energy (能源/电力) ===
    if any(k in name for k in ["elec", "energy", "solar", "power", "load", "battery", "grid"]):
        return "Energy"
    
    # === Healthcare (医疗/健康) ===
    if any(k in name for k in ["hospital", "covid", "flu", "health", "disease", "ilinet"]):
        return "Healthcare"
    
    # === Economic (经济/金融) ===
    if any(k in name for k in ["m4", "m3", "finance", "economic", "bitcoin", "exchange", "gdp", "cpi", "fred"]):
        return "Economic"
    
    # === Sales (销售/零售) ===
    if any(k in name for k in ["sales", "m5", "retail", "demand", "favorita", "walmart"]):
        return "Sales"
    
    # === Other (其他：服务器日志、人口统计等) ===
    return "Other"

def generate_properties():
    # 1. 尝试加载现有文件，实现增量更新
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                properties = json.load(f)
            print(f"✅ 已加载现有属性文件，包含 {len(properties)} 条记录")
        except Exception as e:
            print(f"⚠️ 现有文件读取失败 ({e})，将重新创建")
            properties = {}
    else:
        properties = {}
        print("⚠️ 未找到现有属性文件，将创建新文件")

    # 2. 扫描数据目录
    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ 数据目录不存在: {PROCESSED_DIR}")
        return

    # 获取所有子目录
    datasets = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    print(f"🔍 扫描到目录中共有 {len(datasets)} 个数据集文件夹")
    
    updated_count = 0
    
    for ds_name in tqdm(datasets, desc="Generating Properties"):
        ds_path = os.path.join(PROCESSED_DIR, ds_name)
        key_name = ds_name.lower()
        
        try:
            # 加载数据集以获取频率信息
            ds = load_from_disk(ds_path)
            
            freq = "H" # 默认频率
            if len(ds) > 0:
                if 'freq' in ds[0]:
                    raw_freq = ds[0]['freq']
                    # 频率标准化
                    if raw_freq == '1H': freq = 'H'
                    elif raw_freq == '1min': freq = 'min'
                    elif raw_freq == '10T': freq = '10T' # QuitoBench 的 10 分钟频率
                    elif raw_freq == '1D': freq = 'D'
                    else: freq = raw_freq
            
            # 构建属性字典
            properties[key_name] = {
                "domain": get_domain_from_name(ds_name),
                "frequency": freq, 
                "dim": 1,
                "num_variates": 1,
                "prediction_length": 96
            }
            updated_count += 1
            
        except Exception as e:
            # 如果加载失败跳过
            pass
            
    # 3. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(properties, f, indent=4, sort_keys=True)
        
    print(f"✅ 属性文件已更新: {OUTPUT_JSON}")
    print(f"📊 总条目数: {len(properties)} (本次处理/更新: {updated_count})")

if __name__ == "__main__":
    generate_properties()