#!/bin/bash

# ================= 配置区 =================
DATA_ROOT="correction_datasets_new"
OUTPUT_ROOT="results/grid_search_0122_2"

# === 遍历目标 (Encoders) ===
ENCODERS=('hybrid_math' 'timesfm' 'units' 'moirai') # "fft" "stats" "wavelet"  

# === 目标 Correctors (空格分隔的列表) ===
# 这里把所有要跑的一次性传进去
CORRECTORS_LIST="meta_corrector learnable_weight weighted_base sim_weight std_tf"

TRAIN_CONF="standard" # debug

# ================= 执行循环 =================
echo "🚀 开始优化版网格搜索..."
echo "📂 数据源: $DATA_ROOT"
echo "🎯 待测 Correctors: $CORRECTORS_LIST"

for enc in "${ENCODERS[@]}"; do
    
    echo "========================================================"
    echo "▶️  Processing Encoder Group: [$enc]"
    echo "========================================================"
    
    # 调用 python，传入 corrector 列表
    # 注意：$CORRECTORS_LIST 不要加引号，以便被解析为多个参数
    python train_corrector.py \
        --encoder_config "$enc" \
        --corrector_configs $CORRECTORS_LIST \
        --train_config "$TRAIN_CONF" \
        --data_root "$DATA_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --train_group "lotsa_train_clean" \
        --test_group "ge_test_all" \
        --seed 2025
        
    if [ $? -ne 0 ]; then
        echo "❌ Group $enc failed!"
    fi
    
done

echo "🎉 All Grid Search Complete!"