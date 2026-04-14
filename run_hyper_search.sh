#!/bin/bash

# ================= 配置区 =================
DATA_ROOT="correction_datasets_new"
# 结果保存到专门的目录
OUTPUT_ROOT="results/hyper_search_v1"

# === 1. 指定要搜索的空间 ===
# 对应 hyper_config.py 中的 key (例如 "retrieval_search" 或 "optim_search")
SEARCH_SPACE="full_grid"

# === 2. 锁定 Encoder (建议一次跑一个，控制显存) ===
# ENCODERS=('fft' 'timesfm') 
ENCODER="units"

# === 3. 遍历 Corrector ===
CORRECTORS=("sim_weight" "std_tf")

# ================= 执行循环 =================
echo "🚀 开始超参数网格搜索..."
echo "📂 数据源: $DATA_ROOT"
echo "🔍 搜索空间: $SEARCH_SPACE"

for corr in "${CORRECTORS[@]}"; do
    
    echo "========================================================"
    echo "▶️  Processing: Encoder=[$ENCODER] | Corrector=[$corr]"
    echo "========================================================"
    
    python train_hyper_search.py \
        --encoder_config "$ENCODER" \
        --corrector_name "$corr" \
        --search_space "$SEARCH_SPACE" \
        --data_root "$DATA_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --train_group "lotsa_train_clean" \
        --test_group "ge_test_all" \
        --seed 2025
        
    echo "✅ 完成 $corr 的搜索"
    echo ""
done

echo "🎉 所有超参搜索任务完成！结果保存在: $OUTPUT_ROOT"