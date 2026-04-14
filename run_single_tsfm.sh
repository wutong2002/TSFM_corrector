#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
# ================= 配置区 =================
DATA_ROOT="correction_datasets_new"
OUTPUT_ROOT="results/single_tsfm_grid_0122"

# === 1. 指定目标 TSFM ===
TARGET_TSFMS=("kairos_10m" "timesfm_2.5" "moirai_small" "chronos_bolt_tiny"  "tirex_base")

# === 2. 选择搜索模式 ===
GRID_MODE="std_tf_opt"

# === 3. 指定要跑的组合 ===
ENCODERS=("advanced_hybrid_math") #"timesfm" "moirai"  "units" "hybrid_math"
CORRECTORS=("std_tf")
for TARGET_TSFM in "${TARGET_TSFMS[@]}"; do
    # ================= 执行循环 =================
    echo "🚀 启动单模型专项网格搜索: [$TARGET_TSFM]"

    for enc in "${ENCODERS[@]}"; do
        for corr in "${CORRECTORS[@]}"; do
            
            echo "========================================================"
            echo "▶️  Task: Encoder=[$enc] | Corrector=[$corr]"
            echo "========================================================"
            
            python train_single_tsfm_grid.py \
                --encoder "$enc" \
                --corrector "$corr" \
                --target_tsfm "$TARGET_TSFM" \
                --grid_mode "$GRID_MODE" \
                --data_root "$DATA_ROOT" \
                --output_root "$OUTPUT_ROOT" \
                --seed 2025
                
            if [ $? -ne 0 ]; then
                echo "❌ 任务失败: $enc + $corr"
            else
                echo "✅ 任务完成"
            fi
            echo ""
        done
    done
    echo "✅ 单模型专项网格搜索完成: [$TARGET_TSFM]"
    echo ""
done
echo "🎉 全部任务完成!"