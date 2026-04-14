#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE

# ================= 配置区 =================

# 1. 数据集路径 (必须是包含 local_residuals 的新版数据)
# 对应 generate_correction_dataset.py 中的 OUTPUT_ROOT
DATA_ROOT="correction_datasets_double_res_0"

# 2. 结果保存路径
OUTPUT_ROOT="results/dual_source_experiment_0331_V2"

# 3. 指定目标基础模型 (Target TSFM) "kairos_10m" "tirex_base" "moirai_small" "tirex_base" "timesfm_2.5" "kairos_10m"
# 可以根据你的显卡资源增减"chronos_bolt_tiny"   "moirai_small"    "moirai_small" "kairos_10m" "timesfm_2.5"   "tirex_base" "moirai_small"  "timesfm_2.5" "moirai_small"
TARGET_TSFMS=("chronos_bolt_tiny" "timesfm_2.5"  "moirai_small" "tirex_base" "kairos_10m" )
#"chronos_bolt_tiny" "moirai_small" "tirex_base" "kairos_10m" 
# 4. 选择搜索模式 (对应 single_grid_config.py 中的新 Key)
# 这里我们使用刚才定义的 dual_mode_search 来遍历 alpha"dual_mode_search"
GRID_MODE="dual_mode_search"

# 5. 指定编码器和修正器 "units" 
# 编码器建议使用混合数学编码器或你认为最强的编码器"timesfm" "moirai"  "units"  "hybrid_math" "random_nn" "advanced_hybrid_math"  "random_nn_frozen" "units"
ENCODERS=("advanced_hybrid_math" ) 
# 修正器必须选择支持双输入的版本"dual_gated_mlp" "intra_inter_router" "mean_retrieval""global_bias" "dual_fusion_large" "dual_set_mlp"  "dual_res_mlp" "dual_gated_mlp" "local_ar" "dual_frozen_set_mlp" "dual_fusion_large" "dual_set_mlp"  "mean_retrieval" "dual_set_mlp"
CORRECTORS=("mean_retrieval" "dual_fusion_V2" "dual_latent_cross_attn")
#"mean_retrieval" "dual_fusion_V2" "dual_set_mlp"  "dual_latent_cross_attn"
# ================= 执行循环 =================

for TARGET_TSFM in "${TARGET_TSFMS[@]}"; do
    echo "🚀 启动双重指纹修正实验: [$TARGET_TSFM]"
    echo "📂 数据源: $DATA_ROOT"

    for enc in "${ENCODERS[@]}"; do
        for corr in "${CORRECTORS[@]}"; do
            
            echo "--------------------------------------------------------"
            echo "▶️  Running: TSFM=[$TARGET_TSFM] | Encoder=[$enc] | Model=[$corr]"
            echo "    Grid Mode: [$GRID_MODE] (Searching Alpha...)"
            echo "--------------------------------------------------------"
            
            # 注意：这里我们不通过命令行传 --retrieval_alpha
            # 因为我们在 GRID_MODE 里定义了 alpha 的列表 [1.0, 0.5, 0.0]
            # python 脚本会自动遍历这些值
            
            python train_single_double_res.py \
                --encoder "$enc" \
                --corrector "$corr" \
                --target_tsfm "$TARGET_TSFM" \
                --grid_mode "$GRID_MODE" \
                --data_root "$DATA_ROOT" \
                --output_root "$OUTPUT_ROOT" \
                --seed 2025
                
            if [ $? -ne 0 ]; then
                echo "❌ 任务失败: $enc + $corr"
                # exit 1  # 如果希望遇到错误就停止，取消注释
            else
                echo "✅ 任务完成"
            fi
            echo ""
        done
    done
    echo "✅ [$TARGET_TSFM] 所有实验组合完成"
    echo ""
done

echo "🎉🎉🎉 所有双源修正实验已结束! 请查看 $OUTPUT_ROOT 下的结果。"