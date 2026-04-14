#!/bin/bash

# ====================================================
# 1. 环境与路径配置
# ====================================================
# [核心修复] 将 Datasets/processed_datasets 加入 PYTHONPATH，确保能找到 dataset_config.py
export PYTHONPATH="$PWD:$PWD/Datasets/processed_datasets:$PWD/UniTS-main:$PYTHONPATH"
export PYTHONUTF8=1

# === 基础模型配置 ===
MODEL_FAMILY="chronos"
MODEL_SIZE="bolt_tiny" 
CONTEXT_LEN=512
PRED_LEN=48

# === 数据源配置 ===
# 指向生成的 .pkl 校正数据路径
DB_PATH="correction_datasets/chronos_bolt_tiny/cl_original/correction_data"

# === 数据集分组 ===
# 使用去重后的训练集，防止数据污染
TRAIN_GROUP="lotsa_train_clean"
# 使用完整的测试集
TEST_GROUP="ge_test_all"

# === 缺失值控制开关 ===
# 1 = 允许并填充 (困难模式，对应 MSE 4.13)
# 0 = 严格剔除 (简单模式，对应 MSE 0.06)
ALLOW_MISSING=0

# === 输出目录 ===
OUTPUT_ROOT="results/grid_search_0119_2"
mkdir -p "$OUTPUT_ROOT"

# ====================================================
# 2. 定义搜索空间 (Grid Space)
# ====================================================
# 基础训练参数
EPOCHS=500
PATIENCE=300

# === [关键新增] 训练集平衡与加速参数 ===
# 每个数据集最多保留多少个样本 (Fail Fast 加速入库)
# 设为 2000 表示每个数据集只取前 2000 个样本
MAX_SAMPLES_PER_DS=2000

# === [新增] 测试集样本数限制 ===
# 每个测试数据集最多保留多少个样本
# 设为 -1 表示不限制，使用完整测试集
# 设为较小值（如 100）可加速验证过程
MAX_TEST_SAMPLES_PER_DS=-1

# 编码器参数statistical"" "timesfm"
ENCODER_TYPE_LIST=("units" )
EMBED_DIM_LIST=(128)

# 学习率
LR_LIST=(5e-5)

# Batch Size
BS_LIST=(64)

# 检索参数
TOP_K_LIST=(100)
DIVERSITY_LIST=(30)
RETRIEVAL_SCOPE="cross_dataset"
ENFORCE_CROSS_TIME_RESTRICTION=0

# 模型结构参数attention similarity_weighted  standard_transformer
ARCH_LIST=("standard_transformer")
HIDDEN_DIM_LIST=(128)
NUM_LAYERS_LIST=(8)
NUM_HEADS_LIST=(8)
SHUFFLE_RETRIEVED_ORDER_LIST=(True)  # 1 = 启用顺序打乱
# 特殊处理：使用空字符串表示默认值 (None)
DIM_FEEDFORWARD_LIST=("") 

DROPOUT_LIST=(0.3)
TEMPERATURE_LIST=(1.0)

# 训练技巧参数
OPTIMIZERS=("adam")
SCHEDULERS=("cosine")
WEIGHT_DECAY_LIST=(1e-4)

# === [新增] 数据检查参数 ===
ENABLE_DATA_CHECK_LIST=(0)  # 1 = 启用数据检查
CHECK_LEVEL_LIST=("full")  # 检查级别：basic/full
PSEUDO_RATIO_LIST=(0)  # 伪样本比例
PSEUDO_STRENGTH_LIST=(0)  # 伪样本强度

# === [新增] 归一化总开关 ===
# 0 = 启用归一化 (默认)
# 1 = 关闭所有归一化，直接在原始值尺度上学习
DISABLE_NORMALIZATION_LIST=(0)
# ====================================================
# 3. 执行网格搜索
# ====================================================
echo "🚀 开始网格搜索任务..."
echo "📂 数据源: $DB_PATH"
echo "⚙️ 缺失值填充: $ALLOW_MISSING"
echo "⚖️ 单数据集样本上限: $MAX_SAMPLES_PER_DS"
echo "🧠 架构: ${ARCH_LIST[*]}"
echo "📚 层数: ${NUM_LAYERS_LIST[*]}"
echo "🌊 检索多样性: ${DIVERSITY_LIST[*]}"
echo "🔄 编码器类型: ${ENCODER_TYPE_LIST[*]}"
echo "📅 调度器: ${SCHEDULERS[*]}"
echo "========================================================"

count=0
# 计算总任务数
total_runs=$((${#LR_LIST[@]} * ${#BS_LIST[@]} * ${#TOP_K_LIST[@]} * ${#DIVERSITY_LIST[@]} * ${#HIDDEN_DIM_LIST[@]} * ${#ARCH_LIST[@]} * ${#NUM_LAYERS_LIST[@]} * ${#ENCODER_TYPE_LIST[@]} * ${#DROPOUT_LIST[@]} * ${#SCHEDULERS[@]} * ${#WEIGHT_DECAY_LIST[@]} * ${#NUM_HEADS_LIST[@]}))

# 嵌套循环遍历所有超参数组合
for arch in "${ARCH_LIST[@]}"; do
  for encoder_type in "${ENCODER_TYPE_LIST[@]}"; do
    for embed_dim in "${EMBED_DIM_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        for bs in "${BS_LIST[@]}"; do
          for k in "${TOP_K_LIST[@]}"; do
            for div in "${DIVERSITY_LIST[@]}"; do  # [循环] 多样性
              for h_dim in "${HIDDEN_DIM_LIST[@]}"; do
                for n_layers in "${NUM_LAYERS_LIST[@]}"; do
                  for n_heads in "${NUM_HEADS_LIST[@]}"; do
                    for dropout in "${DROPOUT_LIST[@]}"; do
                      for scheduler in "${SCHEDULERS[@]}"; do
                        for weight_decay in "${WEIGHT_DECAY_LIST[@]}"; do
                          for dim_ff in "${DIM_FEEDFORWARD_LIST[@]}"; do
                          
                              count=$((count+1))
                              
                              # 构造实验 ID (包含所有超参数)
                              # 注意：如果 dim_ff 为空，ID 中显示 Default
                              ff_str="Default"
                              if [ -n "$dim_ff" ]; then ff_str="$dim_ff"; fi
                              
                              EXP_ID="${MODEL_FAMILY}_${arch}_${encoder_type}_L${n_layers}_H${n_heads}_lr${lr}_bs${bs}_topk${k}_div${div}_hdim${h_dim}_do${dropout}_sched${scheduler}_wd${weight_decay}_ff${ff_str}"
                              
                              if [ "$ALLOW_MISSING" -eq 0 ]; then
                                  EXP_ID="${EXP_ID}_strict"
                              fi
                              
                              SAVE_DIR="${OUTPUT_ROOT}/${EXP_ID}"
                              LOG_FILE="${SAVE_DIR}/train.log"
                              
                              mkdir -p "$SAVE_DIR"
                              
                              echo "[${count}/${total_runs}] Running: ${EXP_ID}"
                              echo "   Log: ${LOG_FILE}"
                              
                              # 构建命令数组
                              CMD=(python.exe train_corrector.py \
                                  --model_family "$MODEL_FAMILY" \
                                  --model_size "$MODEL_SIZE" \
                                  --context_len "$CONTEXT_LEN" \
                                  --pred_len "$PRED_LEN" \
                                  --encoder_type "$encoder_type" \
                                  --embed_dim "$embed_dim" \
                                  --retrieval_scope "$RETRIEVAL_SCOPE" \
                                  --enforce_cross_time_restriction "$ENFORCE_CROSS_TIME_RESTRICTION" \
                                  --corrector_arch "$arch" \
                                  --hidden_dim "$h_dim" \
                                  --num_layers "$n_layers" \
                                  --num_heads "$n_heads" \
                                  --dropout "$dropout" \
                                  --allow_missing_values "$ALLOW_MISSING" \
                                  --epochs "$EPOCHS" \
                                  --corrector_bs "$bs" \
                                  --lr "$lr" \
                                  --top_k "$k" \
                                  --diversity_max_per_dataset "$div" \
                                  --correction_output_dir "$SAVE_DIR" \
                                  --db_source_path "$DB_PATH" \
                                  --train_group "$TRAIN_GROUP" \
                                  --test_group "$TEST_GROUP" \
                                  --log_file "$LOG_FILE" \
                                  --optimizer "${OPTIMIZERS[0]}" \
                                  --scheduler "$scheduler" \
                                  --early_stop_patience "$PATIENCE" \
                                  --weight_decay "$weight_decay" \
                                  --timesfm_ckpt "checkpoints/timesfm-2.5-200m-pytorch" \
                                  --max_samples_per_dataset "$MAX_SAMPLES_PER_DS" \
                                  --max_test_samples_per_dataset "$MAX_TEST_SAMPLES_PER_DS" \
                                  --enable_data_check "${ENABLE_DATA_CHECK_LIST[0]}" \
                                  --check_level "${CHECK_LEVEL_LIST[0]}" \
                                  --debug 0 \
                                  --shuffle_retrieved_order "${SHUFFLE_RETRIEVED_ORDER_LIST[0]}" \
                                  --pseudo_ratio "${PSEUDO_RATIO_LIST[0]}" \
                                  --pseudo_strength "${PSEUDO_STRENGTH_LIST[0]}" \
                              )
                              
                              # 仅当 dim_ff 不为空时添加该参数
                              if [ -n "$dim_ff" ]; then
                                  CMD+=(--dim_feedforward "$dim_ff")
                              fi
                              
                              # 仅当需要关闭归一化时添加该参数
                              if [ "${DISABLE_NORMALIZATION_LIST[0]}" -eq 1 ]; then
                                  CMD+=(--disable_normalization)
                              fi

                              # 执行命令
                              "${CMD[@]}"

                              if [ $? -eq 0 ]; then
                                  echo "   ✅ 完成."
                              else
                                  echo "   ❌ 失败! 请检查日志."
                                  exit 1
                              fi
                              
                              echo "--------------------------------------------------------"
                              
                          done  # dim_feedforward
                        done  # weight_decay
                      done  # scheduler
                    done  # dropout
                  done  # num_heads
                done  # num_layers
              done  # hidden_dim
            done  # diversity
          done  # top_k
        done  # batch_size
      done  # learning_rate
    done  # embed_dim
  done  # encoder_type
done  # architecture

echo "🎉 所有实验结束！"