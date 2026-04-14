#!/bin/bash
# 优化器对比实验脚本

echo "=========================================="
echo "  优化器对比实验 (SGD vs AdamW)"
echo "=========================================="
echo ""

# 公共参数
COMMON_ARGS="\
    --model_family TimesFM \
    --model_size 200m \
    --context_len 512 \
    --pred_len 96 \
    --corrector_arch deep_transformer \
    --corrector_bs 64 \
    --epochs 30 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --debug 1"

echo "📋 实验计划:"
echo "   1. AdamW (baseline)"
echo "   2. SGD with momentum"
echo "   3. SGD with Nesterov momentum"
echo ""

read -p "是否开始对比实验? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "实验已取消"
    exit 1
fi

# 实验1: AdamW (baseline)
echo ""
echo "🚀 实验1: AdamW (baseline)"
echo ""
python train_corrector.py $COMMON_ARGS \
    --optimizer adamw \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --log_file "optimizer_adamw.log"

echo ""
echo "✅ 实验1完成"
echo ""

# 实验2: SGD with momentum
echo "🚀 实验2: SGD with momentum"
echo ""
python train_corrector.py $COMMON_ARGS \
    --optimizer sgd \
    --lr 0.01 \
    --weight_decay 1e-4 \
    --log_file "optimizer_sgd.log"

echo ""
echo "✅ 实验2完成"
echo ""

# 实验3: SGD with Nesterov
echo "🚀 实验3: SGD with Nesterov momentum"
echo ""
python train_corrector.py $COMMON_ARGS \
    --optimizer sgd \
    --lr 0.01 \
    --weight_decay 1e-4 \
    --nesterov True \
    --log_file "optimizer_sgd_nesterov.log"

echo ""
echo "✅ 实验3完成"
echo ""
echo "=========================================="
echo "  所有实验完成!"
echo "=========================================="
echo "日志文件:"
echo "  - optimizer_adamw.log"
echo "  - optimizer_sgd.log"
echo "  - optimizer_sgd_nesterov.log"
echo ""
echo "建议对比:"
echo "  1. 训练loss的下降速度"
echo "  2. 最终验证loss"
echo "  3. 训练时间 (epoch耗时)"
echo ""
