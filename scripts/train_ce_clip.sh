
# 指定使用的GPU ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 多进程分布式训练设置
export MASTER_PORT=12345  # 设置一个唯一的端口号，确保不与其他程序冲突
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # 使用的GPU数量
export MASTER_ADDR=localhost  # 主节点地址设为本地

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master address: $MASTER_ADDR, Master port: $MASTER_PORT, World size: $WORLD_SIZE"


train_data='../data/generated_data/coco_train/'
upper_bound=10
threshold_type=mean # 设置为固定或均值阈值类型
fixed_threshold_value=10
lr=1e-06
bs=32
cd ./src
cmr_weight=0.4
imc_weight=0.2
if [ "$threshold_type" == "fixed" ]; then
    output_name=coco_hn_imc${imc_weight}-cmr${cmr_weight}-fixed${fixed_threshold_value}-${lr}
else
    output_name=coco_hn_imc${imc_weight}-cmr${cmr_weight}-mean-ub${upper_bound}-${lr}_${bs}
fi
output_file=./Outputs/$output_name

# 检查输出目录是否存在
if [[ -d "$output_file" ]]; then
    echo "$output_name already exists"
else
    echo "Running $output_name"
    torchrun --nproc_per_node=$WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    -- main.py \
    --method ceclip \
    --wandb-project-name open_clip \
    --train-data $train_data \
    --seed 42 \
    --dataset-type npy \
    --save-frequency 1 \
    --warmup 50 \
    --batch-size $bs \
    --lr $lr \
    --wd 0.1 \
    --precision amp \
    --epochs 5 \
    --workers 10 \
    --pretrained openai \
    --model ViT-B-32 \
    --logs Outputs \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-06 \
    --log-every-n-steps 10 \
    --hardnegative \
    --threshold-type $threshold_type \
    --fixed-threshold-value $fixed_threshold_value \
    --upper-bound $upper_bound \
    --name $output_name \
    --use-bn-sync \
    --imc-loss-weight $imc_weight \
    --imc-loss \
    --cmr-loss-weight $cmr_weight \
    --cmr-loss \

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo "Training failed. Cleaning up..."
        rm -rf $output_file
    fi
fi