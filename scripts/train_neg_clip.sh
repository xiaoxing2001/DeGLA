
# 指定使用的GPU ID
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 多进程分布式训练设置
export MASTER_PORT=12352 # 设置一个唯一的端口号，确保不与其他程序冲突
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # 使用的GPU数量
export MASTER_ADDR=localhost  # 主节点地址设为本地

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master address: $MASTER_ADDR, Master port: $MASTER_PORT, World size: $WORLD_SIZE"

train_data='/home/face/kaichengyang/xiaoxinghu/Enhance-FineGrained/data/negCLIP.json'
# train_data='/home/face/kaichengyang/vllm/final_text_json/level_0_level1_replace_adj_new_no.json' #retrain
lr=1e-06
bs=64
cd ./src
output_name=negCLIP_40k
output_file=./Outputs/$output_name
pretrained=openai

# 检查输出目录是否存在
if [[ -d "$output_file" ]]; then
    echo "$output_name already exists"
else
    echo "Running $output_name"
    torchrun --nproc_per_node=$WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    main.py \
    --method negclip\
    --wandb-project-name neg_clip \
    --train-data $train_data \
    --seed 42 \
    --dataset-type negclip \
    --save-frequency 1 \
    --warmup 50 \
    --batch-size $bs \
    --lr $lr \
    --wd 0.1 \
    --precision amp \
    --epochs 5 \
    --workers 10 \
    --pretrained $pretrained \
    --model ViT-B-32 \
    --logs Outputs \
    --beta1 0.9 \
    --beta2 0.98 \
    --eps 1e-06 \
    --log-every-n-steps 10 \
    --name $output_name \
    --use-bn-sync \
    --csv-hard-captions-key neg_caption\
    --report-to wanbd\
    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo "Training failed. Cleaning up..."
        rm -rf $output_file
    fi
fi