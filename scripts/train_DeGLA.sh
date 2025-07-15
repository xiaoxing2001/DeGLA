
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12345
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export MASTER_ADDR=localhost

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master address: $MASTER_ADDR, Master port: $MASTER_PORT, World size: $WORLD_SIZE"

train_data=data/DeGLA.json
lr=1e-06
bs=32
cd ./src
output_name=DeGLA
output_file=./Outputs/$output_name
pretrained=openai

if [[ -d "$output_file" ]]; then
    echo "$output_name already exists"/
else
    echo "Running $output_name"
    torchrun --nproc_per_node=$WORLD_SIZE --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    main.py \
    --ema_teacher \
    --ema_alpha 0.9996\
    --wandb-project-name DeGLA \
    --text_local_contrast \
    --text_local_weight 0.1\
    --image_local_contrast \
    --image_local_weight 0.1\
    --distill \
    --distill_weight 0.005 \
    --neg_text 4\
    --distill_mse \
    --teacher ViT-B-32 \
    --train-data $train_data \
    --seed 42 \
    --dataset-type DeGLA \
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
    if [ $? -ne 0 ]; then
        echo "Training failed. Cleaning up..."
        rm -rf $output_file
    fi
fi


