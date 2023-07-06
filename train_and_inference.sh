YOUR_DATA_DIR=''
YOUR_OUTPUT_DIR=''
MODEL_NAME="bigscience/bloom-7b"
deepspeed --master_port=29500 main.py \
    --model_name_or_path $MODEL_NAME \
    --data_dir $YOUR_DATA_DIR \
    --output_dir $YOUR_OUTPUT_DIR \
    --max_length 2048 \
    --batch_size 64 \
    --deepspeed_config ./data_utils/deepspeed_config.json \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 4 \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --lora \
    --learning_rate 2e-4 \
    --do_train \
    --do_eval
