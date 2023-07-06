# MODEL_NAME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/models/openqa/PLM/560m/
source activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/project_conda
source /opt/rh/devtoolset-8/enable

MODEL_NAME='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/models/openqa/PLM/llama-7b-hf'
deepspeed --master_port=29500 main.py \
    --model_name_or_path $MODEL_NAME \
    --data_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/data/math/prm800k_sft/sub \
    --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/models_llm/math_debug/llama-7b \
    --max_length 1024 \
    --batch_size 64 \
    --deepspeed_config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangkeheng/llm_project/llm_sft/data_utils/deepspeed_config.json \
    --gradient_accumulation_steps 32 \
    --eval_batch_size 2 \
    --num_train_epochs 1 \
    --save_steps 100000 \
    --lora \
    --learning_rate 2e-4 \
    --do_train \
    --do_eval
