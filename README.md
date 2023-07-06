envs配置：
```
torch==2.0.0+cu117
transformers==4.28.1
tokenizers >= 0.13.1
deepspeed == 0.9.2
peft == 0.2.0
```

*****   数据格式    *****
jsonl形式， 存放路径
data/
    train.json
    dev.json
Refer to data_example.json for more details.

调用代码示例
```
MODEL_NAME='bigscience/bloom'
DATA_DIR='data/'
deepspeed --master_port=29500 main.py \
    --model_name_or_path $MODEL_NAME \
    --data_dir $DATA_DIR \
    --output_dir $YOUR_OUTPUT_DIR \
    --max_length 2048 \
    --eval_batch_size 4 \
    --num_train_epochs 5 \
    --do_train \
    --do_eval
```
#### Supported Models
The following models are tested:
- [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)
- [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
- [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

use --lora to enable lora training, --do_eval will generate a pred_for_eval.json file in output_dir with the following format:
```
[
    {
        'input': 'model_input',
        'output': 'model_eval_output'
    },
]
```
踩坑：
llama config.json 的 pad_token_id 为-1，需要手动改成 0，不然有时候会报 cuda-side trigger error

  
