import shutil
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertTokenizerFast,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    LlamaConfig,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
    StoppingCriteriaList,
    TrainingArguments,
)
# from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.nn.functional as F
from transformers.deepspeed import HfDeepSpeedConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import deepspeed
from dataclasses import dataclass, asdict
import pandas as pd
import json
import logging
import math
import os
import random
import re
import warnings
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import numpy as np
from data_utils.data_utils import *
from data_utils.ds_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim import AdamW, Adam
from typing import List, Dict
from peft import LoraConfig, get_peft_model
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import time

# from eval_metric import *
# from colossalai.nn.optimizer import HybridAdam
# from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
# from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

world_size = int(os.getenv("WORLD_SIZE", '1'))
print("Total using GPUs: ", world_size)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The input data dir. Should contain the source and target files for the task.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Path to the fine-tuned model checkpoint.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to save trained model.",
)

parser.add_argument(
    "--eval_data_path",
    type=str,
    default=None
)

parser.add_argument(
    "--mode",
    type=str,
    default="sft"
)

parser.add_argument(
    "--data_utils/deepspeed_config.json",
    type=str,
    default="deepspeed_config.json",
    help="Path to save trained model.",
)

parser.add_argument(
    "--num_train_epochs",
    default=10,
    type=int,
    help="Number of training epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1, type=int,
    help="gradient accumulation steps",
)

parser.add_argument(
    "--lr_scheduler",
    default="linear",
    type=str,
    help="The learning scheduler to be applied.",
)

parser.add_argument(
    "--warmup_ratio",
    default=0.1,
    type=float,
    help="The ratio of warmup.",
)
parser.add_argument(
    '--local_rank', 
    default=-1
)
parser.add_argument(
    '--local-rank', 
    default=-1
)
parser.add_argument(
    "--warmup_steps",
    default=None,
    type=int
)
parser.add_argument(
    "--gradient_checkpointing",
    action='store_true'
)

parser.add_argument(
    "--single_answer",
    action='store_true'
)

parser.add_argument(
    "--learning_rate",
    default=3e-5,
    type=float
)
parser.add_argument(
    "--max_seq_length",
    default=256, type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--max_length",
    default=2048, type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--stage",
    default=2, type=int,
    help="Deepspeed stage to use.",
)
parser.add_argument(
    '--weight_decay',
    default=0.0, type=float,
    help='weight decay when updating parameters.'
)

parser.add_argument(
    '--save_steps',
    default=1000, type=int,
)
parser.add_argument(
    "--zero_shot", action='store_true',
)

parser.add_argument(
    "--lora", action='store_true',
)
parser.add_argument(
    "--lora_dim", type=int, default=16,
)
parser.add_argument(
    "--lora_alpha", type=int, default=16,
)
parser.add_argument(
    "--lora_dropout", type=float, default=0.05,
)
parser.add_argument(
    "--lora_module_name", type=str, default='q_proj,k_proj,v_proj,query_key_value',
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int
)
parser.add_argument(
    "--eval_batch_size",
    default=4,
    type=int
)
parser.add_argument(
    "--top_k",
    default=None,
    type=int
)
parser.add_argument(
    "--num_return_sequences",
    default=1,
    type=int
)
parser.add_argument(
    "--num_beams",
    default=1,
    type=int
)
parser.add_argument(
    "--seed",
    default=42,
    type=int
)

parser.add_argument(
    "--top_p",
    type=float,
    default=None
)

parser.add_argument(
    "--clip_norm",
    type=float,
    default=1.0
)

parser.add_argument(
    "--temp",
    type=float,
    default=None,
    help='Temperature for model generation.'
)
parser.add_argument(
    "--do_train",
    action='store_true'
)
parser.add_argument(
    "--do_eval",
    action='store_true'
)
parser.add_argument(
    "--evaluate_every_epoch",
    action='store_true'
)
parser.add_argument(
    "--offload_optimizer",
    action='store_true'
)
parser.add_argument(
    "--save_every_epoch",
    action='store_true'
)

args = parser.parse_args()


do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None
evaluate_every_epoch = args.evaluate_every_epoch
do_final_eval = not evaluate_every_epoch
save_every_epoch = args.save_every_epoch
eval_data_path = args.eval_data_path

# eval_result_path = args.eval_result_path if args.eval_result_path is not None else args.output_dir
# os.makedirs(eval_result_path, exist_ok=True)

model_args = {
    "model_name_or_path": args.model_name_or_path,
    "checkpoint_dir": args.checkpoint_dir,
    "data_dir": args.data_dir,
    "max_seq_length": args.max_seq_length,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "save_steps": args.save_steps,
    "output_dir": args.output_dir,
    "max_length": args.max_length,
    "warmup_ratio": args.warmup_ratio,
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    'data_dir': args.data_dir,
    "lora": args.lora,
    "lora_dim": args.lora_dim,
    "lora_dropout": args.lora_dropout,
    "lora_alpha": args.lora_alpha,
    "lora_module_name": args.lora_module_name,
    "num_beams": args.num_beams,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "do_sample": do_sample,
    "seed": args.seed,
    "do_train": args.do_train,
    "do_eval": args.do_eval,
    "offload_optimizer": args.offload_optimizer,
    "deepspeed_config": args.deepspeed_config,
    "zero_shot": args.zero_shot,
    "mode": args.mode,
    "gradient_checkpointing": args.gradient_checkpointing,
    "lr_scheduler": args.lr_scheduler,
    "num_return_sequences": args.num_return_sequences,
    "stage": args.stage,
    "single_answer": args.single_answer,
}
args = ModelArgs()
args.update(model_args)

print(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_dim,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_module_name.split(","),
    bias='none',
)
with open(args.deepspeed_config, 'r', encoding='utf-8') as f:
    deepspeed_config = json.load(f)

deepspeed_config["zero_optimization"]['stage'] = args.stage
deepspeed_config["train_batch_size"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
deepspeed_config['gradient_clipping'] = args.clip_norm
if deepspeed_config["zero_optimization"]["stage"] == 3:
    deepspeed_config["zero_optimization"]['mics_shard_size'] = world_size
    if world_size > 8:
        deepspeed_config["zero_optimization"]['mics_hierarchical_params_gather'] = True

if args.offload_optimizer or args.zero_shot:
    deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"

optimizer_class = DeepSpeedCPUAdam if deepspeed_config["zero_optimization"]\
    ["offload_optimizer"]["device"] == "cpu" else AdamW


def getOptimizerGroup(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    
    return optimizer_grouped_parameters

def _get_input_dict(batch):
    input_ids, labels, attention_mask, type_token_ids = batch["input_ids"], \
        batch["labels"], batch["attention_mask"], batch["type_token_ids"]
    
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device)
        
    }
## save relevant files for loading model e.g. modeling_chatglm.py
for sub_file in os.listdir(args.model_name_or_path):
    if sub_file.endswith(".py") or sub_file.endswith(".json") or sub_file.endswith(".md"):
        file_flag = True
    else:
        file_flag = False
    if args.do_train and file_flag:
        source_file = os.path.join(args.model_name_or_path, sub_file)
        target_file = os.path.join(args.output_dir, sub_file)
        shutil.copy2(source_file, target_file)
        

## prepare model
if "chatglm" in args.model_name_or_path.lower():
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model_class = AutoModel
    # model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
    deepspeed_config["bfloat16"]["enabled"] = False
    deepspeed_config["fp16"]["enabled"] = True
elif "t5" in args.model_name_or_path.lower():
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_class = T5ForConditionalGeneration
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    deepspeed_config["bfloat16"]["enabled"] = False
    deepspeed_config["fp16"]["enabled"] = False
    args.model_type = "encoder-decoder"
    deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "none"
elif "baichuan" in args.model_name_or_path.lower():
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    model_class = AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
elif "bloom" in args.model_name_or_path.lower() or "falcon" in args.model_name_or_path.lower():
    ## for bloom, falcon
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model_class = AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
else:
    ## for llama, vicuna, belle
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    model_class = LlamaForCausalLM
    # model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    # from transformers import AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast
    # tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, padding_side="left", trust_remote_code=True, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

torch_dtype = torch.float16 if args.model_type == "decoder" else torch.float32
micro_batch_size = args.batch_size / (world_size * args.gradient_accumulation_steps)
ds_config = get_train_ds_config(
    offload=args.offload_optimizer,
    dtype="fp16",
    enable_tensorboard=False,
    train_batch_size=args.batch_size,
    micro_batch_size=micro_batch_size,
    max_out_tokens=args.max_length,
    tp_gather_partition_size=world_size
)

if args.stage == 3:
    # dschf = HfDeepSpeedConfig(deepspeed_config)
    # with deepspeed.zero.MiCS_Init(config_dict_or_path=deepspeed_config):
    #     model = model_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype)

    dschf = HfDeepSpeedConfig(ds_config)
    model = model_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype)
    if args.gradient_checkpointing:
        model.config.use_cache = False
else:
    model = model_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch_dtype)


if args.lora:
    model = get_peft_model(model, lora_config)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token

if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token

if args.model_type == "decoder":
    tokenizer.padding_side = "left"

if args.gradient_checkpointing:
    if "chatglm" in args.model_name_or_path:
        model.supports_gradient_checkpointing = True
        model.transformer.gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

if args.checkpoint_dir is not None:
    model_save_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(model_save_path):
        state_dict = torch.load(model_save_path, map_location='cpu')
        new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_k = k.replace
        model.load_state_dict(state_dict, strict=False)
        del state_dict
    elif os.path.exists(os.path.join(args.checkpoint_dir, "latest")):
        model = load_state_dict_from_zero_checkpoint(model, args.output_dir)


num_parameters = get_parameter_number(model)
with open(os.path.join(args.output_dir, "model_params.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(num_parameters, indent=5))

## prepare data
train_file = os.path.join(args.data_dir, "train.json")

dev_file = os.path.join(args.data_dir, "dev.json") if eval_data_path is None else eval_data_path
if not os.path.exists(dev_file) and args.do_eval:
    print("*****    Desire to evaluate, but dev.json file not found *****")
    args.do_eval = False
train_dataset, dev_dataset = None, None
train_collator, dev_collator = None, None
if args.do_train:
    df_train = read_data(train_file)
    train_dataset = Seq2SeqDataset(df_train)
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")
if args.do_eval:
    dev_datasets = []
    df_dev = read_data(dev_file)
    dev_dataset = Seq2SeqDataset(df_dev)
    dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")

stop_word_list = ['sep_token_id', 'eos_token_id', 'pad_token_id']
stop_ids = []
for stop_word in stop_word_list:
    id_ = getattr(tokenizer, stop_word)
    if id_ is not None:
        stop_ids.append(id_)
stop_criteria = KeywordsStoppingCriteria(stop_ids)
## prepare deepspeed model training
print(f'''**********\n{json.dumps(deepspeed_config, indent=5)}\n**********''')
if args.do_train:
    t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    args.warmup_steps = warmup_steps
    print(f'''Total steps is {t_total}\nWarmup steps is {warmup_steps}''')

    optimizer_grouped_parameters = getOptimizerGroup(model=model)
    optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    if args.lr_scheduler == "cosine":
        print("*****    Using CosineLR learning scheduler   *****")
        lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=t_total, eta_min=0)
        # lr_scheduler = scheduler_map[args.lr_scheduler](optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=warmup_steps)
    else:
        print("*****    Using linear decay learning scheduler   *****")
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=warmup_steps)
    # lr_scheduler = CosineLRScheduler(optimizer=optimizer,)
    del optimizer_grouped_parameters
    model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=deepspeed_config,
        collate_fn=train_collator,
        dist_init_required=True,
    )

    should_save = True
elif args.do_eval:
    if not args.zero_shot and not args.model_name_or_path == args.output_dir:
        model_save_path = os.path.join(args.output_dir, "pytorch_model.bin")
        if os.path.exists(model_save_path):
            state_dict = torch.load(model_save_path, map_location='cpu')
            new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     new_k = k.replace
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        elif os.path.exists(os.path.join(args.output_dir, "latest")):
            model = load_state_dict_from_zero_checkpoint(model, args.output_dir)
    dtype = torch.float16 if args.model_type == "decoder" else torch.float32
    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        replace_with_kernel_inject=True,
        dtype=dtype,
    )
    model = model_engine.module

if __name__ == "__main__":
    
    os.makedirs(os.path.join(args.output_dir, "tensorboard"), exist_ok=True)
    local_rank = torch.distributed.get_rank()
    if args.do_train:
        writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
        model.train()
        global_steps, loss_record = 0, 0
        for epoch in range(args.num_train_epochs):
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Runing epoch{epoch} / {args.num_train_epochs}",
                disable=False,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                # batch = _get_input_dict(batch)
                if "type_token_ids" in batch:
                    type_token_ids = batch.pop("type_token_ids")
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                outputs = model(**batch)
                loss = outputs.loss

                model.backward(loss)
                model.step()
                    
                current_loss = loss.item()
                loss_record += current_loss
                batch_iterator.set_description(
                    f"Epochs {epoch}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                )

                if step % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    should_save = True
                    if int(local_rank) == 0:
                        writer.add_scalar("loss", loss_record / args.gradient_accumulation_steps, global_steps)
                        loss_record = 0
                if global_steps % args.save_steps == 0 and should_save:
                    model.save_checkpoint(args.output_dir)
                    half_model_path = os.path.join(args.output_dir, "checkpoint-{}".format(global_steps))
                    os.makedirs(half_model_path, exist_ok=True)
                    model.save_16bit_model(half_model_path)
                    tokenizer.save_pretrained(half_model_path)
                    config.save_pretrained(half_model_path)
                    args.save(half_model_path)
                    should_save = False
            if save_every_epoch and epoch < args.num_train_epochs - 1:
                model.save_checkpoint(args.output_dir)
                half_model_path = os.path.join(args.output_dir, "epoch-{}".format(epoch + 1))
                os.makedirs(half_model_path, exist_ok=True)
                model.save_16bit_model(half_model_path)
                tokenizer.save_pretrained(half_model_path)
                config.save_pretrained(half_model_path)
                args.save(half_model_path)
            if args.do_eval and evaluate_every_epoch:
                model.save_checkpoint(args.output_dir)
                model.eval()
                targets = list(df_dev["output"])
                task_type = list(df_dev["task_type"])
                eval_sampler = DistributedSampler(dev_dataset, shuffle=False)
                eval_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler, collate_fn=dev_collator, num_workers=8)
                all_outputs = []

                preds_for_eval_path = os.path.join(args.output_dir, "preds_for_eval_epoch_{}.json".format(epoch))
                preds_for_eval_path_time = os.path.join(args.output_dir, "preds_for_eval_epoch_{}_tmp.json".format(epoch))
                print("\n*****    Evaluating  *****\n")
                # eval_inputs_iter = []
                f_time = open(preds_for_eval_path_time, 'a', encoding='utf-8')
                for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
                    eval_batch, eval_target, eval_task_type, eval_choices = eval_batch
                    eval_batch = eval_batch.to(model.device)
                    max_length_this_batch = eval_batch["input_ids"].size(-1) if args.model_type == "decoder" else 0
                    with torch.no_grad():
                        if "chatglm" in args.model_name_or_path:
                            outputs = model.generate(
                                **eval_batch,
                                num_beams=args.num_beams,
                                max_length=max_length_this_batch + args.max_length,
                                do_sample=args.do_sample,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                early_stopping=True,
                                num_return_sequences=args.num_return_sequences,
                            )
                        else:
                            if "token_type_ids" in eval_batch:
                                token_type_ids = eval_batch.pop("token_type_ids")
                            outputs = model.generate(
                                **eval_batch,
                                num_beams=args.num_beams,
                                top_k=args.top_k,
                                top_p=args.top_p,
                                early_stopping=True,
                                do_sample=args.do_sample,
                                max_length=max_length_this_batch + args.max_length,
                                length_penalty=2.0,
                                repetition_penalty=1.0,
                                num_return_sequences=args.num_return_sequences,
                            )
                    outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
                    all_outputs.extend(outputs)

                    outs = [tokenizer.decode(o_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o_id in outputs]
                    this_batch_inputs = [tokenizer.decode(e_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for e_id in eval_batch['input_ids']]
                    for index, this_input in enumerate(this_batch_inputs):
                        sub_o = outs[index * args.num_return_sequences: (index + 1) * args.num_return_sequences]
                        this_input = this_batch_inputs[index]
                        new_sub_o = []
                        for o in sub_o:
                            if args.model_type == "decoder":
                                if this_input in o:
                                    answer = o.replace(this_input, "").strip().rstrip()
                                else:
                                    output_ids = all_outputs[index][args.max_seq_length: ]
                                    answer = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            else:
                                answer = o
                            answer = answer.strip().rstrip()
                            new_sub_o.append(answer)
                            this_eval_instance = {
                                "input": this_input,
                                "output": new_sub_o,
                                "target": eval_target[index],
                                "task_type": eval_task_type[index],
                                "choice": eval_choices[index]
                            }
                            preds_for_eval.append(this_eval_instance)
                            f_time.writelines([json.dumps(this_eval_instance, ensure_ascii=False), '\n'])
                            f_time.flush()
                f_time.close()
                ## 方便查看预测输出的json文件
                all_preds = open(preds_for_eval_path_time, 'r', encoding='utf-8').readlines()
                all_preds = [json.loads(p) for p in all_preds]
                with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(all_preds, indent=5, ensure_ascii=False))
                model.train()
        model.save_16bit_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        args.save(args.output_dir)
        writer.close()

    if args.do_eval and do_final_eval:
        model.eval()
        targets = list(df_dev["output"])
        task_type = list(df_dev["task_type"])
        # answer_choices = list(df_dev['choice'])
        eval_sampler = DistributedSampler(dev_dataset)
        eval_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler, collate_fn=dev_collator, num_workers=8)
        all_outputs = []
        preds_for_eval = []

        preds_for_eval_path = os.path.join(args.output_dir, "preds_for_eval.json")
        preds_for_eval_path_time = os.path.join(args.output_dir, "preds_for_eval_tmp.json")
        print("\n*****    Evaluating  *****\n")
        # eval_inputs_iter = []
        f_time = open(preds_for_eval_path_time, 'a', encoding='utf-8')
        print(preds_for_eval_path_time)
        for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
            eval_batch, eval_target, eval_task_type, eval_choices = eval_batch
            eval_batch = eval_batch.to(model.device)
            # eval_inputs_iter.extend(eval_batch["input_ids"])
            max_length_this_batch = eval_batch["input_ids"].size(-1) if args.model_type == "decoder" else 0
            with torch.no_grad():
                if "chatglm" in args.model_name_or_path:
                    outputs = model.generate(
                        **eval_batch,
                        num_beams=args.num_beams,
                        max_length=max_length_this_batch + args.max_length,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        early_stopping=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                else:
                    if "token_type_ids" in eval_batch:
                        token_type_ids = eval_batch.pop("token_type_ids")
                    outputs = model.generate(
                        **eval_batch,
                        num_beams=args.num_beams,
                        do_sample=args.do_sample,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        early_stopping=True,
                        max_length=max_length_this_batch + args.max_length,
                        length_penalty=2.0,
                        repetition_penalty=1.0,
                        num_return_sequences=args.num_return_sequences,
                    )
            outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
            all_outputs.extend(outputs)

            outs = [tokenizer.decode(o_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o_id in outputs]
            this_batch_inputs = [tokenizer.decode(e_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for e_id in eval_batch['input_ids']]
            for index, this_input in enumerate(this_batch_inputs):
                sub_o = outs[index * args.num_return_sequences: (index + 1) * args.num_return_sequences]
                this_input = this_batch_inputs[index]
                new_sub_o = []
                for o in sub_o:
                    if args.model_type == "decoder":
                        if this_input in o:
                            answer = o.replace(this_input, "").strip().rstrip()
                        else:
                            output_ids = all_outputs[index][args.max_seq_length: ]
                            answer = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    else:
                        answer = o
                    answer = answer.strip().rstrip()
                    new_sub_o.append(answer)
                    this_eval_instance = {
                        "input": this_input,
                        "output": new_sub_o,
                        "target": eval_target[index],
                        "task_type": eval_task_type[index],
                        "choice": eval_choices[index]
                    }
                    preds_for_eval.append(this_eval_instance)
                    f_time.writelines([json.dumps(this_eval_instance, ensure_ascii=False), '\n'])
                    f_time.flush()
        f_time.close()
        ## 方便查看预测输出的json文件
        all_preds = open(preds_for_eval_path_time, 'r', encoding='utf-8').readlines()
        all_preds = [json.loads(p) for p in all_preds]
        with open(preds_for_eval_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(all_preds, indent=5, ensure_ascii=False))
