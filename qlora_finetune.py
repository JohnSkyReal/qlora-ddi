import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import argparse
import warnings


'''
QLORA with int4

validated supported LLMs (Nvidia T4 16GB, CPU-RAM 12.7GB, max_len=256, mbs=8):
llama: huggyllama/llama-7b (, 7.2GB for llama-13b)
llama2: meta-llama/Llama-2-7b-hf (4.1GB VRAM, 7.2GB for llama2-13b)
bloom: bigscience/bloom-7b1 (loading RAM OOM, GPU OOM even bloom-3b)
    sharded weight: mrm8488/bloom-7b1-sharded-bf16
bloomz: bigscience/bloomz-7b1 (loading RAM OOM, GPU OOM even bloomz-3b)
    sharded weight: mrm8488/bloomz-7b1-sharded-bf16
opt: facebook/opt-6.7b (loading RAM OOM, 6.9GB for opt-13b)
gpt-neox: EleutherAI/gpt-neox-20b (11.8GB VRAM, training GPU OOM)
gpt-j: EleutherAI/gpt-j-6b


tokenizer
bloom and bloomz
pad: 3 <pad>
unk: 0 <unk>
bos: 1 <s>
eos: 2 </s>
default padding side: left

gpt-neox-20b
pad: None None
unk: 0 <|endoftext|>
bos: 0 <|endoftext|>
eos: 0 <|endoftext|>
default padding side: right

opt
pad: 1 <pad>
unk: 2 </s>
bos: 2 </s>
eos: 2 </s>
default padding side: right
'''


parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False) # 如果想激活wandb，就传入--wandb即可，不用添加数值
parser.add_argument("--use_auth", action="store_true", default=False) # Enables using Huggingface auth token from Git Credentials.
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--test_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="qlora-output")
parser.add_argument("--model_path", type=str, default="huggyllama/llama-7b")
parser.add_argument("--token_path", type=str, default=None)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--micro_batch_size", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=512)
# parser.add_argument("--warmup_steps", type=int, default=20)  # 换用下面 warmup_ratio
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--logging_steps", type=int, default=1)
parser.add_argument("--eval_steps", type=int, default=10)
parser.add_argument("--save_steps", type=int, default=80)  # 暂停使用，已经改为按epoch存储
parser.add_argument("--save_total_limit", type=int, default=30)
parser.add_argument("--do_test", type=int, default=0)   # 0: 不用测试集，1：用测试集
parser.add_argument("--resume_from_checkpoint", type=str, default=None)  # 暂停使用
parser.add_argument("--ignore_data_skip", type=str, default="False")  # False为从断点处数据继续训练，True为从头开始  # 暂停使用
parser.add_argument("--cache_dir", type=str, default=None) # 123 server: ../ptms/llama-13b
args = parser.parse_args()

print(args)

if not args.wandb:
    os.environ["WANDB_MODE"] = "disabled"

# Setting for A100 - For 3090
MICRO_BATCH_SIZE = args.micro_batch_size  # change to 4 for 3090
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs  # paper uses 3
LEARNING_RATE = args.learning_rate  # from the original paper
CUTOFF_LEN = args.max_length  # 256 accounts for about 96% of the data
LORA_R = 64
LORA_ALPHA = 16  # 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.do_test #2000

DATA_PATH = args.data_path
TEST_DATA_PATH = args.test_path
OUTPUT_DIR = args.output_path

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

print(args.model_path)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map=device_map,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    use_auth_token=args.use_auth if 'Llama-2' not in args.model_path else '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',
    cache_dir=args.cache_dir,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path if not args.token_path else args.token_path,
    add_eos_token=True,
    trust_remote_code=True,
    use_auth_token=args.use_auth if 'Llama-2' not in args.model_path else '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',
    cache_dir=args.cache_dir,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)

tokenizer.padding_side = "left"

if 'Llama' in args.model_path or 'llama' in args.model_path:
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
elif 'gpt-neo' in args.model_path:
    tokenizer.pad_token_id = 1
elif 'gpt-j' in args.model_path:
    tokenizer.pad_token_id = 50257


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


if 'opt' in args.model_path:
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

else:
    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
# 找到所有需要插入adapter的全连接层
target_modules = find_all_linear_names(model)
print('target_modules: ', target_modules)
# 初始化lora配置
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.torch_dtype = torch.float32


if args.resume_from_checkpoint:
# Check the available weights and load them
    checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint 'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = None  # So the trainer won't try loading its state
            warnings.warn(f"Checkpoint {checkpoint_name} not found, now training from scratch!")
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):  # 找到有该checkpoint目录
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        model = set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
    print("checkpoint_name", checkpoint_name)


data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    if "instruction" in data_point.keys():
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
        else:
            return f"""### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    else:
        return data_point["prompt"] + data_point["completion"]


def add_bos_eos_tok(text):
    if 'Llama' in args.model_path or 'llama' in args.model_path:
        return text
    elif 'gpt-neo' in args.model_path or 'bloom' in args.model_path or 'gpt-j' in args.model_path:
        return tokenizer.bos_token + text + tokenizer.eos_token
    elif 'opt' in args.model_path:
        return text + tokenizer.eos_token


data = data.shuffle().map(
    lambda data_point: tokenizer(
        add_bos_eos_tok(generate_prompt(data_point)),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)

if VAL_SET_SIZE > 0:
    test_data = load_dataset("json", data_files=TEST_DATA_PATH)
    test_data = test_data.shuffle().map(
        lambda data_point: tokenizer(
            add_bos_eos_tok(generate_prompt(data_point)),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )
else:
    test_data = None

print("train data length: {}".format(len(data['train'])))
print("dev/test data length: {}".format(len(test_data['train']) if test_data else 0))
print(data)
print(test_data)
print(tokenizer.decode(data['train'][0]["input_ids"]))
print(tokenizer.decode(test_data['train'][0]["input_ids"]) if test_data else None)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=test_data["train"] if test_data else None,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        # save_strategy="steps",
        save_strategy="epoch",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        # save_steps=args.save_steps,
        optim="paged_adamw_32bit",
        output_dir=OUTPUT_DIR,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

# old_state_dict = model.state_dict
# model.state_dict = (
#     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
# ).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)