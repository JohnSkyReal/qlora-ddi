import os
import json
import time
import torch
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoConfig, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--use_auth", action="store_true", default=False) # Enables using Huggingface auth token from Git Credentials.
parser.add_argument("--data_path", type=str, default="llama_test_filter_re_rp.jsonl")
parser.add_argument("--output_path", type=str, default="DDI_prediction_llama-13b-filter-re-rp-len512-cp2000")
parser.add_argument("--model_path", type=str, default="huggyllama/llama-7b")
parser.add_argument("--token_path", type=str, default=None)
parser.add_argument("--lora_path", type=str, default="llama-13b-filter-re-rp-len512-cp2000-ep10")
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--cache_dir", type=str, default=None) # 123 server: ../ptms/llama-13b
args = parser.parse_args()

device_map = "auto"
# if we are in a distributed setting, we need to set the device map and max memory per device
if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

peft_model_id = args.lora_path
config = PeftConfig.from_pretrained(peft_model_id)

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


# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = PeftModel.from_pretrained(model, peft_model_id)


# dump:写入json文件
def dump_json(path, data):
    with open(path, 'w', encoding='utf-8')as f:
        json.dump(data, f)


def batch_completion(all_pairs, save_path):
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.2,
        top_k=40,
        do_sample=False,
        num_beams=4,

    )
    
#     generation_config = GenerationConfig(
#         temperature=1,
#         top_p=1,
#         repetition_penalty=1,
#         top_k=50,
#         do_sample=False,
#         num_beams=1,

#     )

    batch_size = args.test_batch_size
    input_texts = []
    for pair in all_pairs:
        input_text = f"""
### Instruction:
{pair["instruction"]}

### Input:
{pair["input"]}

### Response:"""
        input_texts.append(input_text)
    print('----example----\n', input_texts[0])
    # Split input_texts into smaller batches
    input_text_batches = [input_texts[i:i + batch_size] for i in range(0, len(input_texts), batch_size)]

    i = 0
    for input_text_batch in input_text_batches:
        inputs = tokenizer(
            input_text_batch,
            padding=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].cuda()
        print(input_ids)
        # print(tokenizer.decode(input_ids))

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1024,
        )

        for s in generation_output.sequences:
            print(i + 1)
            response = tokenizer.decode(s, skip_special_tokens=True)
            print(response)
            print('-' * 20)
            all_pairs[i]['response'] = response
            json_path = save_path + r'/response_{}.json'.format(i + 1)
            dump_json(json_path, all_pairs[i])
            i += 1


def auto_completion(data_point):
    if "instruction" in data_point.keys():
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            input_text = f"""### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:"""
        else:
            input_text = f"""### Instruction:
{data_point["instruction"]}

### Response:"""
    else:
        input_text = data_point["prompt"]

    if 'gpt-neo' in args.model_path or 'bloom' in args.model_path or 'gpt-j' in args.model_path:
        input_text =  tokenizer.bos_token + input_text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    )

    if 'Llama' in args.model_path or 'llama' in args.model_path or 'bloom' in args.model_path:
        input_ids = inputs["input_ids"].cuda()

        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            repetition_penalty=1.2,
            top_k=40,
            do_sample=False,
            num_beams=4,

        )

        print("Generating...")

        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=args.max_new_tokens
        )
        for s in generation_output.sequences:
            response = tokenizer.decode(s, skip_special_tokens=True)  # 新增参数，来自opt
    else:
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(response)
    print('-' * 20)

    return response


def extract_prompt(lines):
    all_pairs = []
    for line in lines:
        pair = {
            'instruction': line["instruction"],
            'input': line["input"],
            'label': line["output"],
            'response': ''
        }
        all_pairs.append(pair)
    return all_pairs


def main():
    save_path = args.output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = args.data_path
    # lines = readxls(path)
    lines = load_dataset("json", data_files=path)
    all_pairs = lines["train"]
    print(len(all_pairs))
    # batch_completion(all_pairs, save_path)
    i = 1
    for pair in all_pairs:
        print(i)
        pair = dict(pair)
        response = auto_completion(pair)
        pair['response'] = response
        json_path = save_path + r'/response_{}.json'.format(i)
        dump_json(json_path, pair)
        i += 1
        # time.sleep(2)


if __name__ == '__main__':
    main()
