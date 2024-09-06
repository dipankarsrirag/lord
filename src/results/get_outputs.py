import os
import argparse
import pathlib

os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from tqdm import tqdm

import torch

import pandas as pd

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def get_prompt(model="mistral"):
    prompt = ""
    if model == "mistral":
        prompt = """<s> [INST] From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} [/INST]"""
    elif model == "gemma":
        prompt = """<start_of_turn>user From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} <end_of_turn>"""
    else:
        raise NotImplementedError
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("model_id", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("exp_name", type=str)
    parser.add_argument("twp", action="store_true")
    parser.add_argument("cont", action="store_true")
    parser.add_argument("test_path", type=pathlib.Path)
    parser.add_argument("twp_dir", type=pathlib.Path)
    parser.add_argument("con_dir", type=pathlib.Path)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    delimiter = {"mistral": "<start_of_turn>model", "gemma": "[/INST]"}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.twp:
        model = PeftModel.from_pretrained(model, args.twp_dir, adapter_name="twp")
    if args.con:
        model = PeftModel.from_pretrained(model, args.con_dir, adapter_name="con")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    test = pd.read_json(args.test_path, lines=True)

    gens = []
    template = get_prompt(model_name=args.model_name)
    for text in tqdm(list(test["transcript"]), desc="Generating"):
        messages = template.format(text)
        encodeds = tokenizer(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(
            input_ids=encodeds["input_ids"],
            attention_mask=encodeds["attention_mask"],
            max_length=encodeds["input_ids"].shape[1] + 6,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated_ids)
        word = decoded[0].split(delimiter[args.model_name], 1)[1].strip()
        gens.append(word)
        print(word)
        torch.cuda.empty_cache()
        del model_inputs
        del generated_ids
        del decoded
        del encodeds
        del messages
        del word

    test[args.exp_name] = gens
    test.to_json(args.test_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
