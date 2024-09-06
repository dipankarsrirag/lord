import argparse
import pathlib
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
)
import warnings
import torch

warnings.filterwarnings("ignore")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and save the model locally")
    parser.add_argument("model_id", type=str)
    parser.add_argument("attn_implementation", type=str)
    parser.add_argument("task_type", type=str)
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("adaptor_dir", type=pathlib.Path)
    parser.add_argument("cache_dir", type=pathlib.Path)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if args.task_type == "CAUSAL_LM":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            quantization_config=bnb_config if device == "cuda" else None,
        )
    else:
        model = AutoModel.from_pretrained(
            args.model_id,
            device_map="auto",
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            quantization_config=bnb_config if device == "cuda" else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["[MASK]"])
    tokenizer.mask_token = "[MASK]"

    model.resize_token_embeddings(len(tokenizer))

    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)

    print(model)
    print(tokenizer)


if __name__ == "__main__":
    main()
