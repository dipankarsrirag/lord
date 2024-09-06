import argparse
import os
from tqdm import tqdm
import pathlib

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd
from datasets import Dataset

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
dtype = torch.bfloat16

MAX_LEN = 512


def convert_embeddings_to_tensor(dataset):
    tensor_embeddings = []

    for embeddings in dataset:
        tensor_embedding = torch.stack(
            [torch.tensor(embedding) for embedding in embeddings]
        )
        tensor_embeddings.append(tensor_embedding)

    original_embedding_tensor = torch.stack(tensor_embeddings)
    return original_embedding_tensor


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


def get_prompt(data, label="original", model="mistral"):
    if model == "mistral":
        prompt = """<s> [INST] From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} [/INST]"""
    elif model == "gemma":
        prompt = """<start_of_turn>user From the conversation, Replace "[MASK]" with the most relevant word. Generate a single token, do not give explanation.\nConversation:{} <end_of_turn>"""
    else:
        raise NotImplementedError

    prompts = []
    for text in data[label]:
        prompts.append(prompt.format(text))
    return prompts


def train_model_with_supconloss(
    model,
    tokenizer,
    dataset,
    adaptor_dir,
    device,
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    patience=3,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    criterion = nn.CosineEmbeddingLoss(margin=0.25, reduction="mean")

    mask_token_id = tokenizer.mask_token_id

    best_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered. Training halted.")
            break

        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            transformed_input_ids = torch.stack(
                [example for example in batch["input_ids"]]
            ).to(device)

            transformed_attention_mask = torch.stack(
                [example for example in batch["attention_mask"]]
            ).to(device)

            original_hidden_mask = (
                convert_embeddings_to_tensor(batch["original_embedding"])
                .permute([2, 0, 1])
                .mean(dim=2)
                .to(device)
            )

            transformed_outputs = model(
                input_ids=transformed_input_ids,
                attention_mask=transformed_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            transformed_hidden_mask = (
                torch.stack(
                    [
                        hidden_state[mask_token_id, :, :]
                        for hidden_state in transformed_outputs.hidden_states
                    ],
                )
                .permute([1, 0, 2])
                .mean(dim=2)
                .to(device)
            )

            if original_hidden_mask.dtype != transformed_hidden_mask.dtype:
                transformed_hidden_mask = transformed_hidden_mask.type(
                    original_hidden_mask.dtype
                )

            labels = torch.tensor(batch["label"], dtype=torch.float).to(device)

            seq_loss = criterion(original_hidden_mask, transformed_hidden_mask, labels)

            if torch.isnan(seq_loss).any():
                print("NaNs detected in sequence loss.")
                continue

            optimizer.zero_grad()
            seq_loss.backward()

            optimizer.step()
            total_loss += seq_loss.item()

            print(f"Seq Loss: {seq_loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Average Seq Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            model.save_pretrained(adaptor_dir)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: No improvement in {patience} epochs.")
            early_stop = True

    print("Training complete.")


def get_model_tokenizer(
    model_id,
    attn_implementation,
    use_lora,
    task_type,
    lora_alpha,
    lora_r,
    lora_dropout,
    lora_bias,
):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if use_lora:
        print("Return a PEFT Model")
        modules = find_all_linear_names(model)
        print(modules)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias=lora_bias,
            target_modules=modules,
            task_type=task_type,
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("model_id", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("attn_implementation", type=str)
    parser.add_argument("use_lora", action="store_true")
    parser.add_argument("task_type", type=str)
    parser.add_argument("lora_alpha", type=float)
    parser.add_argument("lora_r", type=float)
    parser.add_argument("lora_dropout", type=float)
    parser.add_argument("lora_bias")
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("adaptor_dir", type=pathlib.Path)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    def tokenize_text(examples):
        result = tokenizer(
            examples["transformed"], padding="max_length", max_length=MAX_LEN
        )
        return result

    def get_original_embeddings(examples):
        result = tokenizer(
            examples["original"], padding="max_length", max_length=MAX_LEN
        )
        mask_token_id = tokenizer.mask_token_id
        original_embeddings = []
        batch_size = args.batch_size
        for i in range(0, len(result["input_ids"]), batch_size):
            input_ids_batch = torch.tensor(result["input_ids"][i : i + batch_size]).to(
                device
            )
            attention_mask_batch = torch.tensor(
                result["attention_mask"][i : i + batch_size]
            ).to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids_batch,
                    attention_mask_batch,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_mat = (
                torch.stack(
                    [
                        hidden_state[:, mask_token_id, :]
                        for hidden_state in outputs.hidden_states
                    ],
                )
                .permute([1, 0, 2])
                .detach()
                .cpu()
            )
            hidden_mat = hidden_mat.squeeze(1)
            torch.cuda.empty_cache()

            original_embeddings.extend(hidden_mat.numpy())

            del input_ids_batch, attention_mask_batch, outputs, hidden_mat
            torch.cuda.empty_cache()

        examples["original_embedding"] = original_embeddings
        return examples

    def get_data(path, batch_size):
        data = pd.read_json(path)
        data["original"] = get_prompt(
            data=data, label="original", model=args.model_name
        )
        data["transformed"] = get_prompt(
            data=data, label="transformed", model=args.model_name
        )

        dataset = Dataset.from_pandas(data)

        dataset = dataset.map(
            tokenize_text,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        dataset = dataset.map(
            get_original_embeddings,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=True,
            desc="Original Embeddings for Untransformed Input",
        )

        return dataset

    model, tokenizer = get_model_tokenizer(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        use_lora=args.use_lora,
        task_type=args.task_type,
        lora_alpha=args.lora_alpha,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
    )

    dataset = get_data(path=args.data_path, batch_size=args.batch_size)

    train_model_with_supconloss(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        adaptor_dir=args.adaptor_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
