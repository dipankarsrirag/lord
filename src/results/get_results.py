import pandas as pd
import argparse
import pathlib
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embeds(sentences):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def get_similarity(embed_1, embed_2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(embed_1, embed_2)[0].item()


def get_sim_score(data):
    cols = data.columns
    sim_scores = dict([(model, []) for model in cols[2:]])
    for i in tqdm(range(len(data))):
        embeds = get_embeds(list(data.iloc[i][cols[1:]]))
        for i, key in enumerate(sim_scores.keys()):
            sim_scores[key].append(get_similarity(embeds[[0]], embeds[[i + 1]]))
    return sim_scores


def get_acc_score(data):
    cols = data.columns
    acc_scores = dict([(model, []) for model in cols[2:]])
    for i in tqdm(range(len(data))):
        score = []
        for _, model in enumerate(cols[2:]):
            if data.iloc[i, 1].lower() in data.iloc[i][model].lower():
                score.append(1)
            else:
                score.append(0)
        for i, key in enumerate(acc_scores.keys()):
            acc_scores[key].append(score[i])
    return acc_scores


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("out_path", type=pathlib.Path)
    parser.add_argument("sim_path", type=pathlib.Path)
    parser.add_argument("acc_path", type=pathlib.Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    out = pd.read_json(args.out_path, lines=True)

    pd.DataFrame(get_sim_score(out)).mean().to_json(args.sim_path, orient="records")
    pd.DataFrame(get_acc_score(out)).mean().to_json(args.acc_path, orient="records")
