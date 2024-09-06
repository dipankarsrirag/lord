from tqdm import tqdm
import pandas as pd
import pathlib
import argparse

tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("src_train_path", type=pathlib.Path)
    parser.add_argument("src_valid_path", type=pathlib.Path)

    parser.add_argument("tar_train_path", type=pathlib.Path)
    parser.add_argument("tar_valid_path", type=pathlib.Path)

    parser.add_argument("out_path", type=pathlib.Path)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    src_train = pd.read_json(args.src_train_path)
    tar_train = pd.read_json(args.tar_train_path)

    src_valid = pd.read_json(args.src_valid_path)
    tar_valid = pd.read_json(args.tar_valid_path)

    src = pd.concat([src_train, src_valid], axis="rows").reset_index(drop=True)
    tar = pd.concat([tar_train, tar_valid], axis="rows").reset_index(drop=True)

    src_words = set(src["target_word"])
    tar_words = set(tar["target_word"])

    words = src_words.intersection(tar_words)

    src = src.loc[src["target_word"].isin(words)].reset_index(drop=True)
    tar = tar.loc[tar["target_word"].isin(words)].reset_index(drop=True)

    contrastive = {"original": [], "transformed": [], "target_word": [], "label": []}

    for i in range(len(src)):
        if (
            src["transcript"].iloc[i] is None
            or len(src["transcript"].iloc[i].split("[MASK]")) > 2
        ):
            continue
        for j in range(len(tar)):
            if (
                tar["transcript"].iloc[j] is None
                or len(tar["transcript"].iloc[j].split("[MASK]")) > 2
            ):
                continue
            contrastive["original"].append(tar["transcript"].iloc[j])
            contrastive["transformed"].append(src["transcript"].iloc[i])
            contrastive["target_word"].append(src["target_word"].iloc[i])
            label = int(tar["target_word"].iloc[j] == src["target_word"].iloc[i])
            contrastive["label"].append(-1 if label == 0 else label)

    contrastive = pd.DataFrame(contrastive)

    contrastive.to_json(args.out_path, orient="records")
