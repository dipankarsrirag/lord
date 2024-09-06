import pandas as pd
import pathlib
import argparse


def post_process(text):
    word = text.lower().strip()

    if "[/inst]" in word:
        word = word.split("[/inst]")[1]

    word = word.replace("/", "")
    word = word.replace("\\", "")
    word = word.replace("*", "")
    word = word.replace("?", "")
    word = word.replace("[", "")
    word = word.replace("]", "")
    word = word.replace("{", " ")
    word = word.replace("}", " ")
    word = word.replace("(", "")
    word = word.replace(")", "")
    word = word.replace(",", "")
    word = word.replace(":", "")
    word = word.replace('"', "")
    word = word.replace("'", "")
    word = word.replace(".", "")
    word = word.replace(";", " ")
    word = word.replace("`", "")
    word = word.strip()

    if "<end_of_turn>model" in word:
        word = word.split("<end_of_turn>model")[1]

    word = word.split("<end_of_turn>")[0]
    word = word.split("<s>")[0].split("</s>")[0].strip().split("\n")[0]
    word = word.replace(">", "")
    word = word.replace("<", "")
    word = word.split("\n")[0]
    word = word.split(" or ")[0]
    word = word.split(" and ")[0]
    if len(word.split()) > 3:
        word = " ".join(word.split()[:3])
    word = word.strip()

    words = [i for i in word.split(" ") if i.isalpha()]
    word = " ".join(words)

    return word


def parse_args():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("out_path", type=pathlib.Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    out = pd.read_json(args.test_path, lines=True)
    words = out["target_word"]
    words = words.apply(lambda x: post_process(x))
    gens = pd.DataFrame(
        {
            "transcript": out["transcript"],
            "target_word": words,
            "twp_ind": out["twp_ind"].apply(lambda x: post_process(x)),
            "twp_us": out["twp_us"].apply(lambda x: post_process(x)),
            "twp_trans": out["twp_trans"].apply(lambda x: post_process(x)),
            "twp_multi": out["twp_multi"].apply(lambda x: post_process(x)),
            "twp_us_ind": out["twp_us_ind"].apply(lambda x: post_process(x)),
            "twp_ai_ind": out["twp_ai_ind"].apply(lambda x: post_process(x)),
            "twp_multi_ind": out["twp_multi_ind"].apply(lambda x: post_process(x)),
            "twp_us_multi": out["twp_us_multi"].apply(lambda x: post_process(x)),
            "con_us_ind_twp_us_ind": out["con_us_ind_twp_us_ind"].apply(
                lambda x: post_process(x)
            ),
            "con_ai_ind_twp_us_ind": out["con_ai_ind_twp_us_ind"].apply(
                lambda x: post_process(x)
            ),
        }
    )

    gens.to_json(args.out_path, orient="records", lines=True)


if __name__ == "__main__":
    main()
