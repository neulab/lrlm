import argparse
import pickle
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")
from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--fasttext-path", type=Path, help="Path to compiled fastText binary")
parser.add_argument("--dataset-dir", type=Path, help="Path to generated dataset")
parser.add_argument("--use-only-first-section", default=False, action='store_true')
parser.add_argument("--output-dir", type=Path, help="Path to save fastText embeddings")
args = parser.parse_args()


def run_fasttext_training(fasttext_path: Path, input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    FASTTEXT_ARGS = {
        "input": input_path.absolute(),
        "output": output_path.absolute(),
        "lr": 0.025,
        "dim": 300,
        "ws": 5,
        "epoch": 3,
        "minCount": 5,
        "neg": 5,
        "loss": "ns",
        "bucket": 2000000,
        "minn": 3,
        "maxn": 6,
        "thread": 4,
        "t": 1e-4,
        "lrUpdateRate": 100,
    }

    args = [str(fasttext_path.absolute()), "skipgram"]
    for k, v in FASTTEXT_ARGS.items():
        args.extend([f"-{k}", str(v)])
    print(args)
    subprocess.run(args, check=True)


def dump_training_data(dataset_dir: Path, use_only_first_section=False) -> Path:
    with (dataset_dir / "train.pkl").open("rb") as f:
        train_data = pickle.load(f)
    file_name = "train.txt" if not use_only_first_section else "train-short.txt"
    data_path = dataset_dir / file_name
    with data_path.open("w") as f:
        for ex in tqdm(train_data, desc="Dumping training data"):
            sentence = ex[0]
            if use_only_first_section:
                sentence = sentence[:Dataset.find_first_section(sentence)]
            f.write(' '.join(sentence) + '\n')
    return data_path


def main():
    input_path = dump_training_data(args.dataset_dir, args.use_only_first_section)
    run_fasttext_training(args.fasttext_path, input_path, args.output_dir)


if __name__ == '__main__':
    main()
