import argparse
import pickle
import random
from pathlib import Path


def load_unique_tokens(path):
    """Load the pickled dataset (after preprocess_wikitext.py). """
    tokens = []
    for m in ["train", "valid", "test"]:
        with (path / f"{m}.pkl").open("rb") as f:
            data = pickle.load(f)
            tokens += [w for ex in data for w in ex[0]]
    return list(set(tokens))


def split_data(data, splits):
    """Split the tokens into charlm dataset splits."""
    total = len(data)
    split = [i / sum(splits) for i in splits]
    split = [int(total * i) for i in split] + [0.]
    split = [0] + [sum(split[:i]) for i in range(1, len(split)-1)] + [total]
    slices = [slice(split[idx], split[idx+1]) for idx in range(len(split)-1)]

    print(f"Number of tokens: {len(data)}, split into {slices}")

    random.shuffle(data)
    data_splits = [data[sl] for sl in slices]
    return data_splits


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    else:
        assert not (output_dir / "train.pkl").exists(), "Some files with same name exist. Abort."
        assert not (output_dir / "valid.pkl").exists(), "Some files with same name exist. Abort."
        assert not (output_dir / "test.pkl").exists(), "Some files with same name exist. Abort."

    unique_tokens = load_unique_tokens(data_dir)
    data_splits = split_data(unique_tokens, args.split)

    mode = ["train", "valid", "test"]
    for m, spl in zip(mode, data_splits):
        with (output_dir / f"{m}.pkl").open("wb") as f:
            # Compatible format as LMDataset.
            spl = [(list(s), ) for s in spl] 
            pickle.dump(spl, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4731)
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the original dataset.")
    parser.add_argument("--split", nargs="+", type=int, required=True, help="Ratio of splits.")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    main(args)
