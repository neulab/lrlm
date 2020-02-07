import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

import utils
from arguments import LMArguments
from repl import repl
from run import create_dataset, create_model_and_optimizer, evaluate_model, train_model

LOGGER = logging.getLogger(__name__)


def postprocess_probdump(dataset, exp_dir):
    all_logprobs = []
    for split_ in ["train", "valid", "test"]:
        with (Path(exp_dir) / f"prob_dump_{split_}.txt").open("r") as f:
            logprobs = f.read().strip().split("\n")
        all_batches = dataset.get_batches(split_, shuffle=False)
        tokens = ["".join(ex[1:-1]) for batches in all_batches for b in batches[1] for ex in b.raw_sequence]
        assert len(tokens) == len(logprobs)

        all_logprobs += list(zip(tokens, logprobs))

    with (Path(exp_dir) / "unk_probs.txt").open("w") as f:
        for ex in all_logprobs:
            print("\t".join(ex), file=f)


def run():
    args = LMArguments()

    # Seed RNGs for reproducibility
    if args.seed > 0:
        print(f"Random seed set to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    # Configure logging
    if args.save:
        logfile = utils.create_exp_dir(args.exp, args.script, overwrite=args.overwrite)
    else:
        logfile = None

    # must init logging before SummaryWriter, otherwise it adds handler to root logger so basicConfig does not work
    logging.basicConfig(
        datefmt="%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.getLevelName(args.logging_level),
        filename=logfile,
    )

    # Print out all the arguments set.
    LOGGER.info("Arguments passed: " + args.to_string(max_width=80))
    print(args.exp, flush=True)

    LOGGER.info(f"Running the model on {'GPU (CUDA)' if args.cuda else 'CPU'}")
    if args.cuda:
        props = torch.cuda.get_device_properties(0)
        LOGGER.info(f"GPU name: {props.name}, CUDA version {props.major}.{props.minor}, "
                    f"available memory: {props.total_memory / 1024 / 1024:.2f}MB.")

    # Create dataset
    dataset = create_dataset(args)

    # Create model
    model, optimizer = create_model_and_optimizer(args, dataset)

    # Print model parameter info
    n_params = sum(p.nelement() for p in model.parameters())
    LOGGER.info(f"Model parameters: {n_params}")
    LOGGER.info(f"Model structure:\n{str(model)}")

    if args.repl:
        # REPL mode
        repl(dataset, model)
        sys.exit(0)

    if args.mode == "train":
        # Training mode
        try:
            train_model(model, dataset, optimizer, args, writer=None)
        except KeyboardInterrupt:
            LOGGER.info("Training halted.")

        # load best model
        best_path, best_epoch = utils.get_best_model(args.exp)
        model.load_state_dict(torch.load(best_path)['model'])
        LOGGER.info(f"Loaded best model (epoch {best_epoch})")

        # Evaluate AND dump the avg logprobs of each word
        evaluate_model(model, dataset, args, split="train", writer=None)
        evaluate_model(model, dataset, args, split="valid", writer=None)
        evaluate_model(model, dataset, args, split="test", writer=None)

        postprocess_probdump(dataset, args.exp)

    else:
        # Evaluation mode
        for split in ['train', 'valid', 'test']:
            evaluate_model(model, dataset, args, split=split, writer=None)


if __name__ == '__main__':
    run()
