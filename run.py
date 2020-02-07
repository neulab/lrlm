import logging
import math
import os
import random
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch import Tensor
from tqdm import trange

import utils
from arguments import LMArguments
from dataset import BatchSequence, AliasLMDataset, Dataset, LMDataset, LRLMDataset, NKLMDataset, Relation
from models import AliasLM, LRLM, NKLM, VanillaLM
from models.base import BaseLM
from models.utils import linear_weight_init
from nnlib.utils import Logging, WeightedAverage

LOGGER = logging.getLogger(__name__)


def create_dataset(args: LMArguments):
    dataset_kwargs = dict(
        use_anchor=args.use_anchor,
        include_train=(args.mode == 'train'),
        exclude_entity_disamb=args.exclude_entity_disamb,
        exclude_alias_disamb=args.exclude_alias_disamb,
        create_batches=not args.repl,
        use_only_first_section=args.use_only_first_section,
        cache_batches=args.cache_dataset,
        fasttext_model_path=args.fasttext_model_path,
    )
    if args.use_unk_probs:
        dataset_kwargs.update(
            unk_probs_path=os.path.join(args.path, 'unk_probs.txt'),
            use_upp=args.use_upp,
        )
    dataset_class: Type[Dataset]
    if args.model == 'VanillaLM':
        dataset_class = LMDataset
    elif args.model == 'LRLM':
        dataset_class = LRLMDataset
    elif args.model == 'AliasLM':
        dataset_class = AliasLMDataset
    elif args.model == 'NKLM':
        dataset_class = NKLMDataset
        dataset_kwargs.update(unk_rels_strategy=args.unk_rels_strategy)
    else:
        raise ValueError(f"Invalid model choice '{args.model}'")
    dataset = dataset_class(args.path, args.batch_size, args.vocab_dir, args.bptt_size,
                            vocab_size=args.vocab_size, min_freq=args.min_freq, **dataset_kwargs)
    return dataset


def create_model_and_optimizer(args: LMArguments, dataset) -> Tuple[BaseLM, Optional[torch.optim.Optimizer]]:
    model: BaseLM
    if args.model == 'VanillaLM':
        model = VanillaLM(args, vocab_size=len(dataset.word_vocab))
    elif args.model == 'LRLM':
        model = LRLM(args, vocab_size=len(dataset.word_vocab), rel_vocab_size=len(dataset.rel_vocab),
                     max_unkrel=dataset.max_unkrel)
    elif args.model == 'NKLM':
        model = NKLM(args, vocab_size=len(dataset.word_vocab), rel_vocab_size=len(dataset.rel_vocab),
                     max_unkrel=dataset.max_unkrel)
    elif args.model == 'AliasLM':
        model = AliasLM(args, vocab_size=len(dataset.word_vocab))
    else:
        raise ValueError(f"Invalid model choice '{args.model}'")

    if args.multi_gpu:
        model = nn.DataParallel(model)  # type: ignore

    optimizer: Optional[torch.optim.Optimizer] = None
    if args.mode == 'train' and not args.repl:
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), args.lr)
        else:
            raise ValueError(f"Invalid optimizer setting {args.optimizer}")

    if args.repl and args.pretrained is None:
        print(f"Pretrained model path not specified, using default experiment directory.")
        args.pretrained = args.exp
    if args.pretrained is not None:
        if os.path.isdir(args.pretrained):
            path, _ = utils.get_best_model(args.pretrained)
            assert path is not None
        else:
            path = args.pretrained
        states = torch.load(path, map_location='cpu')
        incompatible_keys = model.load_state_dict(states['model'], strict=False)
        if incompatible_keys is not None and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
            print(repr(incompatible_keys))
        model.to(model.device)
        if optimizer is not None:
            optimizer.load_state_dict(states['optimizer'])
        LOGGER.info(f"Loaded model weights from {path}")
        print(f"Loaded model weights from {path}")
    else:
        def weights_init(m):
            if type(m) is nn.Linear:
                linear_weight_init(m.weight, m.bias)
            elif type(m) is nn.Embedding:
                m.weight.data.normal_(0.0, 0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].fill_(0)

        model.to(model.device)
        model.apply(weights_init)

    return model, optimizer


Callback = Callable[[Tensor, BatchSequence], None]


def compute_batch_loss(model: BaseLM, init_batch: List[List[Relation]], batches: List[BatchSequence],
                       progress_bar=None, use_unk_probs: bool = False,
                       calc_loss_kwargs: Optional[Dict[str, Any]] = None,
                       callback: Optional[Callback] = None,
                       evaluate: bool = False) -> float:
    hidden = model.init_hidden(batches[0].batch_size, init_batch)
    total_loss = 0.0

    if calc_loss_kwargs is None:
        calc_loss_kwargs = {}

    for batch in batches:
        # The loss is used to calculate PPL. Remove <EOS>
        if evaluate and batch.has_article_end:
            batch = batch.remove_last_token()
        if batch.ntokens == 0:
            continue  # this could happen when the entire batch are all <EOS> tokens
        batch = batch.to(model.device, persistent=False)
        loss, hidden = model.calc_loss(batch, hidden, use_unk_probs, **calc_loss_kwargs)
        hidden = utils.repackage_hidden(hidden)  # do it ASAP
        loss_val = loss.item()
        total_loss += loss_val * batch.ntokens

        if callback is not None:
            callback(loss, batch)
        del loss

        if progress_bar is not None:
            progress_bar.update(1)
    return total_loss


def train_model(model, dataset, optimizer, args: LMArguments, writer: Optional[SummaryWriter] = None,
                max_steps: Optional[int] = None):
    LOGGER.info("Training starts..")

    update_frequency = args.update_batch_size // args.batch_size

    best_checkpoint = None
    steps_since_checkpoint = 0
    interval = args.checkpoint_interval
    # by default, validation happens every epoch
    if args.checkpoint_interval == -1:
        interval = sum(len(b) for _, b in dataset.get_batches('train')) // update_frequency

    train_steps = 0
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000, eta_min=0.0)
    model.train()
    for eph in trange(1, args.num_epochs + 1, ncols=80, desc="Epochs", ascii=True):
        eph_loss = 0.0
        num_backwards = 0  # Track the number of gradient calc
        learn_batch_loss = WeightedAverage()  # Cumulative batch loss for one logging interval

        def grad_callback(loss: Tensor, batch: BatchSequence, *_):
            nonlocal num_backwards, steps_since_checkpoint, train_steps
            learn_batch_loss.add(loss.item(), batch.ntokens)
            scaled = loss / update_frequency
            scaled.backward()
            # gc.collect()  # forced GC to free unused parts of the graph
            num_backwards += 1

            # clip if needed
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update the parameters every update_batch_size
            if num_backwards % update_frequency == 0:
                optimizer.step()
                optimizer.zero_grad()
                steps_since_checkpoint += 1
                train_steps += 1

                if args.warm_up_steps is not None and train_steps < args.warm_up_steps:
                    lr = args.lr * train_steps / args.warm_up_steps
                    optimizer.param_groups[0]['lr'] = lr
                elif scheduler is not None:
                    scheduler.step(train_steps)

                if args.logging_level != -1 and train_steps % args.log_interval == 0:
                    LOGGER.info(f"Epoch {eph:2d}"
                                f" | Step {train_steps:5d}"
                                f" | LR {optimizer.param_groups[0]['lr']:1.5f}"
                                f" | LOSS {learn_batch_loss.value():7.4f}"
                                f" | PPL {math.exp(learn_batch_loss.value()):9.3f}")
                    learn_batch_loss.clear()

        train_batches = dataset.get_batches('train')
        progress_bar = utils.get_progress_bar(train_batches, verbose=args.progress)

        for init_batch, batches in train_batches:
            model.train()
            eph_loss += compute_batch_loss(
                model, init_batch, batches, progress_bar=progress_bar,
                use_unk_probs=args.use_unk_probs, callback=grad_callback)

            # Evaluate on validation set if needed.
            if steps_since_checkpoint >= interval:
                steps_since_checkpoint = 0

                model.eval()
                val_loss = 0.0
                valid_batches = dataset.get_batches('valid')
                valid_progress_bar = utils.get_progress_bar(valid_batches, desc="Validating", verbose=args.progress)
                with torch.no_grad():
                    for eval_init_batch, eval_batches in valid_batches:
                        val_loss += compute_batch_loss(
                            model, eval_init_batch, eval_batches, progress_bar=valid_progress_bar,
                            use_unk_probs=args.use_unk_probs, evaluate=True)
                valid_progress_bar.close()
                val_loss /= dataset.ntokens['valid']  # per char or word
                LOGGER.info("VALID |                 "
                            f" | LOSS {val_loss:3.4f} | BPC {(val_loss / math.log(2)):3.4f}"
                            f" | PPL {math.exp(val_loss):3.4f}")

                # Save current checkpoint because it's the best
                if best_checkpoint is None or val_loss < best_checkpoint.val_loss:
                    best_checkpoint = utils.Checkpoint(
                        eph, val_loss,
                        utils.cpu_state_dict(model.state_dict()),
                        utils.cpu_state_dict(optimizer.state_dict()))

                    # Also save the model
                    if args.save:
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                        torch.save(state, os.path.join(args.exp, f'model{eph}.pt'))
                        LOGGER.info(f"      | Best model achieved at eph {eph}, saved.")

                # Otherwise revert to the previous best checkpoint
                elif args.optimizer_strategy == 'reset':
                    # Reset both
                    new_lr = optimizer.param_groups[0]['lr'] * args.lr_scaler
                    model.load_state_dict(best_checkpoint.model_state)
                    optimizer.load_state_dict(best_checkpoint.optim_state)
                    optimizer.param_groups[0]['lr'] = new_lr
                    LOGGER.info(f"      | Loaded best model at epoch {best_checkpoint.epoch}.")

                if writer is not None:
                    writer.add_scalar('valid/ephloss', val_loss, eph)
                    writer.add_scalar('valid/perplexity', math.exp(val_loss), eph)

            if max_steps is not None and train_steps >= max_steps:
                progress_bar.close()
                return

        progress_bar.close()

        eph_loss /= dataset.ntokens['train']  # per token
        if writer is not None:
            writer.add_scalar('train/ephloss', eph_loss, eph)
            writer.add_scalar('train/perplexity', math.exp(eph_loss), eph)
        LOGGER.info("TRAIN"
                    f" | Eph {eph:2d} | LR {optimizer.param_groups[0]['lr']:1.5f}"
                    f" | LOSS {eph_loss:3.4f} | BPC {(eph_loss / math.log(2)):3.4f}"
                    f" | PPL {math.exp(eph_loss):3.4f}")
        if args.lr_decay > 0.0:
            optimizer.param_groups[0]['lr'] *= 1.0 - args.lr_decay


def evaluate_model(model, dataset, args: LMArguments, split, writer=None):
    LOGGER.info(f"Evaluation starts. Running model on {split} set.")

    model.eval()
    total_loss = 0.0
    data_length = 0

    callback: Optional[Callback] = None
    if args.dump_probs:
        batch_log_probs: List[List[float]] = []  # list(seq_len) of list(batch_size)
        split_log_probs: List[List[float]] = []  # list(n_examples) of list(seq_len)

        def callback(_loss: Tensor, _batch: BatchSequence):
            batch_log_probs.append(model.model_cache['log_probs'])

    with torch.no_grad():
        if split == 'train_sample':
            all_batches = dataset.get_batches('train')[:5]
        else:
            all_batches = dataset.get_batches(split, shuffle=False)
        data_length += sum(sum(b.ntokens for b in bs) for _, bs in all_batches)
        progress_bar = utils.get_progress_bar(all_batches, verbose=args.progress, desc=f"{split} batches", leave=True)
        for init_batch, batches in all_batches:
            total_loss += compute_batch_loss(
                model, init_batch, batches, progress_bar=progress_bar,
                use_unk_probs=args.use_unk_probs, calc_loss_kwargs={'dump_probs': args.dump_probs},
                callback=callback, evaluate=True)
            if args.dump_probs:
                for b in range(len(batch_log_probs[0])):
                    split_log_probs.append([log_probs[b] for log_probs in batch_log_probs])
                batch_log_probs = []
        progress_bar.close()

    if args.dump_probs:
        with open(os.path.join(args.exp, f'prob_dump_{split}.txt'), 'w') as f:
            # noinspection PyUnboundLocalVariable
            for log_probs in split_log_probs:
                f.write(' '.join(str(p) for p in log_probs) + '\n')

    # All the lengths
    total_loss /= data_length  # per char
    if writer is not None:
        writer.add_scalar(f'{split}/ephloss', total_loss, 1)
        writer.add_scalar(f'{split}/perplexity', math.exp(total_loss), 1)

    LOGGER.info(f"{split.upper()} |                 "
                f" | LOSS {total_loss:3.4f} | BPC {(total_loss / math.log(2)):3.4f}"
                f" | PPL {math.exp(total_loss):3.4f}")


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

    if args.writer:
        writer_path = os.path.join(args.tbdir, args.exp)
        if os.path.exists(writer_path):
            shutil.rmtree(writer_path)
        writer = SummaryWriter(writer_path)
    else:
        writer = None

    # Print out all the arguments set.
    LOGGER.info("Arguments passed: " + args.to_string(max_width=80))
    print(args.exp, flush=True)

    if args.profile:
        LOGGER.info(Logging.color(
            col='yellow',
            s=f"Profiling performance{' (including data loading)' if args.profile_data else ''}, "
              f"running the model for {args.profile_steps} steps..."))
        import cProfile
        profiler = cProfile.Profile()
    else:
        profiler = None

    n_gpus = torch.cuda.device_count()
    LOGGER.info("Running the model on " + (f"CUDA with {n_gpus} GPU(s)" if args.cuda else "CPU"))
    for device in range(n_gpus):
        props = torch.cuda.get_device_properties(device)
        LOGGER.info(f"GPU ({device}) name: {props.name}, CUDA version {props.major}.{props.minor}, "
                    f"available memory: {props.total_memory / 1024 / 1024:.2f}MB.")

    if args.profile and args.profile_data:
        profiler.enable()
    # Create dataset
    dataset = create_dataset(args)

    # Create model
    model, optimizer = create_model_and_optimizer(args, dataset)

    # Print model parameter info
    n_params = sum(p.nelement() for p in model.parameters())
    LOGGER.info(f"Model parameters: {n_params}")
    LOGGER.info(f"Model structure:\n{utils.repr_module(model)}")

    if args.repl:
        # REPL mode
        from repl import repl
        repl(dataset, model)
        sys.exit(0)

    if args.profile and not args.profile_data:
        profiler.enable()
    if args.mode == 'train':
        # Training mode
        try:
            train_model(model, dataset, optimizer, args, writer,
                        max_steps=args.profile_steps if args.profile else None)
        except KeyboardInterrupt:
            LOGGER.info("Training halted.")

        if not args.profile:
            # load best model
            best_path, best_epoch = utils.get_best_model(args.exp)
            if best_path is not None:
                model.load_state_dict(torch.load(best_path)['model'])
                LOGGER.info(f"Loaded best model (epoch {best_epoch})")
                evaluate_model(model, dataset, args, split='test', writer=writer)
            else:
                LOGGER.info(Logging.color('red', "No saved checkpoints, skipping evaluation"))

    else:
        # Evaluation mode
        for split in ['valid', 'test']:
            evaluate_model(model, dataset, args, split=split, writer=writer)

    if args.profile:
        import pstats
        profiler.disable()
        pstats.Stats(profiler).sort_stats('cumulative').print_stats()


if __name__ == '__main__':
    run()
