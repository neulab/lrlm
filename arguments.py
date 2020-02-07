from enum import auto
from typing import Optional

from nnlib.arguments import Arguments, validator
from nnlib.arguments.custom_types import Choices

__all__ = [
    'AliasDisamb',
    'RNNDropoutPos',
    'LMArguments',
]


class AliasDisamb(Arguments.Enum):
    Oracle = auto()  # use the ground truth alias
    FastText = auto()  # use FastText vectors for disambiguating between aliases


class RNNDropoutPos(Arguments.Enum):
    Early = auto()  # embedding dropout only
    Late = auto()  # LSTM output dropout only
    Both = auto()  # dropout at both positions


class LMArguments(Arguments):
    # General
    seed: int = 4731
    exp: str = None  # experiment name
    exp_suffix: Optional[str] = None  # suffix to append to the exp
    cuda = Arguments.Switch()
    mode: Choices['train', 'eval'] = 'train'
    repl = Arguments.Switch()  # enter REPL mode after loading model
    debug = Arguments.Switch()  # enter debug mode
    profile = Arguments.Switch()  # performance profiling
    profile_data = Arguments.Switch()  # also profile data loader
    profile_steps: int = 50  # number of steps to run when profiling
    multi_gpu = Arguments.Switch()  # whether to enable multi-GPU training via DataParallel

    # Logging
    tbdir: str = "./tb/"  # path to where tensorboard logs are written
    script: str = None  # specify the script you're running
    logging_level: Choices['DEBUG', 'INFO', 'WARNING', 'ERROR', 'NOTSET'] = 'INFO'
    log_interval: int = 10  # report loss every n batches
    progress = Arguments.Switch(default=True)  # if False, do not show the progress bar
    writer = Arguments.Switch(default=True)

    # Dataset
    path: str = None  # path to data
    vocab_size: Optional[int] = 50000  # set to none to prune vocabulary by frequency
    min_freq: Optional[int] = None
    vocab_dir: str = "data/vocab/"
    use_unk_probs = Arguments.Switch()  # whether to include back-off log-probs for UNK tokens
    use_upp = Arguments.Switch()  # whether to compute UPP (uniform back-off for UNKs) or not
    cache_dataset = Arguments.Switch(default=True)  # whether to cache processed batches to speedup future loads

    use_anchor = Arguments.Switch()  # whether to use anchor
    exclude_alias_disamb = Arguments.Switch()  # exclude mentions matched with aliases ambiguous for entities
    exclude_entity_disamb = Arguments.Switch()  # exclude mentions matched with entities ambiguous for relation types
    use_only_first_section = Arguments.Switch()  # use only the first section (summary part) of the articles

    # Model
    base_rnn: Choices['lstm', 'transformer'] = 'lstm'
    model: Choices['VanillaLM', 'LRLM', 'NKLM', 'AliasLM'] = 'VanillaLM'
    embed_size: int = 512  # input word embedding size
    hidden_size: int = 1024  # RNN's hidden unit size
    num_layers: int = 4  # number of RNN layers
    dropout: float = 0.5  # inter-layer dropout rate
    bptt_size: int = 140  # BPTT length
    vocab_mlp_hidden_dim: int = -1  # vocab MLP hidden layer size
    vocab_mlp_activation: Choices['relu', 'tanh', 'sigmoid', 'none'] = 'relu'
    vocab_mlp_dropout: float = None  # vocab MLP dropout, `None` for same value as 'dropout'
    use_rel_mlp: bool = False
    rnn_dropout_pos: RNNDropoutPos = 'late'  # LSTM dropout position
    adaptive_embed = Arguments.Switch()  # use adaptive input embeddings & softmax
    tie_embed_weights = Arguments.Switch()  # tie input embeddings & softmax weights
    fasttext_model_path: Optional[str] = None
    normalize_fasttext_embeds = Arguments.Switch()

    # Transformer-specific
    num_heads: int = 10  # number of heads in multi-head attention
    head_dim: int = 41  # per-head attention dimension
    ffn_inner_dim: int = 2100  # fully-connected layer dimension
    attention_dropout: float = 0.0
    memory_size: int = 150  # maximum length of cached memory
    pre_lnorm = Arguments.Switch()  # whether to apply layer norm before residual

    # NKLM-specific
    pos_embed_dim: int = 40  # position embedding size
    pos_embed_count: int = 20  # number of position embeddings
    kb_embed_dim: int = 50  # TransE embedding size
    fact_key_mlp_hidden_dim: int = -1  # `fact_key` MLP hidden layer size, -1 for paper value
    copy_mlp_hidden_dim: int = -1  # copy MLP hidden layer size, -1 for paper value
    pos_mlp_hidden_dim: int = -1  # pos MLP hidden layer size
    mask_invalid_pos = Arguments.Switch()  # mask invalid copy positions during training

    # LRLM-specific
    rel_mlp_hidden_dim: int = -1  # rel MLP hidden layer size
    rel_mlp_activation: Choices['relu', 'tanh', 'sigmoid', 'none'] = 'relu'
    rel_mlp_dropout: float = None  # relation MLP dropout, `None` for same value as 'dropout'
    use_knowledge_embed = Arguments.Switch()  # use knowledge embeddings in selector & relation predictors
    train_relation_vec: bool = Arguments.Switch()  # if True, relation embeddings are fine-tuned

    # Strategies
    unk_rels_strategy: Choices[
        'remove',  # remove such relations
        'unk',  # treat them as the [UNK] special relation
        'params',  # learn separate params for each relation
    ] = 'params'  # strategy for UNK relations
    fact_sel_strategy: Choices[
        'gold',  # use the gold fact from data
        'argmax',  # use the fact with highest score
    ] = 'gold'  # strategy for fact selection (which fact to use for wrod/pos prediction) during training
    alias_disamb_strategy: AliasDisamb = AliasDisamb.Oracle  # strategy for selecting surface form given entity

    # Optimization
    num_epochs: int = 20
    batch_size: int = 30
    update_batch_size: int = 60
    lr: float = 1e-3
    lr_scaler: float = 0.5
    lr_decay: float = 0.0
    clip: Optional[float] = None
    optimizer: Choices['adam', 'sgd'] = 'adam'
    lr_scheduler: Choices['cosine', 'none'] = 'none'
    optimizer_strategy: Choices[
        'none',  # do nothing
        'reset',  # when validation results degrade, reset model & optimizer state to previous best state
    ] = 'reset'
    warm_up_steps: Optional[int] = None  # no. of steps to linearly increase lr from 0 to `lr`, useful for Transformers

    # Load/Save
    save = Arguments.Switch(default=True)
    pretrained: Optional[str] = None  # path to pretrained model
    checkpoint_interval: int = -1  # number of iterations to run before validating, -1 for one epoch
    overwrite = Arguments.Switch()  # overwrite the experiment log directory if it exists

    dump_probs = Arguments.Switch()  # flag to dump log-prob for each example

    def preprocess(self):
        if self.vocab_mlp_dropout is None:
            self.vocab_mlp_dropout = self.dropout
        if self.rel_mlp_dropout is None:
            self.rel_mlp_dropout = self.dropout
        self.use_knowledge_embed.set_default(self.model == 'NKLM')
        if self.vocab_mlp_hidden_dim == -1:
            self.vocab_mlp_hidden_dim = self.hidden_size

    def validate(self):
        if self.update_batch_size % self.batch_size != 0:
            raise ValueError("'update-batch-size' must be a multiple of 'batch-size'")

        if self.vocab_size is None and self.min_freq is None:
            raise ValueError("'vocab-size' or 'min-freq' cannot both be 'None'")

        if self.tie_embed_weights and self.embed_size != self.vocab_mlp_hidden_dim:
            raise ValueError("When tying weights embeddings and softmax, "
                             "vocabulary MLP hidden dim (defaults to hidden size) must be equal to embedding dim.")

        if self.alias_disamb_strategy is AliasDisamb.FastText and self.fasttext_model_path is None:
            raise ValueError("'fasttext_model_path' must not be 'None' when 'alias_disamb_strategy' is 'FastText'.")

        return [
            (r'path', validator.is_path()),
            (r'fasttext_model_path', validator.is_path(nullable=True)),
            (r'.*dropout', validator.is_dropout()),
            (r'.*_mlp_hidden_dim', lambda x: x == -1 or x > 0),
        ]

    def postprocess(self):
        if self.repl:
            self.save = False
            self.logging_level = 'WARNING'

        if self.debug:
            self.save = False
            self.writer = False
            self.progress = False

        # If it's just an evaluation, prefix the exp name
        if self.mode != 'train':
            self.exp = 'EVAL-' + self.exp

        if self.exp_suffix is not None:
            self.exp += '-' + self.exp_suffix
