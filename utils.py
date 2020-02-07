import gc
import gzip
import logging
import os
import pickle
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union, overload

import torch
from torch import nn
from torch import Tensor

import nnlib
from nnlib.utils import Logging

__all__ = [
    'Checkpoint',
    'cpu_state_dict',
    'repackage_hidden',
    'loadpkl',
    'savepkl',
    'get_best_model',
    'get_progress_bar',
    'create_exp_dir',
    'repr_module',
]

LOGGER = logging.getLogger(__name__)


class Checkpoint(NamedTuple):
    epoch: int
    val_loss: float
    model_state: OrderedDict
    optim_state: dict


def cpu_state_dict(state_dict):
    res = {}
    for k, v in state_dict.items():
        if isinstance(v, Tensor):
            res[k] = v.to('cpu')
        elif isinstance(v, dict):
            res[k] = cpu_state_dict(v)
        else:
            res[k] = v
    return res


@overload
def repackage_hidden(h: Tensor) -> Tensor: ...


@overload
def repackage_hidden(h: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]: ...


# From AWD-LSTM
def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def loadpkl(file_name: Union[Path, str], gz: bool = False):
    """A wrapper for loading function with pickle.
    Parameters:
        file_name (str): File path to load a pickle file from.
        gz (bool): Whether to open file with gz or not.
    Returns:
        An object loaded from the file_name.
    """

    if gz:
        with gzip.open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_name, 'rb') as f:
            return pickle.load(f)


def savepkl(obj, file_name: Union[Path, str], gz: bool = False):
    """A wrapper for saving function with pickle.
    Parameters:
        file_name (str): File path to save a pickle file to.
        obj: The object to save.
        gz (bool): Whether to save file with gz or not. Automatically append
            ".gz" to the provided filename if True.
    Returns:
        None
    """
    if gz:
        with gzip.open(f'{file_name}.gz', 'wb') as f:
            pickle.dump(obj, f, protocol=4)
    else:
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
    return


# https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
def mem_report():
    """Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported"""

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation """
        print("Storage on %s" % (mem_type))
        print("-" * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print("%s\t\t%s\t\t%.2f" % (element_type, size, mem))
        print("-" * LEN)
        print("Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (total_numel, total_mem))
        print("-" * LEN)

    LEN = 65
    print("=" * LEN)
    objects = gc.get_objects()
    print("%s\t%s\t\t\t%s" % ("Element type", "Size", "Used MEM(MBytes)"))
    tensors = [obj for obj in objects if isinstance(obj, Tensor)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("=" * LEN)


def get_best_model(path: str) -> Tuple[Optional[str], int]:
    try:
        best_epoch = max((int(p[5:-3]) for p in os.listdir(path)
                          if p.startswith('model') and p.endswith('.pt')), default=-1)
        best_model_path = os.path.join(path, f'model{best_epoch}.pt') if best_epoch != -1 else None
        return best_model_path, best_epoch
    except FileNotFoundError:
        return None, -1


def get_progress_bar(batches, verbose=True, desc=None, leave=False):
    return nnlib.utils.progress(verbose=verbose, ncols=80, desc=desc or "Batches",
                                ascii=True, total=sum(len(b) for _, b in batches), leave=leave)


def create_exp_dir(path: str, script_path: str, overwrite=False) -> str:
    """
    Create experiment directory, and return path to log file.
    """
    if os.path.exists(path):
        if not overwrite:
            print(Logging.color(col='red', s=f"The experiment name: {path} already exists. "
                                             f"Training will not proceed without the `--overwrite` flag."))
            sys.exit(1)
        else:
            # Radical
            print(Logging.color(col='green', s=f"Overwriting the experiment: {path} ..."))
            shutil.rmtree(path)

    os.mkdir(path)
    shutil.copy(script_path, path)
    logfile = os.path.join(path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    return logfile


def _add_indent(s_, n_spaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(n_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _convert_id(keys: List[str]) -> Iterator[str]:
    start = end = None
    for key in keys:
        if key.isnumeric() and end == int(key) - 1:
            end = int(key)
        else:
            if start is not None:
                if start == end:
                    yield f"id {start}"
                else:
                    yield f"ids {start}-{end}"
            if key.isnumeric():
                start = end = int(key)
            else:
                start = end = None
                yield key
    if start is not None:
        if start == end:
            yield f"id {start}"
        else:
            yield f"ids {start}-{end}"


def _compress_reprs(reprs: List[Tuple[str, str]]) -> List[str]:
    lines = []
    prev_mod_str = None
    keys: List[str] = []
    for key, mod_str in reprs:
        if prev_mod_str is None or prev_mod_str != mod_str:
            if prev_mod_str is not None:
                for name in _convert_id(keys):
                    lines.append(f"({name}): {prev_mod_str}")
            prev_mod_str = mod_str
            keys = [key]
        else:
            keys.append(key)
    if len(keys) > 0:
        for name in _convert_id(keys):
            lines.append(f"({name}): {prev_mod_str}")
    return lines


def repr_module(module: nn.Module) -> str:
    r"""Create a compressed representation by combining identical modules in
    `nn.ModuleList`s and `nn.ParameterList`s.
    """
    # We treat the extra repr like the sub-module, one item per line
    extra_lines: List[str] = []
    extra_repr = module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_reprs = [(key, _add_indent(repr_module(submodule), 2))
                   for key, submodule in module.named_children()]
    child_lines = _compress_reprs(child_reprs)
    lines = extra_lines + child_lines

    main_str = module.__class__.__name__ + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


def _ParameterList_extra_repr(self):
    param_reprs = []
    for key, param in self._parameters.items():
        if param is not None:
            size_str = 'x'.join(str(size) for size in param.size())
            device_str = '' if not param.is_cuda else ' (GPU {})'.format(param.get_device())
            para_str = f'Parameter containing: [{torch.typename(param.data)} of size {size_str}{device_str}]'
        else:
            para_str = 'None'
        param_reprs.append((key, para_str))
    child_lines = _compress_reprs(param_reprs)
    main_str = '\n'.join(child_lines)
    return main_str


# monkey-patch the output for ParameterList
nn.ParameterList.extra_repr = _ParameterList_extra_repr  # type: ignore
