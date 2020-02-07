"""
Miscellaneous Functions
"""
import functools
import io
import warnings
from typing import Dict, Iterable, List, TypeVar, Union, overload

__all__ = ['progress', 'deprecated', 'map_to_list', 'memoization', 'reverse_map']

T = TypeVar('T')


@overload
def progress(iterable: Iterable[T], verbose=True, **kwargs) -> Iterable[T]:
    ...


@overload
def progress(iterable: int, verbose=True, **kwargs) -> Iterable[int]:
    ...


try:
    # noinspection PyUnresolvedReferences
    import tqdm


    @overload
    def progress(iterable=None, verbose=True, **kwargs) -> tqdm.tqdm:
        ...


    class _DummyTqdm:
        """ Ignores everything """

        @staticmethod
        def nop(*args, **kwargs):
            pass

        def __getattr__(self, item):
            return _DummyTqdm.nop


    def progress(iterable=None, verbose=True, **kwargs):
        if not verbose:
            if iterable is None:
                return _DummyTqdm()
            if isinstance(iterable, int):
                return range(iterable)
            return iterable
        # if 'bar_format' not in kwargs:
        #     kwargs['bar_format'] = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} ~{remaining}]'
        if isinstance(iterable, int):
            return tqdm.trange(iterable, **kwargs)
        if iterable is None:
            return tqdm.tqdm(**kwargs)
        if isinstance(iterable, io.IOBase):
            from nnlib.utils.io import FileProgressWrapper  # avoid circular import
            return FileProgressWrapper(iterable, verbose=verbose, **kwargs)
        return tqdm.tqdm(iterable, **kwargs)

except ImportError:
    # noinspection PyUnusedLocal
    def progress(iterable=None, verbose=True, **_kwargs):
        if not verbose:
            if isinstance(iterable, int):
                return range(iterable)
            return iterable
        warnings.warn("`tqdm` package is not installed, no progress bar is shown.", category=ImportWarning)
        if isinstance(iterable, int):
            return range(iterable)
        return iterable


def deprecated(new_func=None):
    def decorator(func):
        warn_msg = f"{func.__name__} is deprecated."
        if new_func is not None:
            warn_msg += f" Use {new_func.__name__} instead."

        def wrapped(*args, **kwargs):
            warnings.warn(warn_msg, category=DeprecationWarning)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def map_to_list(d: Dict[int, T]) -> List[T]:
    """
    Given a dict mapping indices (continuous indices starting from 0) to values, convert it into a list.

    :type d: dict
    """
    return [d[idx] for idx in range(len(d))]


def memoization(f):
    @functools.wraps(f)
    def wrapped(*args):
        key = tuple(args)
        if key in wrapped.__states__:
            return wrapped.__states__[key]
        ret = f(*args)
        wrapped.__states__[key] = ret
        return ret

    wrapped.__states__ = {}
    return wrapped


def reverse_map(d: Union[Dict[T, int], List[T]]) -> List[T]:
    """
    Given a dict containing pairs of (`item`, `id`), return a list where the `id`-th element is `item`.

    Or, given a list containing a permutation, return its reverse.

    :type d: dict | list | np.ndarray
    """
    if isinstance(d, dict):
        return [k for k, _ in sorted(d.items(), key=lambda xs: xs[1])]

    rev = [0] * len(d)
    for idx, x in enumerate(d):
        rev[x] = idx
    return rev
