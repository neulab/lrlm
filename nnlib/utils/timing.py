"""
Timing Functions
"""
import contextlib
import time
from functools import wraps
from typing import Dict

from nnlib.utils.logging import Logging

__all__ = ['work_in_progress', 'tic', 'toc', 'report_timing', 'tic_toc']


@contextlib.contextmanager
def work_in_progress(msg):
    begin_time = time.time()
    Logging.verbose(msg + "... ", end='')
    yield
    Logging.verbose(f"done. ({time.time() - begin_time:.2f}s)")


class _TimingHelperClass:
    ticks = 0
    time_records: Dict[str, float] = {}


def tic():
    _TimingHelperClass.ticks = time.time()


def toc(key=None):
    ticks = time.time() - _TimingHelperClass.ticks

    if key not in _TimingHelperClass.time_records:
        _TimingHelperClass.time_records[key] = ticks
    else:
        _TimingHelperClass.time_records[key] += ticks


def report_timing(level=Logging.VERBOSE):
    Logging(level).log("Time consumed:")
    for k in sorted(_TimingHelperClass.time_records.keys()):
        v = _TimingHelperClass.time_records[k]
        Logging(level).log(f"> {k}: {v:f}")
    Logging(level).log("------")
    _TimingHelperClass.time_records = {}


def tic_toc(f):
    func_name = f.__module__ + '.' + f.__name__

    @wraps(f)
    def func(*args, **kwargs):
        tic()
        result = f(*args, **kwargs)
        toc(func_name)
        return result

    return func
