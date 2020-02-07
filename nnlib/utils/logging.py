import sys
import time
import warnings
from typing import List, TextIO

__all__ = ['Logging']


class Logging:
    CLEAR_LINE = '\033[2K\r'
    RESET_CODE = '\033[0m'
    COLOR_CODE = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[94m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'gray': '\033[37m',
        'grey': '\033[37m'
    }

    ERROR = 0
    WARNING = DEFAULT = 1
    INFO = VERBOSE = 2
    DEBUG = 3
    verbosity_level = DEFAULT

    _global_files: List[TextIO] = []

    def __init__(self, level=DEFAULT, file: TextIO = sys.stderr):
        self._level = level
        self._file = file

    @classmethod
    def tee(cls, file):
        cls._global_files.append(file)

    @classmethod
    def _format(cls, *args, **kwargs):
        prefix = kwargs.get('prefix', None)
        color = kwargs.get('color', None)
        prefix_color = kwargs.get('prefix_color', color)
        end = kwargs.get('end', '\n')
        sep = kwargs.get('sep', ' ')
        args = list(args)
        if prefix is not None:
            if prefix_color is not None:
                prefix = cls.COLOR_CODE[prefix_color.lower()] + prefix + cls.RESET_CODE
            prefix += sep
        else:
            prefix = ''
        if kwargs.get('timestamp', False):
            args = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"] + args

        s = sep.join(str(x) for x in args)
        if color is not None:
            s = cls.COLOR_CODE[color.lower()] + s + cls.RESET_CODE
        s = prefix + s + end
        return s

    def _log(self, *args, **kwargs):
        if Logging.verbosity_level >= self._level:
            s = self._format(*args, **kwargs)
            for file in self._global_files + [self._file]:
                file.write(s)
                file.flush()

    def log(self, *args, **kwargs):
        self._log(*args, **kwargs)

    @classmethod
    def warn(cls, *args, category=None, stacklevel=1, **kwargs):
        if cls.verbosity_level >= cls.WARNING:
            ignored = category in [DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning]
            if ignored:
                warnings.simplefilter(action='default', category=category)
            warnings.warn(cls._format(*args, **kwargs), category=category, stacklevel=stacklevel + 1)
            if ignored:
                warnings.filters.pop(0)

    @classmethod
    def info(cls, *args, **kwargs):
        cls(cls.INFO, file=sys.stderr)._log(*args, prefix="INFO:", prefix_color='green', **kwargs)

    @classmethod
    def verbose(cls, *args, **kwargs):
        cls(cls.VERBOSE, file=sys.stderr)._log(*args, **kwargs)

    @classmethod
    def color(cls, col, s):
        return cls.COLOR_CODE[col.lower()] + s + cls.RESET_CODE


# monkey-patch built-in warnings.showwarning
# noinspection PyUnusedLocal
def showwarning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    text = f"{filename}:{lineno}: {Logging.COLOR_CODE['red']}{category.__name__}{Logging.RESET_CODE}: {message}\n"
    file.write(text)  # we don't deal with situations when built-in stuff broke


warnings.showwarning = showwarning
