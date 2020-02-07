# TODO: lot's to be done here
import typing
from pathlib import Path

from nnlib.arguments.validator import ValidationError

__all__ = ['NoneType', 'Path', 'Choices', 'is_choices']

NoneType = type(None)


class ArgType:
    def __call__(self, value):
        try:
            ret = super().__new__(value)
            ret.__init__(value)
            return ret
        except ValueError:
            raise ValidationError(f"Invalid value \"{value}\" for type ") from None


@typing.no_type_check
class _Choices:
    def __new__(cls, values=None):
        self = super().__new__(cls)
        self.__values__ = values
        return self

    def __getitem__(self, values):
        if values == ():
            raise TypeError("Choices must contain at least one string")
        if not isinstance(values, tuple):
            values = (values,)
        values = tuple(values)
        return self.__class__(values)


Choices = _Choices()


def is_choices(typ) -> bool:
    """
    Check whether a type is a choices type. This cannot be checked using traditional methods,
    since Choices is a metaclass.
    """
    return type(typ) is type(Choices)
