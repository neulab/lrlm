import functools
from typing import Any, Callable

Validator = Callable[[Any], bool]


class ValidationError(ValueError):
    def __init__(self, arg, val, vtor):
        super().__init__(f"Argument '{arg}' failed validator '{vtor}' with value: {val!r}")


def validator(nullable=False, default=None):  # TODO: confusing naming
    def decorator(f):
        @functools.wraps(f)
        def wrapped(**kwargs):
            if not nullable and 'nullable' in kwargs:
                raise ValueError(f"Validator \"{f.__name__}\" is not nullable.")
            this_nullable = nullable and kwargs.get('nullable', default)
            if this_nullable is None:
                raise ValueError(f"You must specify whether validator \"{f.__name__}\" is nullable or not.")

            @functools.wraps(f)
            def call(value, **kw):
                if value is None:
                    return this_nullable
                return f(value, **kw)

            if nullable:
                call.__name__ += f'(nullable={this_nullable})'

            return call

        return wrapped

    return decorator


@validator(nullable=True, default=False)
def is_path(s):
    import os
    return os.path.exists(s)


@validator(nullable=True, default=True)
def is_embedding_type(s):
    return s in ['word2vec', 'fasttext']


@validator()
def is_dropout(s):
    return 0 <= s < 1


@validator()
def is_lstm_dropout(s):
    return len(s) == 2 and all(map(is_dropout(), s))


@validator()
def is_activation(s):
    return s in ['relu', 'sigmoid', 'tanh', 'id']


@validator()
def is_optimizer(s):
    return s in ['adam', 'sgd', 'adamax']
