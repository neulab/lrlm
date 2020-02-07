import ast
import enum
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from nnlib.arguments import custom_types
from nnlib.arguments.custom_types import NoneType, is_choices
from nnlib.arguments.validator import ValidationError, Validator
from nnlib.utils import Logging


class ArgumentError(Exception):
    pass


class Arguments:
    r"""

    """

    _reserved_keys_cls = ['switch', 'enum']
    _reserved_keys_method = ['validate', 'preprocess', 'postprocess', 'to_string']
    _reserved_keys_other = ['help']
    _reserved_keys = _reserved_keys_cls + _reserved_keys_method + _reserved_keys_other

    class Enum(enum.Enum):
        # noinspection PyMethodParameters
        def _generate_next_value_(name, start, count, last_values):
            # support return-type-polymorphism:   a, b, c = auto()
            return name.lower()

        def __eq__(self, other):
            return self.value == other or super().__eq__(other)

    class Switch:
        r"""
        A flag argument that takes no values. Similar to the :attr:`action='store_true'` setting of :mod:`argparse`.
        """

        def __init__(self, default: bool = False):
            self._default = default
            self._value = None

        def set_default(self, default: bool):
            self._default = default

        def __bool__(self):
            return self._value if self._value is not None else self._default

        def __repr__(self):
            return f"(Switch) {self._value}"

    pdb = Switch()

    def _check_types(self):
        pass

    class _ArgTypeSpec(NamedTuple):
        typ: type
        nullable: bool
        required: bool
        default: Any

    @classmethod
    def _check_reserved(cls, arg_name) -> bool:
        """
        Check whether an argument name is reserved.
        """
        key = arg_name.lower()
        if key not in Arguments._reserved_keys:
            return False
        if key in Arguments._reserved_keys_other:
            return True
        if key in Arguments._reserved_keys_cls and getattr(cls, arg_name) is not getattr(Arguments, arg_name):
            return True
        if key in Arguments._reserved_keys_method and not inspect.ismethod(getattr(cls, arg_name)):
            return True
        return False

    # TODO
    @classmethod
    def _parse_type_spec(cls) -> Dict[str, _ArgTypeSpec]:
        """
        :return: A dict mapping argument names to their type-specs
        """
        _attr_name = '__type_dict__'
        if hasattr(cls, _attr_name):
            return getattr(cls, _attr_name)

        type_dict = {}

        # get annotations from the current class and all its base classes as well
        annotations = {}
        for base in reversed(cls.__mro__):
            if base not in [object, Arguments]:
                annotations.update(base.__dict__.get('__annotations__', {}))

        bad_names = []
        warn_names = []

        def check_name_conventions(name):
            if name.startswith('_') or name.endswith('_'):
                # names should not start or begin with underscores
                bad_names.append(name)
            if name != name.lower() or any(ord(c) >= 128 for c in name):
                # names are recommended to contain non-uppercase ASCII characters only
                warn_names.append(name)

        # check that all attributes are annotated, except for `Switch`es
        for arg_name in dir(cls):
            if arg_name.startswith('__') or cls._check_reserved(arg_name):  # magic stuff
                continue
            arg_val = getattr(cls, arg_name)
            if isinstance(arg_val, Arguments.Switch):
                check_name_conventions(arg_name)
                # noinspection PyProtectedMember,PyCallByClass
                type_dict[arg_name.lower()] = Arguments._ArgTypeSpec(
                    Arguments.Switch, nullable=False, required=False, default=arg_val._default)
            elif arg_name not in cls.__annotations__:
                raise ArgumentError(f"Type is not specified for argument '{arg_name}'. "
                                    f"Type annotation can omitted only when argument is a `Switch`.")

        # iterate over annotated values and generate type-specs
        for arg_name, arg_typ in annotations.items():
            if cls._check_reserved(arg_name):
                raise ArgumentError(f"'{arg_name}' cannot be used as argument name because it is reserved.")

            check_name_conventions(arg_name)

            nullable = False
            # hacky check of whether `arg_typ` is `Optional`: `Optional` is `Union` with `type(None)`
            if getattr(arg_typ, '__origin__', None) is Union and NoneType in arg_typ.__args__:
                nullable = True
                # extract the type wrapped inside `Optional`
                arg_typ = next(t for t in arg_typ.__args__ if not isinstance(t, NoneType))  # type: ignore

            arg_val = getattr(cls, arg_name, None)
            required = not hasattr(cls, arg_name) or (arg_val is None and not nullable)
            type_dict[arg_name] = Arguments._ArgTypeSpec(
                arg_typ, nullable=nullable, required=required, default=arg_val)

        if len(bad_names) > 0:
            bad_names_str = ', '.join(f"'{s}'" for s in bad_names)
            raise ArgumentError(f"Invalid argument names: {bad_names_str}. "
                                f"Names cannot begin or end with underscores.")
        if len(warn_names) > 0:
            warn_names_str = ', '.join(f"'{s}'" for s in warn_names)
            Logging.warn(f"Consider changing these argument names: {warn_names_str}. "
                         f"Names are recommended to contain non-uppercase ASCII characters only.")

        setattr(cls, _attr_name, type_dict)
        return type_dict

    @classmethod
    def _parse_help(cls) -> Dict[str, str]:
        pass

    def _get_arg_type(self, argname: str) -> Tuple[bool, type]:
        attr = getattr(self, argname)
        typ = self.__annotations__.get(argname, type(attr))
        # TODO: hacks here
        if hasattr(typ, '__origin__') and typ.__origin__ == Union and type(None) in typ.__args__:
            # hacky check of whether `typ` is `Optional`
            typ = next(t for t in typ.__args__ if not isinstance(t, custom_types.NoneType))  # type: ignore
            return True, typ
        return False, typ

    def __init__(self, *args, **kwargs) -> None:
        self._check_types()

        for k, v in kwargs.items():
            setattr(self, k, v)

        # TODO: Add non-null checks
        # TODO: Add "no-" prefix stuff for switches
        # TODO: Generate help by inspecting comments

        if len(args) == 0:
            argv = sys.argv
        elif len(args) == 1:
            argv = args[0]
        else:
            raise ValueError(f"Argument class takes zero or one positional arguments but {len(args)} were given")
        i = 1
        while i < len(argv):
            arg: str = argv[i]
            if arg.startswith('--'):
                argname = arg[2:].replace('-', '_')
                if argname.startswith('no_') and not hasattr(self, argname) and hasattr(self, argname[3:]):
                    attr = getattr(self, argname[3:])
                    if isinstance(attr, Arguments.Switch):
                        attr._value = False
                        i += 1
                        continue

                if hasattr(self, argname):
                    attr = getattr(self, argname)
                    if isinstance(attr, Arguments.Switch):
                        attr._value = True
                        i += 1
                        continue

                    nullable, typ = self._get_arg_type(argname)
                    argval: str = argv[i + 1]
                    if argval.lower() == 'none':
                        if nullable:
                            val = None
                        else:
                            assert typ is str or is_choices(typ), \
                                f"Cannot assign None to non-nullable, non-str argument '{argname}'"
                            val = argval
                    elif isinstance(typ, custom_types.NoneType):  # type: ignore
                        val = None  # just to suppress "ref before assign" warning
                        try:
                            # priority: low -> high
                            for target_typ in [str, float, int]:
                                val = target_typ(argval)
                        except ValueError:
                            pass
                    elif typ is str:
                        val = argval
                    elif isinstance(typ, custom_types.Path) or typ is custom_types.Path:
                        val = Path(argval)
                        if isinstance(typ, custom_types.Path) and typ.exists:
                            assert val.exists(), ValueError(f"Argument '{argname}' requires an existing path, "
                                                            f"but '{argval}' does not exist")
                    elif is_choices(typ):
                        val = argval
                        assert val in typ.__values__, f"Invalid value '{val}' for argument '{arg}', " \
                                                      f"available choices are: {typ.__values__}"
                    elif issubclass(typ, Arguments.Enum):
                        # experimental support for custom enum
                        try:
                            # noinspection PyCallingNonCallable
                            val = typ(argval)
                        except ValueError:
                            valid_args = {x.value for x in typ}
                            raise ValueError(f"Invalid value '{argval}' for argument '{argname}', "
                                             f"available choices are: {valid_args}") from None

                    elif typ is bool:
                        val = argval in ['true', '1', 'True', 'y', 'yes']
                    else:
                        try:
                            val = ast.literal_eval(argval)
                        except ValueError:
                            raise ValueError(f"Invalid value '{argval}' for argument '{argname}'") from None
                    setattr(self, argname, val)
                    i += 2
                else:
                    raise ValueError(f"Invalid argument: '{arg}'")
            else:
                Logging.warn(f"Unrecognized command line argument: '{arg}'")
                i += 1

        if self.pdb:
            # enter IPython debugger on exception
            from IPython.core import ultratb
            ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)

            def excepthook(type, value, traceback):
                if type is KeyboardInterrupt:
                    # don't capture keyboard interrupts (Ctrl+C)
                    sys.__excepthook__(type, value, traceback)
                else:
                    ipython_hook(type, value, traceback)

            sys.excepthook = excepthook

        self.preprocess()

        # check whether non-optional attributes are none
        for arg in dir(self):
            if not arg.startswith('_') and arg not in self._reserved_keys:
                attr = getattr(self, arg)
                nullable, _ = self._get_arg_type(arg)
                if attr is None and not nullable:
                    raise ValueError(f"argument '{arg}' cannot be none")

        self._validate()
        self.postprocess()

        # convert switches to bool
        for arg in dir(self):
            if not arg.startswith('_') and arg not in self._reserved_keys:
                attr = getattr(self, arg)
                typ = self.__annotations__.get(arg, None)
                if isinstance(attr, Arguments.Switch):
                    # noinspection PyProtectedMember
                    setattr(self, arg, bool(attr))
                if isinstance(typ, type) and issubclass(typ, Path) and isinstance(attr, str):
                    setattr(self, arg, Path(attr))

    def _validate(self):
        rules = self.validate() or []
        for pattern, validator in rules:
            regex = re.compile(rf"^{pattern}$")
            matches = 0
            for k in dir(self):
                if not k.startswith('_') and regex.match(k):
                    matches += 1
                    v = getattr(self, k)
                    try:
                        result = validator(v)
                    except Exception:
                        raise ValidationError(k, v, validator.__name__)
                    else:
                        if not result:
                            raise ValidationError(k, v, validator.__name__)
            if matches == 0:
                Logging.warn(f"regex \"{pattern}\" did not match any arguments")

    def preprocess(self) -> None:
        """
        Postprocessing of the arguments. This will be called before validation.
        """
        pass

    def validate(self) -> List[Tuple[str, Validator]]:
        r"""
        Return a list of validation rules. Each validation rule is a tuple of (pattern, validator), where:

        - ``pattern``: A regular expression string. The validation rule is applied to all arguments whose name is
          fully-matched (i.e. ``^$``) by the pattern.
        - ``validator``: A validator instance from ``arguments.validator``.

        Example::

            def validate(self):
                return [
                    ('.*_data_path', validator.is_path()),
                    ('pretrained_embedding_type', validator.is_embedding_type()),
                    ('pretrained_embedding_path', validator.is_path(nullable=True)),
                    ('.*_lstm_dropout', validator.is_dropout()),
                ]

        :return: List of (pattern, validator).
        """
        pass

    def postprocess(self) -> None:
        """
        Postprocessing of the arguments. This will be called after validation.
        """
        pass

    # Credit: https://github.com/eaplatanios/symphony-mt/
    def to_string(self, max_width=None) -> str:
        k_col = "Arguments"
        v_col = "Values"
        valid_keys = [k for k in dir(self) if not (k.startswith('_') or k.lower() in self._reserved_keys)]
        # valid_keys = list(self._get_arg_names())
        valid_vals = [repr(getattr(self, k)) for k in valid_keys]
        max_key = max(len(k_col), max(len(k) for k in valid_keys))
        max_val = max(len(v_col), max(len(v) for v in valid_vals))
        if max_width is not None:
            max_val = min(max_val, max_width - max_key - 7)

        def get_row(k: str, v: str) -> str:
            if len(v) > max_val:
                v = v[:((max_val - 5) // 2)] + ' ... ' + v[-((max_val - 4) // 2):]
                assert len(v) == max_val
            return f"║ {k.ljust(max_key)} │ {v.ljust(max_val)} ║\n"

        s = repr(self.__class__) + '\n'
        s += f"╔═{'═' * max_key}═╤═{'═' * max_val}═╗\n"
        s += get_row(k_col, v_col)
        s += f"╠═{'═' * max_key}═╪═{'═' * max_val}═╣\n"
        for k, v in zip(valid_keys, valid_vals):
            s += get_row(k, v)
        s += f"╚═{'═' * max_key}═╧═{'═' * max_val}═╝\n"
        return s

    def __str__(self):
        return self.to_string()
