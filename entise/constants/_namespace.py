"""A metaclass for classes that group plain string constants, such as ``Types``."""

from collections.abc import Iterator


class Namespace(type):
    """Make a class of string constants iterable and checkable for membership.

    The constants are the upper-case class attributes that hold a ``str``. They stay plain
    strings, so ``Types.DHW`` is ``"dhw"``. The class iterates over their values in the
    order the class body defines them, ``value in cls`` checks for a value, and ``len(cls)``
    counts them. The constants cannot be changed after the class is defined, and the class
    is not instantiated.
    """

    def __new__(mcs, name, bases, namespace):
        constants = {key: value for key, value in namespace.items() if key.isupper() and isinstance(value, str)}
        cls = super().__new__(mcs, name, bases, namespace)
        type.__setattr__(cls, "_constants", constants)
        type.__setattr__(cls, "_values", frozenset(constants.values()))
        return cls

    def __iter__(cls) -> Iterator[str]:
        return iter(cls._constants.values())

    def __contains__(cls, value: object) -> bool:
        try:
            return value in cls._values
        except TypeError:  # an unhashable value is not a constant
            return False

    def __len__(cls) -> int:
        return len(cls._constants)

    def items(cls):
        """Return the constants as ``(name, value)`` pairs, in definition order."""
        return cls._constants.items()

    def __setattr__(cls, name, value):
        raise AttributeError(f"{cls.__name__} constants cannot be changed or added")

    def __delattr__(cls, name):
        raise AttributeError(f"{cls.__name__} constants cannot be removed")

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} groups constants and is not instantiated")
