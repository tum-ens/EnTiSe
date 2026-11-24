"""Methods that calculate several of their types in one call.

A method such as a building model that yields heating and cooling loads together, or
a demand generator that yields heat, cold and electricity from shared inputs, runs
its calculation once per object for all the types that select it. Subclass
``GroupedMethod`` together with ``Method`` and implement ``generate_combined``:

.. code-block:: python

   class SiteDemand(GroupedMethod, Method):
       types = [Types.HEATING, Types.COOLING, Types.ELECTRICITY]
       ...

       def generate(self, obj=None, data=None, results=None, ts_type=Types.HEATING, **inputs):
           obj, data = self._process_kwargs(obj, data, ts_type, **inputs)
           return self.group_result(obj, data, results, ts_type)

       def generate_combined(self, obj, data, results, ts_types):
           # One calculation; columns and summary keys are prefixed with their type,
           # such as "heating:load[W]" and "electricity:demand[Wh]".
           return {Keys.TIMESERIES: frame, Keys.SUMMARY: summary}

The runner calls ``generate_group`` once for the types of ``types`` that select the
method on an object (see ``Runner.resolve``). Each type's result holds the columns and
summary keys prefixed with that type, plus the ``shared_columns`` of the method.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, Sequence, Tuple

from entise.constants import SEP, VALID_TYPES
from entise.constants import Keys as K

if TYPE_CHECKING:
    from entise.core.base import Method

    _Base = Method
else:
    _Base = object


class GroupedMethod(_Base):
    """Mixin for a ``Method`` that calculates several of its types in one call."""

    # Columns that belong to every type's result, such as an indoor temperature.
    shared_columns: Tuple[str, ...] = ()

    def generate_group(self, obj: dict, data: dict, results: dict, ts_types: Iterable[str]) -> Dict[str, dict]:
        """Return the result of each type in ``ts_types`` from one calculation."""
        ts_types = tuple(ts_types)
        combined = self.generate_combined(self.group_object(obj, ts_types), data, results, ts_types)
        return split_by_type(combined, ts_types, self.shared_columns)

    def generate_combined(self, obj: dict, data: dict, results: dict, ts_types: Tuple[str, ...]) -> dict:
        """Calculate the selected types together.

        Args:
            obj: The object's inputs for the selected types, unscoped (see ``group_object``).
            data: Input tables.
            results: Results of the types generated before.
            ts_types: The selected types. The result may hold other types too; they are dropped.

        Returns:
            dict: One result whose columns and summary keys are prefixed with their type.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement 'generate_combined'.")

    def group_object(self, obj: dict, ts_types: Tuple[str, ...]) -> dict:
        """Resolve the object's inputs for the selected types into one unscoped object.

        Each declared key takes its value from the key scoped to one of the types, then
        the plain key, then its default (see ``Method.get_relevant_objects``). A key given
        under two of the types with different values raises a ``ValueError``. Keys scoped
        to other types are not read.
        """
        flat = {key: value for key, value in obj.items() if not is_type_scoped(key)}
        flat.update(self.get_relevant_objects(obj, ts_types))
        return flat

    def group_result(self, obj: dict, data: dict, results: dict, ts_type: str) -> dict:
        """Return the result of one type, as a direct call of ``generate`` with ``ts_type`` does."""
        return self.generate_group(obj, data, results, (ts_type,))[ts_type]


def split_by_type(result: dict, ts_types: Iterable[str], shared_columns: Sequence[str] = ()) -> Dict[str, dict]:
    """Split one result into a result per type by the type prefix of its columns and summary keys.

    Args:
        result: A result with ``Keys.TIMESERIES`` and, optionally, ``Keys.SUMMARY``.
        ts_types: The types to return.
        shared_columns: Columns that every type's result keeps.

    Returns:
        dict: Each type mapped to its result.
    """
    frame = result[K.TIMESERIES]
    summary = result.get(K.SUMMARY, {}) or {}
    shared = set(shared_columns)
    outputs = {}
    for kind in ts_types:
        prefix = f"{kind}{SEP}"
        columns = [col for col in frame.columns if col in shared or str(col).startswith(prefix)]
        outputs[kind] = {
            K.TIMESERIES: frame.loc[:, columns],
            K.SUMMARY: {key: value for key, value in summary.items() if str(key).startswith(prefix)},
        }
    return outputs


def is_type_scoped(key: Any) -> bool:
    """Whether an object key is scoped to a time series type, such as ``heating:power_heating[W]``."""
    return isinstance(key, str) and SEP in key and key.split(SEP, 1)[0] in VALID_TYPES
