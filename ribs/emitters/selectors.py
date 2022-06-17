from abc import ABC, abstractmethod
from ribs._docstrings import DocstringComponents, _core_docs

_args = _core_docs["args"]
_returns = DocstringComponents(
    dict(num_parents="""
    the number of top performing parents to select from the solutions.
    """,))


class SelectorBase(ABC):

    @abstractmethod
    def select(self, emitter, archive, solutions, objective_values,
               behavior_values, metadata, add_statuses, add_values):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for select
    select.__doc__ = f"""
Selects the number of parents that will be used for the evolution strategy.

Args:
{_args.emitter}
{_args.archive}
{_args.solutions}
{_args.objective_values}
{_args.behavior_values}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.num_parents}
    """

    def reset(self, emitter, archive):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for reset
    reset.__doc__ = f"""
Resets the internal state of the selector.

Args:
{_args.emitter}
{_args.archive}
   """


class MuSelector(SelectorBase):

    def select(self, emitter, archive, solutions, objective_values,
               behavior_values, metadata, add_statuses, add_values):
        return emitter.batch_size // 2

    # Generates the docstring for select
    select.__doc__ = f"""
Selects the number of parents that will be used for the evolution strategy

Args:
{_args.emitter}
{_args.archive}
{_args.solutions}
{_args.objective_values}
{_args.behavior_values}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.num_parents}
    """


class FilterSelector(SelectorBase):

    def select(self, emitter, archive, solutions, objective_values,
               behavior_values, metadata, add_statuses, add_values):
        new_sols = 0
        for status in add_statuses:
            if bool(status):
                new_sols += 1
        return new_sols

    # Generates the docstring for select
    select.__doc__ = f"""
Selects all the added solutions and the improved solutions.

Args:
{_args.emitter}
{_args.archive}
{_args.solutions}
{_args.objective_values}
{_args.behavior_values}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.num_parents}
    """
