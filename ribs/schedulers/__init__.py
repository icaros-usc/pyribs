"""Schedulers link the entire QD algorithm together.

Specifically, the scheduler performs two roles. First, the scheduler facilitates
the interaction between the archive and the population of emitters. The
scheduler adds solutions generated by emitters to the archive and passes the
results of evaluation and archive insertion to the emitters. Second, schedulers
select which emitters generate new solutions on each iteration of the algorithm.
Schedulers make decisions on active emitters based on how well each emitter
performs in previous iterations.

.. autosummary::
    :toctree:

    ribs.schedulers.Scheduler
    ribs.schedulers.BanditScheduler
"""
from ribs.schedulers._bandit_scheduler import BanditScheduler
from ribs.schedulers._bayesian_opt_scheduler import \
    BayesianOptimizationScheduler
from ribs.schedulers._scheduler import Scheduler

__all__ = [
    "Scheduler",
    "BanditScheduler",
    "BayesianOptimizationScheduler",
]