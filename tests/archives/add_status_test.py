"""Tests for AddStatus.

Though AddStatus is an enum, we make sure that we uphold several properties that
we use throughout our algorithms.
"""
from ribs.archives import AddStatus


def test_ordering():
    assert AddStatus.NEW > AddStatus.IMPROVE_EXISTING > AddStatus.NOT_ADDED


def test_boolean():
    assert bool(AddStatus.NEW)
    assert bool(AddStatus.IMPROVE_EXISTING)
    assert not bool(AddStatus.NOT_ADDED)
