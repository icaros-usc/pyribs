"""Tests for ribs config functions."""
import pytest

from ribs.config import DEFAULT_CONFIG, merge_with_default, update

# pylint: disable = invalid-name


def test_update_overrides_single_dict():
    default = {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    new = {
        "b": 4,
    }
    update(default, new)

    assert default == {
        "a": 1,
        "b": 4,
        "c": 3,
    }


def test_update_overrides_nested_dict():
    default = {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3,
        },
    }
    new = {
        "a": 4,
        "b": {
            "c": 5,
        },
    }
    update(default, new)

    assert default == {
        "a": 4,
        "b": {
            "c": 5,
            "d": 3,
        }
    }


def test_update_adds_new_keys():
    default = {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3,
        },
    }
    new = {
        "a": 4,
        "b": {
            "c": 5,
            "e": 6,
        },
        "f": 7,
    }
    update(default, new)

    assert default == {
        "a": 4,
        "b": {
            "c": 5,
            "d": 3,
            "e": 6,
        },
        "f": 7,
    }


def test_update_adds_new_dict_keys():
    default = {}
    new = {
        "a": {
            "b": 1,
            "c": 2,
        },
    }
    update(default, new)

    assert default == {
        "a": {
            "b": 1,
            "c": 2,
        },
    }


def test_update_deeper_nested_dict():
    default = {
        "a": {
            "b": {
                "c": {
                    "d": {
                        "e": 1,
                    },
                },
            },
        },
    }
    new = {
        "a": {
            "b": {
                "c": {
                    "d": {
                        "e": 6,
                    },
                },
            },
        },
    }
    update(default, new)

    assert default == {
        "a": {
            "b": {
                "c": {
                    "d": {
                        "e": 6,
                    },
                },
            },
        },
    }


def test_override_non_dict_val_fails():
    with pytest.raises(TypeError):
        default = {
            "a": 1,
        }
        new = {
            "a": {
                "b": 2,
            },
        }
        update(default, new)


def test_merging_default_with_nothing_gives_default():
    assert merge_with_default({}) == DEFAULT_CONFIG
