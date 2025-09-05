"""Tests for ArrayStore."""

import pickle as pkl

import pytest
from array_api_compat import numpy as np

from ribs.archives import ArrayStore


def test_init_reserved_field(xp_and_device):
    xp, device = xp_and_device

    with pytest.raises(ValueError):
        ArrayStore(
            {
                "index": ((), xp.float32),
            },
            capacity=10,
            xp=xp,
            device=device,
        )


def test_init_invalid_field(xp_and_device):
    xp, device = xp_and_device

    with pytest.raises(ValueError):
        ArrayStore(
            {
                # The space makes this an invalid identifier.
                "foo bar": ((), xp.float32),
            },
            capacity=10,
            xp=xp,
            device=device,
        )


@pytest.mark.parametrize(
    "shape", [((), (2,), (10,)), ((), 2, 10)], ids=["tuple", "int"]
)
def test_init(xp_and_device, shape):
    xp, device = xp_and_device

    capacity = 10
    store = ArrayStore(
        {
            "objective": (shape[0], xp.float32),
            "measures": (shape[1], xp.float32),
            "solution": (shape[2], xp.float32),
        },
        capacity=capacity,
        xp=xp,
        device=device,
    )

    assert len(store) == 0
    assert store.capacity == capacity
    assert xp.all(~store.occupied)
    assert len(store.occupied_list) == 0
    assert store.field_desc == {
        "objective": (shape[0], xp.float32),
        "measures": (
            (shape[1],) if isinstance(shape[1], int) else shape[1],
            xp.float32,
        ),
        "solution": (
            (shape[2],) if isinstance(shape[2], int) else shape[2],
            xp.float32,
        ),
    }
    assert store.field_list == ["objective", "measures", "solution"]
    assert store.field_list_with_index == ["objective", "measures", "solution", "index"]
    assert store.dtypes == {
        "objective": xp.float32,
        "measures": xp.float32,
        "solution": xp.float32,
    }
    assert store.dtypes_with_index == {
        "objective": xp.float32,
        "measures": xp.float32,
        "solution": xp.float32,
        "index": xp.int32,
    }


@pytest.fixture
def store(xp_and_device):
    """Simple ArrayStore for testing."""
    xp, device = xp_and_device
    return ArrayStore(
        field_desc={
            "objective": ((), xp.float32),
            "measures": ((2,), xp.float32),
            "solution": ((10,), xp.float32),
        },
        capacity=10,
        xp=xp,
        device=device,
    )


def test_add_wrong_keys(store):
    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0],
                "measures": [[1.0, 2.0], [3.0, 4.0]],
                # Missing `solution` key.
            },
        )


def test_add_mismatch_indices(store, xp_and_device):
    xp, device = xp_and_device

    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0, 3.0],  # Length 3 instead of 2.
                "measures": [[1.0, 2.0], [3.0, 4.0]],
                "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
            },
        )


def test_simple_add_retrieve_clear(store, xp_and_device):
    """Add without transforms, retrieve the data, and clear the archive."""
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    assert len(store) == 2
    assert xp.all(
        store.occupied
        == xp.asarray(
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            dtype=bool,
            device=device,
        )
    )
    assert xp.all(
        xp.sort(store.occupied_list)
        == xp.asarray(
            [3, 5],
            dtype=xp.int32,
            device=device,
        )
    )

    occupied, data = store.retrieve([5, 3])

    assert xp.all(
        occupied
        == xp.asarray(
            [True, True],
            dtype=bool,
            device=device,
        )
    )
    assert data.keys() == {"objective", "measures", "solution", "index"}
    assert xp.all(
        data["objective"]
        == xp.asarray(
            [2.0, 1.0],
            dtype=xp.float32,
            device=device,
        )
    )
    assert xp.all(
        data["measures"]
        == xp.asarray(
            [[3.0, 4.0], [1.0, 2.0]],
            dtype=xp.float32,
            device=device,
        )
    )
    assert xp.all(
        data["solution"]
        == xp.stack(
            (xp.ones(10, device=device), xp.zeros(10, device=device)),
            axis=0,
        )
    )
    assert xp.all(
        data["index"]
        == xp.asarray(
            [5, 3],
            dtype=xp.int32,
            device=device,
        )
    )

    store.clear()

    assert len(store) == 0
    assert xp.all(
        store.occupied
        == xp.asarray(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=bool,
            device=device,
        )
    )
    assert len(store.occupied_list) == 0


def test_add_duplicate_indices(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 3],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    assert len(store) == 1
    assert xp.all(
        store.occupied
        == xp.asarray(
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            dtype=bool,
            device=device,
        )
    )
    assert xp.all(
        store.occupied_list
        == xp.asarray(
            [3],
            dtype=xp.int32,
            device=device,
        )
    )


def test_add_nothing(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [],
        {
            "objective": [],
            "measures": [],
            "solution": [],
        },
    )

    assert len(store) == 0
    assert xp.all(
        store.occupied
        == xp.asarray(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=bool,
            device=device,
        )
    )
    assert xp.all(
        store.occupied_list
        == xp.asarray(
            [],
            dtype=xp.int32,
            device=device,
        )
    )


def test_dtypes(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    _, data = store.retrieve([5, 3])

    # Index is always int32, and other fields were defined as float32 in the
    # `store` fixture.
    assert data["objective"].dtype == xp.float32
    assert data["measures"].dtype == xp.float32
    assert data["solution"].dtype == xp.float32
    assert data["index"].dtype == xp.int32


def test_retrieve_duplicate_indices(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3],
        {
            "objective": [2.0],
            "measures": [[3.0, 4.0]],
            "solution": xp.ones((1, 10)),
        },
    )

    occupied, data = store.retrieve([3, 3])

    assert xp.all(
        occupied
        == xp.asarray(
            [True, True],
            dtype=bool,
            device=device,
        )
    )
    assert data.keys() == {"objective", "measures", "solution", "index"}
    assert xp.all(
        data["objective"]
        == xp.asarray(
            [2.0, 2.0],
            dtype=xp.float32,
            device=device,
        )
    )
    assert xp.all(
        data["measures"]
        == xp.asarray(
            [[3.0, 4.0], [3.0, 4.0]],
            dtype=xp.float32,
            device=device,
        )
    )
    assert xp.all(
        data["solution"]
        == xp.stack(
            (xp.ones(10, device=device), xp.ones(10, device=device)),
            axis=0,
        )
    )
    assert xp.all(
        data["index"]
        == xp.asarray(
            [3, 3],
            dtype=xp.int32,
            device=device,
        )
    )


def test_retrieve_invalid_fields(store):
    with pytest.raises(ValueError):
        store.retrieve([0, 1], fields=["objective", "foo"])


def test_retrieve_invalid_return_type(store):
    with pytest.raises(ValueError):
        store.retrieve([0, 1], return_type="foo")


def test_retrieve_pandas_2d_fields(store, xp_and_device):
    xp, device = xp_and_device

    store = ArrayStore(
        {
            "solution": ((10, 10), xp.float32),
        },
        capacity=10,
        xp=xp,
        device=device,
    )

    with pytest.raises(ValueError):
        store.retrieve([], return_type="pandas")


@pytest.mark.parametrize("return_type", ["dict", "tuple", "pandas"])
def test_retrieve(return_type, store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    occupied, data = store.retrieve([5, 3], return_type=return_type)

    if return_type == "dict":
        assert xp.all(
            occupied
            == xp.asarray(
                [True, True],
                dtype=bool,
                device=device,
            )
        )
        assert data.keys() == {"objective", "measures", "solution", "index"}
        assert xp.all(
            data["objective"]
            == xp.asarray(
                [2.0, 1.0],
                dtype=xp.float32,
                device=device,
            )
        )
        assert xp.all(
            data["measures"]
            == xp.asarray(
                [[3.0, 4.0], [1.0, 2.0]],
                dtype=xp.float32,
                device=device,
            )
        )
        assert xp.all(
            data["solution"]
            == xp.stack(
                (xp.ones(10, device=device), xp.zeros(10, device=device)),
                axis=0,
            )
        )
        assert xp.all(
            data["index"]
            == xp.asarray(
                [5, 3],
                dtype=xp.int32,
                device=device,
            )
        )
    elif return_type == "tuple":
        objective, measures, solution, index = data
        assert xp.all(
            occupied
            == xp.asarray(
                [True, True],
                dtype=bool,
                device=device,
            )
        )
        assert xp.all(
            objective
            == xp.asarray(
                [2.0, 1.0],
                dtype=xp.float32,
                device=device,
            )
        )
        assert xp.all(
            measures
            == xp.asarray(
                [[3.0, 4.0], [1.0, 2.0]],
                dtype=xp.float32,
                device=device,
            )
        )
        assert xp.all(
            solution
            == xp.stack(
                (xp.ones(10, device=device), xp.zeros(10, device=device)),
                axis=0,
            )
        )
        assert xp.all(
            index
            == xp.asarray(
                [5, 3],
                dtype=xp.int32,
                device=device,
            )
        )
    elif return_type == "pandas":
        # In the case of pandas return types, everything should be converted to
        # NumPy before being passed to pandas.
        df = data
        assert (
            df.columns
            == [
                "objective",
                "measures_0",
                "measures_1",
                "solution_0",
                "solution_1",
                "solution_2",
                "solution_3",
                "solution_4",
                "solution_5",
                "solution_6",
                "solution_7",
                "solution_8",
                "solution_9",
                "index",
            ]
        ).all()
        assert (df.dtypes == [np.float32] * 13 + [np.int32]).all()
        assert len(df) == 2
        assert np.all(occupied == [True, True])
        assert np.all(df["objective"] == [2.0, 1.0])
        assert np.all(df["measures_0"] == [3.0, 1.0])
        assert np.all(df["measures_1"] == [4.0, 2.0])
        for i in range(10):
            assert np.all(df[f"solution_{i}"] == [1, 0])
        assert np.all(df["index"] == [5, 3])


@pytest.mark.parametrize("return_type", ["dict", "tuple", "pandas"])
def test_retrieve_custom_fields(store, return_type, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    occupied, data = store.retrieve(
        [5, 3], fields=["index", "objective"], return_type=return_type
    )

    if return_type == "dict":
        assert xp.all(
            occupied
            == xp.asarray(
                [True, True],
                dtype=bool,
                device=device,
            )
        )
        assert data.keys() == {"index", "objective"}
        assert xp.all(
            data["index"]
            == xp.asarray(
                [5, 3],
                dtype=xp.int32,
                device=device,
            )
        )
        assert xp.all(
            data["objective"]
            == xp.asarray(
                [2.0, 1.0],
                dtype=xp.float32,
                device=device,
            )
        )
    elif return_type == "tuple":
        assert xp.all(
            occupied
            == xp.asarray(
                [True, True],
                dtype=bool,
                device=device,
            )
        )
        assert xp.all(
            data[0]
            == xp.asarray(
                [5, 3],
                dtype=xp.int32,
                device=device,
            )
        )
        assert xp.all(
            data[1]
            == xp.asarray(
                [2.0, 1.0],
                dtype=xp.float32,
                device=device,
            )
        )
    elif return_type == "pandas":
        df = data
        assert (
            df.columns
            == [
                "index",
                "objective",
            ]
        ).all()
        assert (df.dtypes == [np.int32, np.float32]).all()
        assert len(df) == 2
        assert np.all(occupied == [True, True])
        assert np.all(df["index"] == [5, 3])
        assert np.all(df["objective"] == [2.0, 1.0])


def test_retrieve_single_field(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    occupied, data = store.retrieve([5, 3], fields="objective")

    assert xp.all(
        occupied
        == xp.asarray(
            [True, True],
            dtype=bool,
            device=device,
        )
    )
    assert xp.all(
        data
        == xp.asarray(
            [2.0, 1.0],
            dtype=xp.float32,
            device=device,
        )
    )


def test_resize_bad_capacity(store):
    with pytest.raises(ValueError):
        store.resize(store.capacity)


def test_resize_to_double_capacity(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    store.resize(store.capacity * 2)

    assert len(store) == 2
    assert xp.all(
        store.occupied
        == xp.asarray(
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=bool,
            device=device,
        )
    )
    assert xp.all(
        xp.sort(store.occupied_list)
        == xp.asarray(
            [3, 5],
            dtype=xp.int32,
            device=device,
        )
    )

    # Spot-check the fields.
    assert xp.all(
        store._fields["objective"][[3, 5]]
        == xp.asarray(
            [1.0, 2.0],
            dtype=xp.float32,
            device=device,
        )
    )


def test_data(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    d = store.data()

    assert d.keys() == {"objective", "measures", "solution", "index"}
    assert all(len(v) == 2 for v in d.values())

    row0 = xp.concat(
        (
            xp.asarray([1.0, 1.0, 2.0], dtype=xp.float32, device=device),
            xp.zeros(10, dtype=xp.float32, device=device),
            xp.asarray([3], dtype=xp.float32, device=device),
        )
    )
    row1 = xp.concat(
        (
            xp.asarray([2.0, 3.0, 4.0], dtype=xp.float32, device=device),
            xp.ones(10, dtype=xp.float32, device=device),
            xp.asarray([5], dtype=xp.float32, device=device),
        )
    )

    flat = [
        xp.concat(
            (
                d["objective"][i][None],
                d["measures"][i],
                d["solution"][i],
                d["index"][i][None],
            )
        )
        for i in range(2)
    ]

    # Either permutation.
    assert ((flat[0] == row0).all() and (flat[1] == row1).all()) or (
        (flat[0] == row1).all() and (flat[1] == row0).all()
    )


def test_data_with_tuple_return_type(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    d = store.data(return_type="tuple")

    assert len(d) == 4  # 3 fields and 1 index.
    assert all(len(v) == 2 for v in d)

    row0 = xp.concat(
        (
            xp.asarray([1.0, 1.0, 2.0], dtype=xp.float32, device=device),
            xp.zeros(10, dtype=xp.float32, device=device),
            xp.asarray([3], dtype=xp.float32, device=device),
        )
    )
    row1 = xp.concat(
        (
            xp.asarray([2.0, 3.0, 4.0], dtype=xp.float32, device=device),
            xp.ones(10, dtype=xp.float32, device=device),
            xp.asarray([5], dtype=xp.float32, device=device),
        )
    )

    flat = [
        xp.concat(
            (
                d[0][i][None],
                d[1][i],
                d[2][i],
                d[3][i][None],
            )
        )
        for i in range(2)
    ]

    # Either permutation.
    assert ((flat[0] == row0).all() and (flat[1] == row1).all()) or (
        (flat[0] == row1).all() and (flat[1] == row0).all()
    )


def test_data_with_pandas_return_type(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    df = store.data(return_type="pandas")

    assert (
        df.columns
        == [
            "objective",
            "measures_0",
            "measures_1",
            "solution_0",
            "solution_1",
            "solution_2",
            "solution_3",
            "solution_4",
            "solution_5",
            "solution_6",
            "solution_7",
            "solution_8",
            "solution_9",
            "index",
        ]
    ).all()
    assert (df.dtypes == [np.float32] * 13 + [np.int32]).all()
    assert len(df) == 2

    row0 = np.concat(([1.0, 1.0, 2.0], np.zeros(10), [3]))
    row1 = np.concat(([2.0, 3.0, 4.0], np.ones(10), [5]))

    # Either permutation.
    assert ((df.loc[0] == row0).all() and (df.loc[1] == row1).all()) or (
        (df.loc[0] == row1).all() and (df.loc[1] == row0).all()
    )


def test_iteration(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": xp.zeros((1, 10)),
        },
    )

    for entry in store:
        assert entry.keys() == {"objective", "measures", "solution", "index"}
        assert entry["objective"] == 1.0
        assert xp.all(
            entry["measures"]
            == xp.asarray(
                [1.0, 2.0],
                dtype=xp.float32,
                device=device,
            )
        )
        assert xp.all(
            entry["solution"]
            == xp.zeros(
                10,
                dtype=xp.float32,
                device=device,
            )
        )
        assert entry["index"] == 3


def test_add_during_iteration(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": xp.zeros((1, 10)),
        },
    )

    # Even with just one entry, adding during iteration should still raise an
    # error, just like it does in a set.
    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in store:
            store.add(
                [4],
                {
                    "objective": [2.0],
                    "measures": [[3.0, 4.0]],
                    "solution": xp.ones((1, 10)),
                },
            )


def test_clear_during_iteration(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": xp.zeros((1, 10)),
        },
    )

    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in store:
            store.clear()


def test_clear_and_add_during_iteration(store, xp_and_device):
    xp, device = xp_and_device

    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": xp.zeros((1, 10)),
        },
    )

    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in store:
            store.clear()
            store.add(
                [4],
                {
                    "objective": [2.0],
                    "measures": [[3.0, 4.0]],
                    "solution": xp.ones((1, 10)),
                },
            )


def test_picklable(xp_and_device):
    xp, device = xp_and_device

    store = ArrayStore(
        field_desc={
            "objective": ((), xp.float32),
            "measures": ((2,), xp.float32),
            "solution": ((10,), xp.float32),
        },
        capacity=10,
        xp=xp,
        device=device,
    )

    store.add(
        [3, 3],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": xp.stack((xp.zeros(10), xp.ones(10)), axis=0),
        },
    )

    # Copy the store into a new one by pickling and unpickling.
    pickled_str = pkl.dumps(store)
    store2 = pkl.loads(pickled_str)

    # Spot check a few properties.
    assert store.capacity == store2.capacity
    assert xp.all(store.occupied == store2.occupied)
    assert xp.all(store.occupied_list == store2.occupied_list)
    assert xp.all(store.data()["measures"] == store2.data()["measures"])
