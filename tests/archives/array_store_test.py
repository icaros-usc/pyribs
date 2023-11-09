"""Tests for ArrayStore."""
import numpy as np
import pytest

from ribs.archives import ArrayStore

# pylint: disable = redefined-outer-name


def test_init_reserved_field():
    with pytest.raises(ValueError):
        ArrayStore(
            {
                "index": ((), np.float32),
            },
            10,
        )


def test_init_invalid_field():
    with pytest.raises(ValueError):
        ArrayStore(
            {
                # The space makes this an invalid identifier.
                "foo bar": ((), np.float32),
            },
            10,
        )


@pytest.mark.parametrize("shape", [((), (2,), (10,)), ((), 2, 10)],
                         ids=["tuple", "int"])
def test_init(shape):
    capacity = 10
    store = ArrayStore(
        {
            "objective": (shape[0], np.float32),
            "measures": (shape[1], np.float32),
            "solution": (shape[2], np.float32),
        },
        capacity,
    )

    assert len(store) == 0
    assert store.capacity == capacity
    assert np.all(~store.occupied)
    assert len(store.occupied_list) == 0
    assert store.field_desc == {
        "objective": (shape[0], np.float32),
        "measures": (
            (shape[1],) if isinstance(shape[1], int) else shape[1], np.float32),
        "solution": (
            (shape[2],) if isinstance(shape[2], int) else shape[2], np.float32),
    }
    assert store.field_list == ["objective", "measures", "solution"]


@pytest.fixture
def store():
    """Simple ArrayStore for testing."""
    return ArrayStore(
        field_desc={
            "objective": ((), np.float32),
            "measures": ((2,), np.float32),
            "solution": ((10,), np.float32),
        },
        capacity=10,
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
            {},  # Empty extra_args.
            [],  # Empty transforms.
        )


def test_add_mismatch_indices(store):
    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0, 3.0],  # Length 3 instead of 2.
                "measures": [[1.0, 2.0], [3.0, 4.0]],
                "solution": [np.zeros(10), np.ones(10)],
            },
            {},  # Empty extra_args.
            [],  # Empty transforms.
        )


def test_simple_add_retrieve_clear(store):
    """Add without transforms, retrieve the data, and clear the archive."""
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    assert len(store) == 2
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert np.all(np.sort(store.occupied_list) == [3, 5])

    occupied, data = store.retrieve([5, 3])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution", "index"])
    assert np.all(data["objective"] == [2.0, 1.0])
    assert np.all(data["measures"] == [[3.0, 4.0], [1.0, 2.0]])
    assert np.all(data["solution"] == [np.ones(10), np.zeros(10)])
    assert np.all(data["index"] == [5, 3])

    store.clear()

    assert len(store) == 0
    assert np.all(store.occupied == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert len(store.occupied_list) == 0


def test_add_duplicate_indices(store):
    store.add(
        [3, 3],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    assert len(store) == 1
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert np.all(store.occupied_list == [3])


def test_dtypes(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    _, data = store.retrieve([5, 3])

    # Index is always int32, and other fields were defined as float32 in the
    # `store` fixture.
    assert data["objective"].dtype == np.float32
    assert data["measures"].dtype == np.float32
    assert data["solution"].dtype == np.float32
    assert data["index"].dtype == np.int32


def test_retrieve_duplicate_indices(store):
    store.add(
        [3],
        {
            "objective": [2.0],
            "measures": [[3.0, 4.0]],
            "solution": [np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    occupied, data = store.retrieve([3, 3])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution", "index"])
    assert np.all(data["objective"] == [2.0, 2.0])
    assert np.all(data["measures"] == [[3.0, 4.0], [3.0, 4.0]])
    assert np.all(data["solution"] == [np.ones(10), np.ones(10)])
    assert np.all(data["index"] == [3, 3])


def test_retrieve_invalid_fields(store):
    with pytest.raises(ValueError):
        store.retrieve([0, 1], fields=["objective", "foo"])


def test_retrieve_invalid_return_type(store):
    with pytest.raises(ValueError):
        store.retrieve([0, 1], return_type="foo")


def test_retrieve_pandas_2d_fields(store):
    store = ArrayStore(
        {
            "solution": ((10, 10), np.float32),
        },
        10,
    )
    with pytest.raises(ValueError):
        store.retrieve([], return_type="pandas")


@pytest.mark.parametrize("return_type", ["dict", "tuple", "pandas"])
def test_retrieve(return_type, store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    occupied, data = store.retrieve([5, 3], return_type=return_type)

    if return_type == "dict":
        assert np.all(occupied == [True, True])
        assert data.keys() == set(
            ["objective", "measures", "solution", "index"])
        assert np.all(data["objective"] == [2.0, 1.0])
        assert np.all(data["measures"] == [[3.0, 4.0], [1.0, 2.0]])
        assert np.all(data["solution"] == [np.ones(10), np.zeros(10)])
        assert np.all(data["index"] == [5, 3])
    elif return_type == "tuple":
        objective, measures, solution, index = data
        assert np.all(occupied == [True, True])
        assert np.all(objective == [2.0, 1.0])
        assert np.all(measures == [[3.0, 4.0], [1.0, 2.0]])
        assert np.all(solution == [np.ones(10), np.zeros(10)])
        assert np.all(index == [5, 3])
    elif return_type == "pandas":
        df = data
        assert (df.columns == [
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
        ]).all()
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
def test_retrieve_custom_fields(store, return_type):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    occupied, data = store.retrieve([5, 3],
                                    fields=["index", "objective"],
                                    return_type=return_type)

    if return_type == "dict":
        assert np.all(occupied == [True, True])
        assert data.keys() == set(["index", "objective"])
        assert np.all(data["index"] == [5, 3])
        assert np.all(data["objective"] == [2.0, 1.0])
    elif return_type == "tuple":
        assert np.all(occupied == [True, True])
        assert np.all(data[0] == [5, 3])
        assert np.all(data[1] == [2.0, 1.0])
    elif return_type == "pandas":
        df = data
        assert (df.columns == [
            "index",
            "objective",
        ]).all()
        assert (df.dtypes == [np.int32, np.float32]).all()
        assert len(df) == 2
        assert np.all(occupied == [True, True])
        assert np.all(df["index"] == [5, 3])
        assert np.all(df["objective"] == [2.0, 1.0])


def test_retrieve_single_field(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    occupied, data = store.retrieve([5, 3], fields="objective")

    assert np.all(occupied == [True, True])
    assert np.all(data == [2.0, 1.0])


def test_add_simple_transform(store):

    def obj_meas(indices, new_data, add_info, extra_args, occupied, cur_data):
        # pylint: disable = unused-argument
        new_data["objective"] = np.sum(new_data["solution"], axis=1)
        new_data["measures"] = np.asarray(new_data["solution"])[:, :2]
        add_info.update(extra_args)
        add_info["bar"] = 5
        return indices, new_data, add_info

    add_info = store.add(
        [3, 5],
        {
            "solution": [np.ones(10), 2 * np.ones(10)],
        },
        {"foo": 4},
        [obj_meas],
    )

    assert add_info == {"foo": 4, "bar": 5}

    assert len(store) == 2
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert np.all(np.sort(store.occupied_list) == [3, 5])

    occupied, data = store.retrieve([3, 5])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution", "index"])
    assert np.all(data["objective"] == [10.0, 20.0])
    assert np.all(data["measures"] == [[1.0, 1.0], [2.0, 2.0]])
    assert np.all(data["solution"] == [np.ones(10), 2 * np.ones(10)])
    assert np.all(data["index"] == [3, 5])


def test_add_empty_transform(store):
    # new_data should be able to take on arbitrary values when no indices are
    # returned, so we make it an empty dict here.
    def empty(indices, new_data, add_info, extra_args, occupied, cur_data):
        # pylint: disable = unused-argument
        return [], {}, {}

    add_info = store.add(
        [3, 5],
        {
            "solution": [np.ones(10), 2 * np.ones(10)],
        },
        {"foo": 4},
        [empty],
    )

    assert add_info == {}

    assert len(store) == 0
    assert np.all(~store.occupied)
    assert len(store.occupied_list) == 0


def test_resize_bad_capacity(store):
    with pytest.raises(ValueError):
        store.resize(store.capacity)


def test_resize_to_double_capacity(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    store.resize(store.capacity * 2)

    assert len(store) == 2
    assert np.all(store.occupied ==
                  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.all(np.sort(store.occupied_list) == [3, 5])

    # Spot-check the fields.
    assert np.all(store._fields["objective"][[3, 5]] == [1.0, 2.0])


def test_as_raw_dict(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    d = store.as_raw_dict()

    assert d.keys() == set([
        "props.capacity",
        "props.occupied",
        "props.n_occupied",
        "props.occupied_list",
        "props.updates",
        "fields.objective",
        "fields.measures",
        "fields.solution",
    ])
    assert d["props.capacity"] == 10
    assert np.all(d["props.occupied"] == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert d["props.n_occupied"] == 2
    assert np.all(np.sort(d["props.occupied_list"][:2]) == [3, 5])
    assert np.all(d["props.updates"] == [1, 0])  # 1 add, 0 clear.
    assert np.all(d["fields.objective"][[3, 5]] == [1.0, 2.0])
    assert np.all(d["fields.measures"][[3, 5]] == [[1.0, 2.0], [3.0, 4.0]])
    assert np.all(d["fields.solution"][[3, 5]] == [np.zeros(10), np.ones(10)])


def test_from_raw_dict_invalid_props(store):
    d = store.as_raw_dict()
    del d["props.capacity"]
    with pytest.raises(ValueError):
        ArrayStore.from_raw_dict(d)


def test_from_raw_dict(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    new_store = ArrayStore.from_raw_dict(store.as_raw_dict())

    assert len(new_store) == 2
    assert np.all(new_store.occupied == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert np.all(np.sort(new_store.occupied_list) == [3, 5])

    occupied, data = new_store.retrieve([5, 3])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution", "index"])
    assert np.all(data["objective"] == [2.0, 1.0])
    assert np.all(data["measures"] == [[3.0, 4.0], [1.0, 2.0]])
    assert np.all(data["solution"] == [np.ones(10), np.zeros(10)])
    assert np.all(data["index"] == [5, 3])


def test_data(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    d = store.data()

    assert d.keys() == set(["objective", "measures", "solution", "index"])
    assert all(len(v) == 2 for v in d.values())

    row0 = np.concatenate(([1.0, 1.0, 2.0], np.zeros(10), [3]))
    row1 = np.concatenate(([2.0, 3.0, 4.0], np.ones(10), [5]))

    flat = [
        np.concatenate(([d["objective"][i]], d["measures"][i], d["solution"][i],
                        [d["index"][i]])) for i in range(2)
    ]

    # Either permutation.
    assert (((flat[0] == row0).all() and (flat[1] == row1).all()) or
            ((flat[0] == row1).all() and (flat[1] == row0).all()))


def test_data_with_tuple_return_type(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    d = store.data(return_type="tuple")

    assert len(d) == 4  # 3 fields and 1 index.
    assert all(len(v) == 2 for v in d)

    row0 = np.concatenate(([1.0, 1.0, 2.0], np.zeros(10), [3]))
    row1 = np.concatenate(([2.0, 3.0, 4.0], np.ones(10), [5]))

    flat = [
        np.concatenate(([d[0][i]], d[1][i], d[2][i], [d[3][i]]))
        for i in range(2)
    ]

    # Either permutation.
    assert (((flat[0] == row0).all() and (flat[1] == row1).all()) or
            ((flat[0] == row1).all() and (flat[1] == row0).all()))


def test_data_with_pandas_return_type(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    df = store.data(return_type="pandas")

    assert (df.columns == [
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
    ]).all()
    assert (df.dtypes == [np.float32] * 13 + [np.int32]).all()
    assert len(df) == 2

    row0 = np.concatenate(([1.0, 1.0, 2.0], np.zeros(10), [3]))
    row1 = np.concatenate(([2.0, 3.0, 4.0], np.ones(10), [5]))

    # Either permutation.
    assert (((df.loc[0] == row0).all() and (df.loc[1] == row1).all()) or
            ((df.loc[0] == row1).all() and (df.loc[1] == row0).all()))


def test_iteration(store):
    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": [np.zeros(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    for entry in store:
        assert entry.keys() == set(
            ["objective", "measures", "solution", "index"])
        assert np.all(entry["objective"] == [1.0])
        assert np.all(entry["measures"] == [[1.0, 2.0]])
        assert np.all(entry["solution"] == [np.zeros(10)])
        assert np.all(entry["index"] == [3])


def test_add_during_iteration(store):
    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": [np.zeros(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    # Even with just one entry, adding during iteration should still raise an
    # error, just like it does in a set.
    with pytest.raises(RuntimeError):
        for _ in store:
            store.add(
                [4],
                {
                    "objective": [2.0],
                    "measures": [[3.0, 4.0]],
                    "solution": [np.ones(10)],
                },
                {},  # Empty extra_args.
                [],  # Empty transforms.
            )


def test_clear_during_iteration(store):
    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": [np.zeros(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    with pytest.raises(RuntimeError):
        for _ in store:
            store.clear()


def test_clear_and_add_during_iteration(store):
    store.add(
        [3],
        {
            "objective": [1.0],
            "measures": [[1.0, 2.0]],
            "solution": [np.zeros(10)],
        },
        {},  # Empty extra_args.
        [],  # Empty transforms.
    )

    with pytest.raises(RuntimeError):
        for _ in store:
            store.clear()
            store.add(
                [4],
                {
                    "objective": [2.0],
                    "measures": [[3.0, 4.0]],
                    "solution": [np.ones(10)],
                },
                {},  # Empty extra_args.
                [],  # Empty transforms.
            )
