"""Transform functions for :meth:`ribs.archives.ArrayStore.add`.

This module is still fairly unstable, hence why it is private. We may make it
public in the future once it becomes more stable.
"""
import numpy as np
from numpy_groupies import aggregate_nb as aggregate


def single_entry_with_threshold(indices, new_data, add_info, extra_args,
                                occupied, cur_data):
    """Transform function for adding a single entry.

    Assumptions:

    - ``indices`` and ``new_data`` have data for only one entry, e.g.,
      ``indices`` is length 1.
    - ``new_data`` has an ``"objective"`` field and needs a ``"threshold"``
      field.
    - ``extra_args`` contains ``"dtype"``, ``"threshold_min"``, and
      ``"learning_rate"`` entries.

    In short, this transform checks if the objective exceeds the current
    threshold, and if it does, it updates the threshold accordingly. There are
    also some special cases to handle CMA-ME (as opposed to CMA-MAE) -- this
    case corresponds to when ``threshold_min=-np.inf`` and ``learning_rate=1``.

    Since this transform operates on solutions one at a time, we do not
    recommend it when performance is critical. Instead, it is included as a
    relatively easy-to-modify example for users creating new archives.
    """
    if len(indices) != 1:
        raise ValueError("This transform only supports single solutions, but "
                         f"indices had a length of {len(indices)}.")

    dtype = extra_args["dtype"]  # e.g., np.float32 or np.float64
    threshold_min = extra_args["threshold_min"]  # scalar value
    learning_rate = extra_args["learning_rate"]  # scalar value

    cur_occupied = occupied[0]

    # Used for computing improvement value.
    cur_threshold = cur_data["threshold"][0]

    # New solutions require special settings for the threshold.
    if not cur_occupied:
        # If threshold_min is -inf, then we want CMA-ME behavior, which will
        # compute the improvement value w.r.t. zero for new solutions.
        # Otherwise, we will compute w.r.t. threshold_min.
        cur_threshold = (dtype(0)
                         if threshold_min == -np.inf else threshold_min)

    # Retrieve candidate objective.
    objective = new_data["objective"][0]

    # Compute status and threshold.
    add_info["status"] = np.array([0])  # NOT_ADDED
    # In the case where we want CMA-ME behavior, threshold_arr[index] is -inf
    # for new cells, which satisfies this if condition.
    if ((not cur_occupied and threshold_min < objective) or
        (cur_occupied and cur_threshold < objective)):
        if cur_occupied:
            add_info["status"] = np.array([1])  # IMPROVE_EXISTING
        else:
            add_info["status"] = np.array([2])  # NEW

        # This calculation works in the case where threshold_min is -inf because
        # cur_threshold will be set to 0.0 instead.
        new_data["threshold"] = [
            (cur_threshold * (1.0 - learning_rate) + objective * learning_rate)
        ]

    # Value is the improvement over the current threshold (can be negative).
    add_info["value"] = np.array([objective - cur_threshold])

    if add_info["status"]:
        return indices, new_data, add_info
    else:
        # new_data is ignored, so make it an empty dict.
        return np.array([], dtype=np.int32), {}, add_info


def _compute_thresholds(indices, objective, cur_threshold, learning_rate,
                        dtype):
    """Computes new thresholds.

    The indices, objective, and cur_threshold should all align. Based on these
    values, we will compute an array that holds the new threshold. The new array
    will have duplicate thresholds that correspond to duplicates in indices.
    """
    if len(indices) == 0:
        return np.array([], dtype=dtype)

    # Compute the number of objectives inserted into each cell. Note that we
    # index with `indices` to place the counts at all relevant indices. For
    # instance, if we had an array [1,2,3,1,5], we would end up with [2,1,1,2,1]
    # (there are 2 1's, 1 2, 1 3, 2 1's, and 1 5).
    #
    # All objective_sizes should be > 0 since we only retrieve counts for
    # indices in `indices`.
    objective_sizes = aggregate(indices, 1, func="len", fill_value=0)[indices]

    # Compute the sum of the objectives inserted into each cell -- again, we
    # index with `indices`.
    objective_sums = aggregate(indices,
                               objective,
                               func="sum",
                               fill_value=np.nan)[indices]

    # Update the threshold with the batch update rule from Fontaine 2022:
    # https://arxiv.org/abs/2205.10752
    #
    # Unlike in single_entry_with_threshold, we do not need to worry about
    # cur_threshold having -np.inf here as a result of threshold_min being
    # -np.inf. This is because the case with threshold_min = -np.inf is handled
    # separately since we compute the new threshold based on the max objective
    # in each cell in that case.
    ratio = dtype(1.0 - learning_rate)**objective_sizes
    new_threshold = (ratio * cur_threshold +
                     (objective_sums / objective_sizes) * (1 - ratio))

    return new_threshold


def batch_entries_with_threshold(indices, new_data, add_info, extra_args,
                                 occupied, cur_data):
    """Transform function for adding a batch of entries.

    Assumptions:

    - ``new_data`` has an ``"objective"`` field and needs a ``"threshold"``
      field.
    - ``extra_args`` contains ``"dtype"``, ``"threshold_min"``, and
      ``"learning_rate"`` entries.

    In short, this transform checks if the batch of solutions exceeds the
    current thresholds of their cells. Among those that exceed the threshold, we
    select the solution with the highest objective value. We also update the
    threshold based on the batch update rule for CMA-MAE:
    https://arxiv.org/abs/2205.10752

    We also handle some special cases for CMA-ME -- this case corresponds to
    when ``threshold_min=-np.inf`` and ``learning_rate=1``.
    """
    dtype = extra_args["dtype"]
    threshold_min = extra_args["threshold_min"]
    learning_rate = extra_args["learning_rate"]

    batch_size = len(indices)

    cur_threshold = cur_data["threshold"]
    cur_threshold[~occupied] = threshold_min  # Default to threshold_min.

    # Compute status -- arrays below are all boolean arrays of length
    # batch_size.
    #
    # In the case where we want CMA-ME behavior, the threshold defaults to -inf
    # for new cells, which satisfies the condition for can_insert.
    can_insert = new_data["objective"] > cur_threshold
    is_new = can_insert & ~occupied
    improve_existing = can_insert & occupied
    add_info["status"] = np.zeros(batch_size, dtype=np.int32)
    add_info["status"][is_new] = 2
    add_info["status"][improve_existing] = 1

    # If threshold_min is -inf, then we want CMA-ME behavior, which will compute
    # the improvement value of new solutions w.r.t zero. Otherwise, we will
    # compute improvement with respect to threshold_min.
    cur_threshold[is_new] = (dtype(0)
                             if threshold_min == -np.inf else threshold_min)
    add_info["value"] = new_data["objective"] - cur_threshold

    # Return early if we cannot insert anything -- continuing would actually
    # throw a ValueError in aggregate() since index[can_insert] would be empty.
    if not np.any(can_insert):
        return np.array([], dtype=np.int32), {}, add_info

    # Select all solutions that can be inserted -- at this point, there are
    # still conflicts in the insertions, e.g., multiple solutions can map to
    # index 0.
    indices = indices[can_insert]
    new_data = {name: arr[can_insert] for name, arr in new_data.items()}
    cur_threshold = cur_threshold[can_insert]

    # Compute the new threshold associated with each entry.
    if threshold_min == -np.inf:
        # Regular archive behavior, so the thresholds are just the objective.
        new_threshold = new_data["objective"]
    else:
        # Batch threshold update described in Fontaine 2022
        # https://arxiv.org/abs/2205.10752 This computation is based on the mean
        # objective of all solutions in the batch that could have been inserted
        # into each cell.
        new_threshold = _compute_thresholds(indices, new_data["objective"],
                                            cur_threshold, learning_rate, dtype)

    # Retrieve indices of solutions that should be inserted into the archive.
    # Currently, multiple solutions may be inserted at each archive index, but
    # we only want to insert the maximum among these solutions. Thus, we obtain
    # the argmax for each archive index.
    #
    # We use a fill_value of -1 to indicate archive indices that were not
    # covered in the batch. Note that the length of archive_argmax is only
    # max(indices), rather than the total number of grid cells. However, this is
    # okay because we only need the indices of the solutions, which we store in
    # should_insert.
    #
    # aggregate() always chooses the first item if there are ties, so the first
    # elite will be inserted if there is a tie. See their default numpy
    # implementation for more info:
    # https://github.com/ml31415/numpy-groupies/blob/master/numpy_groupies/aggregate_numpy.py#L107
    archive_argmax = aggregate(indices,
                               new_data["objective"],
                               func="argmax",
                               fill_value=-1)
    should_insert = archive_argmax[archive_argmax != -1]

    # Select only solutions that will be inserted into the archive.
    indices = indices[should_insert]
    new_data = {name: arr[should_insert] for name, arr in new_data.items()}
    new_data["threshold"] = new_threshold[should_insert]

    return indices, new_data, add_info


def compute_objective_sum(indices, new_data, add_info, extra_args, occupied,
                          cur_data):
    """Computes the new sum of objectives after inserting ``new_data``.

    Assumptions:
    - ``new_data`` and ``cur_data`` have an ``"objective"`` field.
    - ``extra_args`` contains ``"objective_sum"``, the current sum of
      objectives.

    The new sum of objectives will be added to ``add_info`` with the key
    ``"objective_sum"``.

    This transform should be placed near the end of a chain of transforms so
    that it only considers solutions that are going to be inserted into the
    store.
    """
    cur_objective_sum = extra_args["objective_sum"]
    if len(indices) == 0:
        add_info["objective_sum"] = cur_objective_sum
    else:
        cur_objective = cur_data["objective"]
        cur_objective[~occupied] = 0.0  # Unoccupied objectives should be 0.
        add_info["objective_sum"] = (
            cur_objective_sum + np.sum(new_data["objective"] - cur_objective))
    return indices, new_data, add_info


def compute_best_index(indices, new_data, add_info, extra_args, occupied,
                       cur_data):
    """Identifies the index of the best solution among those in new_data.

    Assumptions:

    - ``new_data`` has an ``"objective"`` field.
    - The best solution will be the one with the highest objective value.

    The best index will be added to the ``add_info`` dict with the key
    ``"best_index"``. If there is no best index, then ``"best_index"`` will be
    None.

    This transform should be placed near the end of a chain of transforms so
    that it only considers solutions that are going to be inserted into the
    store.
    """
    # pylint: disable = unused-argument

    if len(indices) == 0:
        add_info["best_index"] = None
    else:
        item_idx = np.argmax(new_data["objective"])
        add_info["best_index"] = indices[item_idx]

    return indices, new_data, add_info
