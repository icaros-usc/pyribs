"""Transform functions for :meth:`ribs.archives.ArrayStore.add`.

This module is still fairly unstable, hence why it is private. We may make it
public in the future once it becomes more stable.
"""
import numpy as np
from numpy_groupies import aggregate_nb as aggregate


def compute_best_index(indices, new_data, add_info, extra_args, occupied,
                       cur_data):
    """Identifies the index of the best solution among those in new_data.

    This method makes the following assumptions:

    - ``new_data`` has an ``"objective"`` field
    - The best solution will be the one with the highest objective value.

    The best index will be added to the ``add_info`` dict with the key
    ``"best_index"``. If there is no best index, then ``"best_index"`` will be
    None.

    This method should be placed near the end of a chain of transforms so that
    it only considers solutions that are going to be inserted into the store.
    """
    # pylint: disable = unused-argument

    if len(indices) == 0:
        add_info["best_index"] = None
    else:
        item_idx = np.argmax(new_data["objective"])
        add_info["best_index"] = indices[item_idx]

    return indices, new_data, add_info


# TODO: Generalize to further fields
# TODO: Rename
# TODO: Tidy up code
def transform_single(indices, new_data, add_info, extra_args, occupied,
                     cur_data):
    """Transform function for adding a single entry to an archive."""
    if len(indices) != 1:
        raise ValueError(
            "This method only supports adding single solutions, but "
            f"indices had a length of {len(indices)}.")

    dtype = extra_args["dtype"]
    threshold_min = extra_args["threshold_min"]
    learning_rate = extra_args["learning_rate"]

    was_occupied = occupied[0]
    objective = new_data["objective"][0]

    # Only used for computing QD score.
    cur_objective = cur_data["objective"][0]

    # Used for computing improvement value.
    cur_threshold = cur_data["threshold"][0]

    # New solutions require special settings for cur_objective and
    # cur_threshold.
    if not was_occupied:
        cur_objective = dtype(0)
        # If threshold_min is -inf, then we want CMA-ME behavior, which will
        # compute the improvement value w.r.t. zero for new solutions.
        # Otherwise, we will compute w.r.t. threshold_min.
        cur_threshold = (dtype(0)
                         if threshold_min == -np.inf else threshold_min)

    add_info["status"] = 0  # NOT_ADDED
    # In the case where we want CMA-ME behavior, threshold_arr[index] is -inf
    # for new cells, which satisfies this if condition.
    if ((not was_occupied and threshold_min < objective) or
        (was_occupied and cur_threshold < objective)):
        if was_occupied:
            add_info["status"] = 1  # IMPROVE_EXISTING
        else:
            add_info["status"] = 2  # NEW

        # This calculation works in the case where threshold_min is -inf because
        # cur_threshold will be set to 0.0 instead.
        new_data["threshold"] = [
            (cur_threshold * (1.0 - learning_rate) + objective * learning_rate)
        ]
    add_info["value"] = objective - cur_threshold

    if add_info["status"]:
        add_info["objective_sum"] = (extra_args["objective_sum"] + objective -
                                     cur_objective)
        return indices, new_data, add_info
    else:
        add_info["objective_sum"] = extra_args["objective_sum"]
        # new_data is ignored, so make it an empty dict.
        return np.array([]), {}, add_info


def _compute_thresholds(indices, objective, cur_threshold, learning_rate,
                        dtype):
    """Computes new thresholds.

    The indices, objective, and cur_threshold should all align. Based on these
    values, we will compute an array that holds the new threshold. It is
    expected that the new array will have duplicate thresholds that correspond
    to duplicates in indices.
    """
    # Even though we do this check, it should not be possible to have empty
    # objective_batch or index_batch in the add() method since we check that at
    # least one cell is being updated by seeing if can_insert has any True
    # values.
    if objective.size == 0 or indices.size == 0:
        return np.array([], dtype=dtype), np.array([], dtype=bool)

    # Compute the number of objectives inserted into each cell. Note that we
    # index with `indices` to place the counts at all relevant indices. For
    # instance, if we had an array [1,2,3,1,5], we would end up with [2,1,1,2,1]
    # (there are 2 1's, 1 2, 1 3, 2 1's, and 1 5).
    objective_sizes = aggregate(indices, 1, func="len", fill_value=0)[indices]

    # Note: All objective_sizes should be > 0 since we do only retrieve counts
    # for indices in `indices`.

    # TODO: remove
    assert np.all(objective_sizes > 0)

    # Compute the sum of the objectives inserted into each cell -- again, we
    # index with `indices`.
    objective_sums = aggregate(indices,
                               objective,
                               func="sum",
                               fill_value=np.nan)[indices]

    # Unlike in add_single, we do not need to worry about old_threshold having
    # -np.inf here as a result of threshold_min being -np.inf. This is because
    # the case with threshold_min = -np.inf is handled separately since we
    # compute the new threshold based on the max objective in each cell in that
    # case.

    ratio = dtype(1.0 - learning_rate)**objective_sizes
    new_threshold = (ratio * cur_threshold +
                     (objective_sums / objective_sizes) * (1 - ratio))

    return new_threshold


def transform_batch(indices, new_data, add_info, extra_args, occupied,
                    cur_data):
    """Transform function for adding a batch of entries to an archive."""
    dtype = extra_args["dtype"]
    threshold_min = extra_args["threshold_min"]
    learning_rate = extra_args["learning_rate"]

    batch_size = len(indices)

    ## Step 1: Compute status and value ##

    # Copy old objectives since we will be modifying the objectives storage.
    cur_objective = np.copy(cur_data["objective"])
    cur_threshold = np.copy(cur_data["threshold"])
    cur_threshold[~occupied] = threshold_min  # Default to threshold_min.

    # Compute status -- arrays below are all boolean arrays of length
    # batch_size.
    #
    # In the case where we want CMA-ME behavior, the threshold defaults to -inf
    # for new cells, which satisfies the condition for can_be_added.
    can_be_added = new_data["objective"] > cur_threshold
    is_new = can_be_added & ~occupied
    improve_existing = can_be_added & occupied
    add_info["status"] = np.zeros(batch_size, dtype=np.int32)
    add_info["status"][is_new] = 2
    add_info["status"][improve_existing] = 1

    # New solutions require special settings for cur_objective and
    # old_threshold.
    cur_objective[is_new] = dtype(0)

    # If threshold_min is -inf, then we want CMA-ME behavior, which will compute
    # the improvement value of new solutions w.r.t zero. Otherwise, we will
    # compute w.r.t. threshold_min.
    cur_threshold[is_new] = (dtype(0)
                             if threshold_min == -np.inf else threshold_min)
    add_info["value"] = new_data["objective"] - cur_threshold

    ## Step 2: Insert solutions into archive. ##

    # Return early if we cannot insert anything -- continuing would actually
    # throw a ValueError in aggregate() since index[can_insert] would be empty.
    can_insert = is_new | improve_existing
    if not np.any(can_insert):
        add_info["objective_sum"] = extra_args["objective_sum"]
        return np.array([]), {}, add_info

    # Select only solutions that can be inserted into the archive.
    index_can = indices[can_insert]
    solution_can = new_data["solution"][can_insert]
    objective_can = new_data["objective"][can_insert]
    measures_can = new_data["measures"][can_insert]
    metadata_can = new_data["metadata"][can_insert]
    cur_threshold_can = cur_threshold[can_insert]
    cur_objective_can = cur_objective[can_insert]

    # Retrieve indices of solutions that should be inserted into the archive.
    # Currently, multiple solutions may be inserted at each archive index, but
    # we only want to insert the maximum among these solutions. Thus, we obtain
    # the argmax for each archive index.
    #
    # We use a fill_value of -1 to indicate archive indices which were not
    # covered in the batch. Note that the length of archive_argmax is only
    # max(index[can_insert]), rather than the total number of grid cells.
    # However, this is okay because we only need the indices of the solutions,
    # which we store in should_insert.
    #
    # aggregate() always chooses the first item if there are ties, so the first
    # elite will be inserted if there is a tie. See their default numpy
    # implementation for more info:
    # https://github.com/ml31415/numpy-groupies/blob/master/numpy_groupies/aggregate_numpy.py#L107
    archive_argmax = aggregate(index_can,
                               objective_can,
                               func="argmax",
                               fill_value=-1)
    should_insert = archive_argmax[archive_argmax != -1]

    # Select only solutions that will be inserted into the archive.
    indices = index_can[should_insert]
    new_data = {
        "solution": solution_can[should_insert],
        "objective": objective_can[should_insert],
        "measures": measures_can[should_insert],
        "metadata": metadata_can[should_insert],
    }
    cur_objective_insert = cur_objective_can[should_insert]

    # Update the thresholds.
    #
    # i.e., compute cur_threshold_insert / new_data["threshold"]
    if threshold_min == -np.inf:
        # Here we want regular archive behavior, so the thresholds should just
        # be the maximum objective.
        new_data["threshold"] = new_data["objective"]
    else:
        # Here we compute the batch threshold update described in the appendix
        # of Fontaine 2022 https://arxiv.org/abs/2205.10752 This computation is
        # based on the mean objective of all solutions in the batch that could
        # have been inserted into each cell. This method is separated out to
        # facilitate testing.
        new_threshold_can = _compute_thresholds(index_can, objective_can,
                                                cur_threshold_can,
                                                learning_rate, dtype)
        new_data["threshold"] = new_threshold_can[should_insert]

    ## Step 3: Update archive stats. ##

    # Since we set the new solutions in the old objective batch to have value
    # 0.0, the objectives for new solutions are added in properly here.
    add_info["objective_sum"] = extra_args["objective_sum"] + np.sum(
        new_data["objective"] - cur_objective_insert)

    return indices, new_data, add_info
