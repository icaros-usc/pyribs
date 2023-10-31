"""Transform functions for ArrayStore."""
import itertools

import numpy as np

# TODO: Document signature above


# TODO: Rename
def transform_single(indices, new_data, add_info, occupied, cur_data):
    """Transform function for adding a single entry to an archive."""
    if len(indices) != 1:
        raise ValueError(
            "This method only supports adding single solutions, but "
            f"indices had a length of {len(indices)}.")

    dtype = add_info["dtype"]
    threshold_min = add_info["threshold_min"]
    learning_rate = add_info["learning_rate"]

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

    new_add_info = {"status": np.array([0])}  # NOT_ADDED
    # In the case where we want CMA-ME behavior, threshold_arr[index]
    # is -inf for new cells, which satisfies this if condition.
    if ((not was_occupied and threshold_min < objective) or
        (was_occupied and cur_threshold < objective)):
        if was_occupied:
            new_add_info["status"] = np.array([1])  # IMPROVE_EXISTING
        else:
            new_add_info["status"] = np.array([2])  # NEW

        # This calculation works in the case where threshold_min is -inf
        # because cur_threshold will be set to 0.0 instead.
        new_data["threshold"] = [
            (cur_threshold * (1.0 - learning_rate) + objective * learning_rate)
        ]
    new_add_info["value"] = np.array([objective - cur_threshold])

    if new_add_info["status"]:
        new_add_info["objective_sum"] = (add_info["objective_sum"] + objective -
                                         cur_objective)
        return indices, new_data, new_add_info
    else:
        new_add_info["objective_sum"] = add_info["objective_sum"]
        for k, v in new_data.items():
            new_data[k] = np.empty((0,) + v.shape[1:], v.dtype)
        new_data["threshold"] = np.empty(0, dtype)
        return np.empty(0, int), new_data, new_add_info
