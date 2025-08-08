"""Provides the BayesianOptimizationEmitter."""

import warnings

import numpy as np
from scipy.stats import entropy, norm
from scipy.stats.qmc import Sobol
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ribs._utils import check_batch_shape, check_finite, validate_batch
from ribs.archives import GridArchive
from ribs.emitters._emitter_base import EmitterBase


class BayesianOptimizationEmitter(EmitterBase):
    """A sample-efficient emitter that models objective and measure functions with
    Gaussian process surrogate models.

    Bayesian Optimisation is used to emit solutions that are predicted to have high
    *Expected Joint Improvement of Elites* (EJIE) acquisition values. Refer to `Kent
    2024 <https://ieeexplore.ieee.org/abstract/document/10472301>`_ for more
    information.

    .. note::

        This emitter requires the `pymoo <https://pymoo.org>`_ package, which can be
        installed with ``pip install pymoo`` or ``conda install pymoo``.

    Args:
        archive (ribs.archives.GridArchive): An archive to use when creating and
            inserting solutions. Currently, the only supported archive type is
            :class:`ribs.archives.GridArchive`.
        bounds (array-like): Bounds of the solution space. Pass an array-like to specify
            the bounds for each dim. Each element in this array-like must be a tuple of
            ``(lower_bound, upper_bound)``. Cannot be ``None`` or ``+-inf`` because
            SOBOL sampling is used.
        search_nrestarts (int): Number of starting points for EJIE pattern search.
        entropy_ejie (bool): If ``True``, augments EJIE acquisition function with
            entropy to encourage measure space exploration. Refer to Sec. 4.1 of `Kent
            2023 <https://dl.acm.org/doi/10.1145/3583131.3590486>`_ for more details.
        upscale_schedule (array-like): An array of increasing archive resolutions
            starting with :attr:`archive.resolution` and ending with the user's intended
            final archive resolution. This will upscale the archive to the next
            scheduled resolution if every cell within the current archive has been
            filled, or the number of evaluated solutions is more than twice
            :attr:`archive.cells`. If ``None``, the archive will not be upscaled.
        min_obj (float or int): The lowest possible objective value. Serves as the
            default objective value within archive cells that have not been filled.
            Mainly used when computing expected improvement.
        num_initial_samples (int): The number of solutions that will be sampled from a
            Sobol sequence as the first batch of training data for gaussian processes.
            Either ``num_initial_samples`` or ``initial_solutions`` must be set.
        initial_solutions (array-like): An (n, solution_dim) array of solutions to be
            used as the first batch of training data for gaussian processes. Either
            ``num_initial_samples`` or ``initial_solutions`` must be set.
        batch_size (int): Number of solutions to return in :meth:`ask`. Must not exceed
            ``search_nrestarts``. It is recommended to set this to 1 for sample
            efficiency.
        seed (int): Seed for the random number generator.
    """

    def __init__(
        self,
        archive,
        bounds,
        *,
        search_nrestarts=5,
        entropy_ejie=False,
        upscale_schedule=None,
        min_obj=0,
        num_initial_samples=None,
        initial_solutions=None,
        batch_size=1,
        seed=None,
    ):
        try:
            # pylint: disable = import-outside-toplevel
            from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
            from pymoo.optimize import minimize
            from pymoo.problems.functional import FunctionalProblem
            from pymoo.termination.default import DefaultSingleObjectiveTermination
        except ImportError as e:
            raise ImportError(
                "pymoo must be installed -- please run `pip install pymoo` "
                "or `conda install pymoo`"
            ) from e
        self._pymoo_mods = {
            "PatternSearch": PatternSearch,
            "minimize": minimize,
            "FunctionalProblem": FunctionalProblem,
            "DefaultSingleObjectiveTermination": DefaultSingleObjectiveTermination,
        }

        check_finite(bounds, "bounds")
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        if not isinstance(archive, GridArchive):
            raise NotImplementedError(
                f"archive type {archive.__class__.__name__} not implemented for"
                " BayesianOptimizationEmitter. Expected GridArchive."
            )

        if (upscale_schedule is not None) and (
            not np.isclose(archive.learning_rate, 1)
        ):
            raise NotImplementedError(
                "Archive upscaling is currently incompatible with archive "
                "learning rate. Since you have specified an upscale schedule "
                f"{upscale_schedule}, the learning rate of the input archive "
                f"must be 1 (currently {archive.learning_rate})."
            )

        self._seed = seed
        self._sobol = Sobol(d=self.solution_dim, scramble=True, seed=self._seed)

        # Initializes a multi-output GP. 1 output for objective function, plus 1
        # output for each measure function
        # NOTE: Using Matern kernal with default parameters
        self._gp = GaussianProcessRegressor(
            kernel=Matern(), normalize_y=True, n_targets=1 + self.measure_dim
        )

        if num_initial_samples is None and initial_solutions is None:
            raise ValueError(
                "Either num_initial_samples or initial_solutions must be provided."
            )
        if num_initial_samples is not None and initial_solutions is not None:
            raise ValueError(
                "num_initial_samples and initial_solutions cannot both be provided."
            )

        if initial_solutions is not None:
            self._initial_solutions = np.asarray(
                initial_solutions, dtype=archive.dtypes["solution"]
            )
        else:
            self._initial_solutions = self._sample_n_rescale(num_initial_samples)

        check_batch_shape(
            self._initial_solutions,
            "initial_solutions",
            archive.solution_dim,
            "archive.solution_dim",
        )

        self._dataset = {
            "solution": np.empty((0, self.solution_dim), dtype=self.dtype),
            "objective": np.empty((0, 1)),
            "measures": np.empty((0, self.measure_dim)),
        }

        self._search_nrestarts = search_nrestarts

        if upscale_schedule is None:
            self._upscale_schedule = None
        else:
            self._upscale_schedule = np.asarray(upscale_schedule)
            self._check_upscale_schedule(self._upscale_schedule)

        self._batch_size = batch_size

        self._misspec = 0
        self._overspec = 0
        self._prev_numcells = len(self.archive)
        self._numitrs_noprogress = 0

        self._entropy_norm = (
            entropy(np.ones(self.archive.cells) / self.archive.cells)
            if entropy_ejie
            else None
        )

        self._min_obj = min_obj

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def cell_prob_cutoff(self):
        """np.float64: Cutoff value (ohm) for :meth:`_get_cell_probs` as described in
        `Kent 2024 <https://ieeexplore.ieee.org/abstract /document/10472301>`_ Sec.IV-D.
        There are some numerical errors involved with cell_probs, so even passing the
        same sample in different shapes/contexts can sometimes return slightly different
        cell_probs, so we return cell_prob_cutoff at a lower precision than cell_probs
        to ensure the same sample consistently passes/fails the threshold check."""
        return round(
            0.5
            * (2 / self.archive.cells)
            ** (
                (10 * self.solution_dim)
                / (self._misspec - 2 * self._overspec + self.num_evals + 1e-6)
            )
            ** 0.5,
            4,
        )

    @property
    def num_evals(self):
        """int: Number of solutions stored in :attr:`_dataset`.

        This is the number of solutions that have been evaluated since the
        initialization of this emitter.
        """
        return self._dataset["solution"].shape[0]

    @property
    def measure_dim(self):
        """int: Number of measure functions."""
        return self.archive.measure_dim

    @property
    def num_sobol_samples(self):
        """int: Number of SOBOL samples to draw when choosing pattern search starting
        points in :meth:`ask`.

        NOTE: If measure function gradients are available, a potentially better way to
        do this might be to do Latin Hypercube sampling within measure space, and then
        use measure gradients to find solutions achieving those measure space samples.
        See `Kent 2024b
        <https://wrap.warwick.ac.uk/id/eprint/189556/1/WRAP_Theses_Kent_2024.pdf>`_ Sec.
        6.3 for more details.
        """
        m = 10 if self.solution_dim < 2 else 1
        return np.clip(
            m * (self.solution_dim**2) * np.prod(self.measure_dim),
            10000,
            100000,
        )

    @property
    def dtype(self):
        """numpy.dtype: Data type of solutions."""
        return self.archive.dtypes["solution"]

    @property
    def upscale_schedule(self):
        """np.ndarray: The archive upscale schedule defined by user when initializing
        this emitter."""
        return self._upscale_schedule

    @property
    def upscale_trigger_threshold(self):
        """int: The maximum number of iterations the emitter is allowed to not find new
        cells before archive upscale is triggered. See `this code
        <https://github.com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD_beta.py#L187>`_
        for more details."""
        return np.floor(np.sqrt(self.archive.cells))

    @property
    def min_obj(self):
        """float or int: The lowest possible objective value. Refer to the documentation
        for this class."""
        return self._min_obj

    @property
    def initial_solutions(self):
        """numpy.ndarray: The initial solutions which are returned when the archive is
        empty (if x0 is not set)."""
        return self._initial_solutions

    @EmitterBase.archive.setter
    def archive(self, new_archive):
        """Allows resetting the archive associated with this emitter (for archive
        upscaling)."""
        self._archive = new_archive

    def post_upscale_updates(self):
        """After the scheduler upscales the archive, updates :attr:`_entropy_norm`
        according to new number of archive cells and resets :attr:`_numitrs_noprogress`
        to 0."""
        if self._entropy_norm is not None:
            self._entropy_norm = entropy(
                np.ones(self.archive.cells) / self.archive.cells
            )

        self._numitrs_noprogress = 0

    def _update_no_coverage_progress(self):
        """Increments :attr:`_numitrs_noprogress` if number of discovered archive cells
        remains the same for two successive calls to this function. Otherwise resets
        :attr:`_numitrs_noprogress` to 0."""
        if len(self.archive) == self._prev_numcells:
            self._numitrs_noprogress += self.batch_size
        else:
            self._numitrs_noprogress = 0
            self._prev_numcells = len(self.archive)

    def _check_upscale_schedule(self, upscale_schedule):
        """Checks that ``upscale_schedule`` is a valid upscale schedule, specifically:
            1. Must be a 2D array where the second dim equals :attr:`measure_dim`.
            2. The resolutions corresponding to each measure must be non-decreasing
               along axis 0.
            3. The first resolution within the schedule must equal :attr:`archive.dims`.

        Example of valid upscale_schedule:
            [
                [5, 5],
                [5, 10],
                [10, 10]
            ]

        Example of invalid upscale_schedule:
            [
                [5, 5],
                [5, 10],
                [10, 5]  <-  resolution for measure 2 decreases
            ]

        Args:
            upscale_schedule (np.ndarray): See ``upscale_schedule`` from
            :meth:`__init__`.
        """
        if upscale_schedule.ndim != 2:
            raise ValueError("upscale_schedule must have 2 dimensions.")

        if upscale_schedule.shape[1] != self.measure_dim:
            raise ValueError(
                f"Expected upscale_schedule of shape (any,{self.measure_dim}), "
                f"actually got {upscale_schedule.shape}."
            )

        if not np.all(np.diff(upscale_schedule, axis=0) >= 0):
            raise ValueError(
                "The resolutions corresponding to each measure must be "
                "non-decreasing along axis 0."
            )

        if not np.all(self.archive.dims == upscale_schedule[0]):
            raise ValueError(
                "Expected the first resolution within upscale_schedule to be "
                f"{self.archive.dims} (the resolution of this emitter's "
                f"archive), actually got {upscale_schedule[0]}."
            )

    def _sample_n_rescale(self, num_samples):
        """Samples `num_samples` solutions from the SOBOL sequence and rescales them to
        the bounds of the search space.

        Args:
            num_samples (int): Number of solutions to sample.

        Returns:
            numpy.ndarray: Array of shape (num_samples, :attr:`solution_dim`) containing
            the sampled solutions.
        """

        # SOBOL samples are in range [0, 1]. Need to rescale to bounds
        sobol_samples = self._sobol.random(n=num_samples)
        rescaled_samples = self.lower_bounds + sobol_samples * (
            self.upper_bounds - self.lower_bounds
        )

        return rescaled_samples

    def _get_expected_improvements(self, obj_mus, obj_stds):
        """Computes expected improvements predicted by :attr:`_gp` for a batch of
        solutions over all cells in the current archive. This function takes in the
        posterior means and standard deviations predicted by the objective gaussian
        process instead of the solutions themselves to avoid redundant computation.

        Args:
            obj_mus (np.ndarray): Array of shape (num_solutions,) containing the
                posterior objective means predicted by the gaussian process.
            obj_stds (np.ndarray): Array of shape (num_solutions,) containing the
                posterior objective standard deviations predicted by the gaussian
                process.

        Returns:
            numpy.ndarray: Array of shape (num_solutions, :attr:`archive.cells`)
            containing the expected improvements for each solution over each cell.
        """
        num_samples = obj_mus.shape[0]
        all_obj = np.full((self.archive.cells,), self.min_obj)
        elite_idx, elite_obj = self.archive.data(
            ["index", "objective"], return_type="tuple"
        )
        all_obj[elite_idx] = elite_obj

        distribution = norm(
            loc=np.repeat(all_obj[None, :], num_samples, axis=0),
            scale=np.repeat(obj_stds[:, None], self.archive.cells, axis=1),
        )

        return (obj_mus[:, None] - all_obj) * distribution.cdf(
            obj_mus[:, None]
        ) + obj_stds[:, None] * distribution.pdf(obj_mus[:, None])

    def _get_cell_probs(self, meas_mus, meas_stds, normalize=True, cutoff=True):
        """Computes archive cell membership probabilities predicted by :attr:`_gp` for a
        batch of solutions. This function takes in the posterior means and standard
        deviations predicted by the measure gaussian processes instead of the solutions
        themselves to avoid redundant computation.

        Args:
            meas_mus (np.ndarray): Array of shape (num_solutions, :attr:`measure_dim`)
                containing the posterior measure means predicted by the gaussian
                process.
            meas_stds (np.ndarray): Array of shape (num_solutions, :attr:`measure_dim`)
                containing the posterior measure standard deviations predicted by the
                gaussian process.
            normalize (bool): If ``True``, normalizes the cell probabilities such that
                they sum to 1 for each solution.
            cutoff (bool): If ``True``, sets cell probabilities below
                :attr:`cell_prob_cutoff` to 0.

        Returns:
            numpy.ndarray: Array of shape (num_solutions, :attr:`archive.cells`)
            containing the predicted cell probabilities for each solution.
        """
        num_solutions = meas_mus.shape[0]

        cell_probs = np.ones((num_solutions, *self.archive.dims))
        for measure_idx, (mus, stds) in enumerate(zip(meas_mus.T, meas_stds.T)):
            distribution = norm(loc=mus, scale=stds)

            # computes the cdf values at each cell boundary, this has shape
            # (num_solutions, num_boundaries).
            cdf_vals = distribution.cdf(self.archive.boundaries[measure_idx][:, None]).T

            # takes the difference between each pair of adjacent boundaries,
            # this has shape (num_solutions, num_boundaries-1) = (num_solutions,
            # measure_resolution)
            cdf_diffs = np.diff(cdf_vals, axis=1)

            # reshapes diffs to be compatible with element-wise multiplication
            for i in range(self.measure_dim):
                if i != measure_idx:
                    # axis i+1 because first axis is num_solutions
                    cdf_diffs = np.expand_dims(cdf_diffs, axis=i + 1)

            cell_probs *= cdf_diffs

        cell_probs = cell_probs.reshape((num_solutions, self.archive.cells))

        if cutoff:
            cell_probs[cell_probs < self.cell_prob_cutoff] = 0

        if normalize:
            # with ``cutoff``, it is possible a solution has 0 prob on all
            # cells, we don't normalize on those to prevent numerical error
            cell_probs_sum = np.sum(cell_probs, axis=1)[:, None]
            cell_probs_sum[cell_probs_sum == 0] = 1
            cell_probs /= cell_probs_sum

        return cell_probs

    def _get_ejie_values(self, samples):
        """Computes *Expected Joint Improvement of Elites* (EJIE) acquisition values of
        samples by multiplying the predicted expected improvements and cell membership
        probabilities. Returns individual EJIE values for each cell in an array of shape
        (num_solutions, :attr:`archive.cells`). You can use `np.sum(result, axis=1)` to
        get the total EJIE on the entire archive. Also returns the predicted cell
        membership probabilities for each sample in an array of shape (num_solutions,
        :attr:`archive.cells`).

        Args:
            samples (np.ndarray): Array of shape (num_samples, :attr:`solution_dim`)
                containing samples whose EJIE values need to be computed.

        Returns:
            tuple of (numpy.ndarray, numpy.ndarray): Returns an array of shape
            (num_solutions, :attr:`archive.cells`) containing each solution's EJIE
            values for each cell. Also returns an array of shape (num_solutions,
            :attr:`archive.cells`) containing the predicted cell membership
            probabilities for each solution.
        """
        mus, stds = self._gp.predict(
            samples.reshape(-1, self.solution_dim), return_std=True
        )

        expected_improvements = self._get_expected_improvements(mus[:, 0], stds[:, 0])

        cell_probs = self._get_cell_probs(
            mus[:, 1:], stds[:, 1:], normalize=True, cutoff=True
        )

        if self._entropy_norm is not None:
            all_zero_filter = np.all(np.isclose(cell_probs, 0), axis=1)
            entropies = np.zeros((mus.shape[0], 1))
            entropies[~all_zero_filter] = entropy(cell_probs[~all_zero_filter], axis=1)[
                :, None
            ]
            ejie_by_cell = (
                expected_improvements
                * cell_probs
                * (1 + entropies / self._entropy_norm)
            )
        else:
            ejie_by_cell = expected_improvements * cell_probs

        return ejie_by_cell, cell_probs

    def ask(self):
        """Returns :attr:`batch_size` solutions that are predicted to have high
        *Expected Joint Improvement of Elites* (EJIE) acquisition values.

        If ``self._gp`` has not been trained on any data and ``self._initial_solutions``
        is set, we return ``self._initial_solutions``, which was either provided by user
        at emitter initialization or sampled from a Sobol sequence.

        If ``self._gp`` has been trained on some data:

        1. Samples :attr:`num_sobol_samples` SOBOL samples.
        2. Computes the EJIE values for each sample, and keeps the top
           :attr:`_search_nrestarts` samples with the largest EJIE values
           and as starting points for pattern search.
        3. Starts a pattern search instance for each starting point to
           maximize their EJIE values.
        4. After all pattern search instances have converged, checks if at
           least :attr:`batch_size` samples with positive EJIE values have
           been found. If not, increments :attr:`_overspec` and repeats the
           process until at least :attr:`batch_size` solutions with positive
           EJIE values have been found.
        5. Returns the top :attr:`batch_size` solutions with the largest
           EJIE values.

        NOTE: This process has been simplified from the original implementation. The
        following are the components that are in the BOP-Elites source codes but removed
        here for simplicity:

        1. `load_previous_points
           <https://github.com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD.py#L337>`_
        2. `gen_elite_children
           <https://github.com/kentwar/BOPElites/blob/ main/algorithm/BOP_Elites_UKD.py#L298>`_
        3. We no longer restrict all starting points to be from unique cells. We
           understand this might compromise performance a bit, but enforcing all
           starting points from unique cells becomes messy in extreme cases when, for
           example, our archive resolution is so low that the number of cells is smaller
           than the number of starting points. Additionally, to my current
           understanding, it is not guaranteed that starting points from unique cells
           will result in higher optimized EJIE, because some cells might be easier to
           improve than others.
        4. We no longer explicitly add samples predicted to be in empty cells to the
           starting point pool, since samples predicted to be in empty cells should
           already have high EJIE.

        Returns:
            numpy.ndarray: Array of shape (:attr:`batch_size`, :attr:`solution_dim`)
            containing the solutions with the largest EJIE values in descending EJIE
            order.
        """
        if self.num_evals == 0:
            return np.clip(self.initial_solutions, self.lower_bounds, self.upper_bounds)

        # pymoo minimizes so need to negate
        pymoo_problem = self._pymoo_mods["FunctionalProblem"](
            n_var=self.solution_dim,
            objs=lambda x: -np.sum(self._get_ejie_values(x)[0], axis=1),
            xl=self.lower_bounds,
            xu=self.upper_bounds,
        )

        termination = self._pymoo_mods["DefaultSingleObjectiveTermination"]()

        optimization_outcomes = {
            "optimized_samples": [],
            "optimized_ejie_by_cell": [],
            "optimized_cell_probs": [],
        }
        while len(optimization_outcomes["optimized_samples"]) < self.batch_size:
            samples = self._sample_n_rescale(self.num_sobol_samples)
            starting_ejie_by_cell, _ = self._get_ejie_values(samples)

            search_starting_points = samples[
                np.argsort(np.sum(starting_ejie_by_cell, axis=1))[
                    -self._search_nrestarts :
                ]
            ]

            # optimizes ejie values of starting points
            found_positive_ejie = False
            for x0 in search_starting_points:
                optimizer = self._pymoo_mods["PatternSearch"](x0=x0)

                # Note: Using default pymoo minimize, PatternSearch, and
                # termination.
                result = self._pymoo_mods["minimize"](
                    problem=pymoo_problem,
                    algorithm=optimizer,
                    termination=termination,
                    copy_algorithm=False,
                    seed=self._seed,
                )

                if -result.F > 0:
                    optimization_outcomes["optimized_samples"].append(result.X)
                    # retrieve the cell-wise EJIE and probs for optimized
                    # solution
                    opt_ejie_by_cell, opt_cell_probs = self._get_ejie_values(result.X)
                    optimization_outcomes["optimized_ejie_by_cell"].append(
                        opt_ejie_by_cell.squeeze()
                    )
                    optimization_outcomes["optimized_cell_probs"].append(
                        opt_cell_probs.squeeze()
                    )
                    found_positive_ejie = True

            # if didn't find any positive ejie after optimization, increments
            # over-specification count
            # (we don't increment the over-specification count if we found
            # some positive EJIEs but not enough to fill the batch)
            if not found_positive_ejie:
                self._overspec += 1

        optimized_samples = np.array(optimization_outcomes["optimized_samples"])
        ejie_by_cell = np.array(optimization_outcomes["optimized_ejie_by_cell"])
        cell_probs = np.array(optimization_outcomes["optimized_cell_probs"])

        total_ejies = np.sum(ejie_by_cell, axis=1)
        # Most likely cell for each optimized solution
        best_cell_idx = np.argmax(cell_probs, axis=1)
        best_cell_probs = cell_probs[range(cell_probs.shape[0]), best_cell_idx]

        # Computes EJIE attributions of the most likely cell for each solution
        ejie_attributions = (
            ejie_by_cell[range(ejie_by_cell.shape[0]), best_cell_idx] / total_ejies
        )

        # Sort by EJIE, take the top :attr:`batch_size` samples
        sorted_idx = np.argsort(total_ejies)[::-1][: self.batch_size]

        # NOTE: BOP-Elites Algorithm 1 implements a different mis-specification
        # check, in which a mis-specification occurs if a sample is predicted
        # to be in a cell with high confidence, but the prediction turns out
        # to be wrong.
        # We implement a new mis-specification check as recommended by the
        # author. New mis-specification checks whether most of a sample's EJIE
        # is attributed to a single cell, which has low predicted cell
        # probability. This corresponds to the (undesirable) scenario in which
        # a cell that is likely unreachable dominates EJIE.
        for best_prob, attr_val in zip(
            best_cell_probs[sorted_idx], ejie_attributions[sorted_idx]
        ):
            if best_prob < 0.5 < attr_val:
                self._misspec += 1

        return optimized_samples[sorted_idx]

    def tell(self, solution, objective, measures, add_info, **fields):
        """Updates the gaussian process given evaluated solutions, objectives, and
        measures. Also upscales the archive if conditions are met.

        The function does the following:

        1. Adds ``solution``, ``objective``, and ``measures`` to :attr:`_dataset`.
        2. Updates :attr:`_gp` with :attr:`_dataset`.
        3. For each solution whose EJIE attribution exceeds 50%, checks whether its
           predicted cell is different from the cell it is actually assigned according
           to its evaluated measures. If so, increments :attr:`_misspec`.
        4. If :attr:`upscale_schedule` is not ``None``, and if the archive upscale
           conditions have been met, sends an upscale signal upstream by returning the
           next resolution to upscale to.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of solutions
                generated by this emitter's :meth:`ask()` method.
            objective (array-like): 1D array containing the objective function value of
                each solution.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with the
                measure values of each solution.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each argument
                should be an array with batch_size as the first dimension.

        Returns:
            numpy.ndarray: A 1D array of shape (:attr:`measure_dim`,) containing the
            next resolution to upscale to. The actual upscaling will be done in the
            scheduler, through
            :meth:`~ribs.schedulers.BayesianOptimizationScheduler.tell`. If no upscaling
            is needed in the current step, returns ``None``.
        """
        data, add_info = validate_batch(
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
        )

        # Adds new data to dataset.
        self._dataset["solution"] = np.vstack(
            (self._dataset["solution"], data["solution"])
        )
        self._dataset["objective"] = np.vstack(
            (self._dataset["objective"], data["objective"].reshape(-1, 1))
        )
        self._dataset["measures"] = np.vstack(
            (self._dataset["measures"], data["measures"])
        )

        # Updates (actually re-trains) GP with updated dataset.
        # sklearn occasionally raises LBFGS ConvergenceWarning, but this does
        # not seem to impact BOP-Elites performance too much.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._gp.fit(
                X=self._dataset["solution"],
                y=np.hstack((self._dataset["objective"], self._dataset["measures"])),
            )

        # Checks upscale conditions and upscales if needed
        # NOTE: BOP-Elites Algorithm 1 implements a slightly different upscale
        # condition, in which the archive upscale is triggered if either all
        # its cells have been filled or if num_evals > 2*cells. However, the
        # old condition may struggle with applications where some cells are not
        # feasible. We implement an improved condition here as recommended by
        # the original author. The new condition triggers the upscale when no
        # new cell has been found for multiple iterations.
        self._update_no_coverage_progress()
        if (self.upscale_schedule is not None) and np.any(
            np.all(self.upscale_schedule > self.archive.dims, axis=1)
        ):
            if self._numitrs_noprogress > self.upscale_trigger_threshold:
                # The next resolution on the schedule that is higher than the
                # current resolution along all measure dims
                next_res = self.upscale_schedule[
                    np.all(self.upscale_schedule > self.archive.dims, axis=1)
                ][0]

                return next_res

        return None
