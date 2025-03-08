import numpy as np

from ribs._utils import check_finite, check_num_sol, validate_batch
from ribs.emitters._emitter_base import EmitterBase

from ribs.archives import GridArchive
from scipy.stats import norm, entropy
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.termination.default import DefaultSingleObjectiveTermination


class BayesianOptimizationEmitter(EmitterBase):
    """A sample-efficient emitter that models objective and measure functions
    with gaussian process surrogate models and uses Bayesian Optimisation to
    emit solutions that are predicted to have high *Expected Joint Improvement
    of Elites* (EJIE) acquisition values.

    Refer to `Kent et al. 2024 <https://ieeexplore.ieee.org/abstract/document
    /10472301>` for more information.

    Args:
        archive (ribs.archives.GridArchive): An archive to use when creating
            and inserting solutions. Currently, the only supported archive type
            is :class:`ribs.archives.GridArchive`.
        init_solution (array-like): (init_batch_size, :attr:`solution_dim`)
            array of solutions used in the initial batch of training data for
            gaussian processes.
        init_objective (array-like): (init_batch_size,) array with objective
            function evaluations of the solutions.
        init_measures (array-like): (init_batch_size, :attr:`measure_dim`)
            array with measure function evaluations of the solutions.
        bounds (array-like): Bounds of the solution space. Pass an array-like
            to specify the bounds for each dim. Each element in this array-like
            must be a tuple of ``(lower_bound, upper_bound)``. Cannot be
            ``None`` or ``+-inf`` because SOBOL sampling is used.
        search_nrestarts (int): Number of starting points for EJIE pattern
            search.
        entropy_ejie (bool): If ``True``, augments EJIE acquisition function
            with entropy to encourage measure space exploration. See <https://
            dl.acm.org/doi/10.1145/3583131.3590486> Sec. 4.1 for more details.
        upscale_schedule (array-like): An array of increasing archive
            resolutions starting with :attr:`archive.resolution` and ending
            with the user's intended final archive resolution. This will
            upscale the archive to the next scheduled resolution if every cell
            within the current archive has been filled, or the number of
            evaluated solutions is more than twice :attr:`archive.cells`. If
            ``None``, the archive will not be upscaled.
        min_obj (float or int): The lowest possible objective value. Serves as
            the default objective value within archive cells that have not been
            filled. Mainly used when computing expected improvement.
        batch_size (int): Number of solutions to return in :meth:`ask`. Must
            not exceed ``search_nrestarts``. It is recommended to set this to 1
            for sample efficiency.
        seed (int): Seed for the random number generator.
    """

    def __init__(
        self,
        archive,
        init_solution,
        init_objective,
        init_measures,
        bounds,
        *,
        search_nrestarts=5,
        entropy_ejie=False,
        upscale_schedule=None,
        min_obj=0,
        batch_size=1,
        seed=None,
    ):
        check_finite(bounds, "bounds")
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        if not (isinstance(archive, GridArchive)):
            raise NotImplementedError(
                f"archive type {type(archive)} not implemented for "
                "BayesianOptimizationEmitter. Expected GridArchive."
            )

        if (not upscale_schedule is None) and (
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
        # TODO: Using Matern kernal with default parameters
        self._gp = GaussianProcessRegressor(
            kernel=Matern(), normalize_y=True, n_targets=1 + self.measure_dim
        )

        # Trains GP with initial data
        check_num_sol(init_solution, init_objective, init_measures)
        self._dataset = {
            "solution": np.asarray(init_solution),
            "objective": np.asarray(init_objective).reshape(-1, 1),
            "measures": np.asarray(init_measures),
        }
        self._gp.fit(
            X=self._dataset["solution"],
            y=np.concatenate(
                (self._dataset["objective"], self._dataset["measures"]), axis=1
            ),
        )

        self._search_nrestarts = search_nrestarts

        if upscale_schedule is None:
            self._upscale_schedule = np.array([])
        else:
            self._upscale_schedule = np.asarray(upscale_schedule)
            self._check_upscale_schedule(self._upscale_schedule)

        self._batch_size = batch_size

        self._misspec = 0
        self._overspec = 0
        self._numitrs_noprogress = 0

        # Saves info on solutions returned by ask()
        self._asked_info = {
            "asked_solutions": np.zeros(
                (self.batch_size, self.solution_dim),
                dtype=self.dtype,
            ),
            "ejie_attributions": np.zeros((self.batch_size,), dtype=np.float64),
            "predicted_cells": np.zeros(
                (self.batch_size, self.measure_dim), dtype=np.int32
            ),
        }

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
        """float: Cutoff value (ohm) for :meth:`_get_cell_probs` as described
        in `Kent et al. 2024 <https://ieeexplore.ieee.org/abstract/document/
        10472301>` Sec.IV-D."""
        return (
            0.5
            * (2 / self.archive.cells)
            ** (
                (10 * self.solution_dim)
                / (self._misspec - 2 * self._overspec + self.num_evals + 1e-6)
            )
            ** 0.5
        )

    @property
    def num_evals(self):
        """int: Number of solutions stored in :attr:`_dataset`. This is the
        number of solutions that have been evaluated since the initialization
        of this emitter."""
        return self._dataset["solution"].shape[0]

    @property
    def measure_dim(self):
        """int: Number of measure functions."""
        return self.archive.measure_dim

    @property
    def num_sobol_samples(self):
        """int: Number of SOBOL samples to draw when choosing pattern search
        starting points in :meth:`ask`.

        TODO: If measure function gradients are available, a potentially better
        way to do this might be to do Latin Hypercube sampling within measure
        space, and then use measure gradients to find solutions achieving those
        measure space samples. See <https://wrap.warwick.ac.uk/id/eprint/189556
        /1/WRAP_Theses_Kent_2024.pdf> Sec. 6.3 for more details.
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
        """np.ndarray: The archive upscale schedule defined by user when
        initializing this emitter."""
        return self._upscale_schedule

    @property
    def upscale_trigger_threshold(self):
        """int: The maximum number of iterations the emitter is allowed to not
        find new cells before archive upscale is triggered. See <https://github.
        com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD_beta.py#L187>
        for more details."""
        return np.floor(np.sqrt(self.archive.cells))

    @property
    def min_obj(self):
        """float or int: The lowest possible objective value. See the `min_obj`
        parameter in :meth:``__init__``."""
        return self._min_obj

    @EmitterBase.archive.setter
    def archive(self, new_archive):
        """Allows resetting the archive associated with this emitter (for
        archive upscaling)."""
        self._archive = new_archive

    def _post_upscale_updates(self):
        """After the upstream scheduler upscales the archive, updates
        :attr:`_entropy_norm` according to new number of archive cells and
        resets :attr:`_numitrs_noprogress` to 0.
        """
        if not self._entropy_norm is None:
            self._entropy_norm = entropy(
                np.ones(self.archive.cells) / self.archive.cells
            )

        self._numitrs_noprogress = 0

    def _update_no_coverage_progress(self):
        """Increments :attr:`_numitrs_noprogress` if number of discovered
        archive cells remains the same for two successive calls to this
        function. Otherwise resets :attr:`_numitrs_noprogress` to 0.
        """
        if not "_prev_numcells" in self.__dict__:
            self._prev_numcells = len(self.archive)

        if len(self.archive) == self._prev_numcells:
            self._numitrs_noprogress += 1
        else:
            self._numitrs_noprogress = 0
            self._prev_numcells = len(self.archive)

    def _check_upscale_schedule(self, upscale_schedule):
        """Checks that ``upscale_schedule`` is a valid upscale schedule,
        specifically:
            1. Must be a 2D array where the second dim equals
               :attr:`measure_dim`.
            2. The resolutions corresponding to each measure must be
               non-decreasing along axis 0.

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

    def _sample_n_rescale(self, num_samples):
        """Samples `num_samples` solutions from the SOBOL sequence and rescales
        them to the bounds of the search space.

        Args:
            num_samples (int): Number of solutions to sample.

        Returns:
            np.ndarray: Array of shape (num_samples, :attr:`solution_dim`)
                containing the sampled solutions.
        """

        # SOBOL samples are in range [0, 1]. Need to rescale to bounds
        sobol_samples = self._sobol.random(n=num_samples)
        rescaled_samples = self.lower_bounds + sobol_samples * (
            self.upper_bounds - self.lower_bounds
        )

        return rescaled_samples

    def _get_expected_improvements(self, obj_mus, obj_stds):
        """Computes expected improvements predicted by :attr:`_gp` for a batch
        of solutions over all cells in the current archive. This function
        takes in the posterior means and standard deviations predicted by the
        objective gaussian process instead of the solutions themselves to
        avoid redundant computation.

        Args:
            obj_mus (np.ndarray): Array of shape (num_solutions,) containing
                the posterior objective means predicted by the gaussian process.
            obj_stds (np.ndarray): Array of shape (num_solutions,) containing
                the posterior objective standard deviations predicted by the
                gaussian process.

        Returns:
            np.ndarray: Array of shape (num_solutions, :attr:`archive.cells`)
                containing the expected improvements for each solution over
                each cell.
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
        """Computes archive cell membership probabilities predicted by
        :attr:`_gp` for a batch of solutions. This function takes in the
        posterior means and standard deviations predicted by the measure
        gaussian processes instead of the solutions themselves to avoid
        redundant computation.

        Args:
            meas_mus (np.ndarray): Array of shape (num_solutions,
                :attr:`measure_dim`) containing the posterior measure means
                predicted by the gaussian process.
            meas_stds (np.ndarray): Array of shape (num_solutions,
                :attr:`measure_dim`) containing the posterior measure standard
                deviations predicted by the gaussian process.
            normalize (bool): If ``True``, normalizes the cell probabilities
                such that they sum to 1 for each solution.
            cutoff (bool): If ``True``, sets cell probabilities below
                :attr:`cell_prob_cutoff` to 0.

        Returns:
            np.ndarray: Array of shape (num_solutions, :attr:`archive.cells`)
                containing the predicted cell probabilities for each solution.
        """
        num_solutions = meas_mus.shape[0]

        cell_probs = np.ones((num_solutions, *self.archive.dims))
        for measure_idx, (mus, stds) in enumerate(zip(meas_mus.T, meas_stds.T)):
            distribution = norm(loc=mus, scale=stds)

            # computes the cdf values at each cell boundary, this has shape
            # (num_solutions, num_boundaries).
            cdf_vals = distribution.cdf(
                self.archive.boundaries[measure_idx][:, None]
            ).T

            # takes the difference between each pair of adjacent boundaries,
            # this has shape (num_solutions, num_boundaries-1) = (num_solutions,
            # measure_resolution)
            cdf_diffs = np.diff(cdf_vals, axis=1)

            # reshapes diffs to be compatible with element-wise multiplication
            # TODO: make this prettier...
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

    def _get_ejie_values(
        self, samples, return_by_cell=False, return_cell_probs=False
    ):
        """Computes *Expected Joint Improvement of Elites* (EJIE) acquisition
        values of samples by multiplying the predicted expected improvements
        and cell membership probabilities.

        Args:
            samples (np.ndarray): Array of shape (num_samples,
                :attr:`solution_dim`) containing samples whose EJIE values need
                to be computed.
            return_by_cell (bool): If ``True``, returns individual EJIE values
                for each cell in an array of shape (num_solutions,
                :attr:`archive.cells`). This option is mainly used for
                determining whether more than half of the EJIE value is
                attributed to a single cell.
            return_cell_probs (bool): If ``True``, returns the predicted cell
                membership probabilities for each sample in an array of shape
                (num_solutions, :attr:`archive.cells`). This option is mainly
                used for identifying the most likely cell for each sample.

        Returns:
            np.ndarray or tuple(np.ndarray, np.ndarray): If
                ``return_by_cell=True``, returns an array of shape
                (num_solutions, :attr:`archive.cells`) containing each
                solution's EJIE values for each cell. If
                ``return_by_cell=False``, returns an array of shape
                (num_solutions,) containing the total EJIE of each sample,
                summed across all cells. If ``return_cell_probs=True``,
                additionally returns an array of shape (num_solutions,
                :attr:`archive.cells`) containing the predicted cell
                membership probabilities for each solution.
        """
        mus, stds = self._gp.predict(
            samples.reshape(-1, self.solution_dim), return_std=True
        )

        expected_improvements = self._get_expected_improvements(
            mus[:, 0], stds[:, 0]
        )

        cell_probs = self._get_cell_probs(
            mus[:, 1:], stds[:, 1:], normalize=True, cutoff=True
        )

        entropies = entropy(cell_probs, axis=1)[:, None]

        ejie_by_cell = expected_improvements * cell_probs
        ejie_entropy_by_cell = (
            ejie_by_cell
            if self._entropy_norm is None
            else ejie_by_cell * (1 + entropies / self._entropy_norm)
        )

        # TODO: Make this prettier...
        if return_by_cell:
            if return_cell_probs:
                return (
                    ejie_entropy_by_cell,
                    cell_probs,
                )
            else:
                return ejie_entropy_by_cell
        else:
            if return_cell_probs:
                return (
                    np.sum(ejie_entropy_by_cell, axis=1),
                    cell_probs,
                )
            else:
                return np.sum(ejie_entropy_by_cell, axis=1)

    def ask(self):
        """Returns :attr:`batch_size` solutions that are predicted to have high
        *Expected Joint Improvement of Elites* (EJIE) acquisition values.

        The function does the following:
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
            4. Returns the top :attr:`batch_size` solutions with the largest
               EJIE values.

        TODO(DrKent): This process has been simplified from the source codes
        implementation. The following are the components that are in the
        BOP-Elites source codes but removed here for simplicity:
            1. load_previous_points (<https://github.com/kentwar/BOPElites/blob/
            main/algorithm/BOP_Elites_UKD.py#L337>)
            2. gen_elite_children (<https://github.com/kentwar/BOPElites/blob/
            main/algorithm/BOP_Elites_UKD.py#L298>)
            3. We no longer restrict all starting points to be from unique
            cells. We understand this might compromise performance a bit, but
            enforcing all starting points from unique cells becomes messy in
            extreme cases when, for example, our archive resolution is so low
            that the number of cells is smaller than the number of starting
            points. Additionally, to my current understanding, it is not
            guaranteed that starting points from unique cells will result in
            higher optimized EJIE, because some cells might be easier to
            improve than others.
            4. We no longer explicitly add samples predicted to be in empty
            cells to the starting point pool, since samples predicted to be in
            empty cells should already have high EJIE.
        It is **very** likely that I made some mistakes in simplifying the
        process. Please definitely feel free to correct me if this is the case.


        Returns:
            np.ndarray: Array of shape (:attr:`batch_size`,
            :attr:`solution_dim`) containing the solutions with the largest
            EJIE values in descending EJIE order.
        """
        # pymoo minimizes so need to negate
        pymoo_problem = FunctionalProblem(
            n_var=self.solution_dim,
            objs=lambda x: -self._get_ejie_values(
                x, return_by_cell=False, return_cell_probs=False
            ),
            xl=self.lower_bounds,
            xu=self.upper_bounds,
        )

        termination = DefaultSingleObjectiveTermination()

        optimized_samples = []
        while len(optimized_samples) < self.batch_size:
            samples = self._sample_n_rescale(self.num_sobol_samples)
            ejie_values = self._get_ejie_values(
                samples, return_by_cell=False, return_cell_probs=False
            )

            search_starting_points = samples[
                np.argsort(ejie_values)[-self._search_nrestarts :]
            ]

            # optimizes ejie values of starting points
            # TODO: Dask this
            found_positive_ejie = False
            for x0 in search_starting_points:
                optimizer = PatternSearch(x0=x0)

                # TODO: Using default pymoo minimize, PatternSearch, and
                # termination.
                result = minimize(
                    problem=pymoo_problem,
                    algorithm=optimizer,
                    termination=termination,
                    copy_algorithm=False,
                    seed=self._seed,
                )

                if -result.F > 0:
                    optimized_samples.append(result.X)
                    found_positive_ejie = True

            # if didn't find any positive ejie after optimization, increments
            # over-specification count
            # (we don't increment the over-specification count if we found
            # some positive EJIEs but not enough to fill the batch)
            if not found_positive_ejie:
                self._overspec += 1

        optimized_samples = np.array(optimized_samples)

        ejie_by_cell, cell_probs = self._get_ejie_values(
            optimized_samples, return_by_cell=True, return_cell_probs=True
        )
        optimized_ejies = np.sum(ejie_by_cell, axis=1)
        # Most likely cell for each optimized solution
        best_cell_idx = np.argmax(cell_probs, axis=1)
        best_cell_probs = cell_probs[
            range(self._search_nrestarts), best_cell_idx
        ]

        # Computes EJIE attributions of the most likely cell for each solution
        ejie_attributions = ejie_by_cell[
            range(self._search_nrestarts), best_cell_idx
        ] / np.sum(ejie_by_cell, axis=1)

        # Sort by EJIE, take the top :attr:`batch_size` samples
        sorted_idx = np.argsort(optimized_ejies)[::-1][: self.batch_size]

        # Saves asked info for later tell() updates
        self._asked_info["asked_solutions"] = optimized_samples[sorted_idx]
        self._asked_info["ejie_attributions"] = ejie_attributions[sorted_idx]
        self._asked_info["predicted_cells"] = best_cell_idx[sorted_idx]

        # New cell mismatch checks whether the most likely cell predicted by
        # the GP has below 50% confidence (instead of matching predicted and
        # evaluated cells).
        # TODO(DrKent): Might need an author's note from Dr. Kent on this.
        for best_prob, attr_val in zip(
            best_cell_probs[sorted_idx], self._asked_info["ejie_attributions"]
        ):
            if best_prob < 0.5 and attr_val > 0.5:
                self._misspec += 1

        return self._asked_info["asked_solutions"].copy()

    def tell(self, solution, objective, measures, add_info, **fields):
        """Updates the gaussian process given evaluated solutions, objectives,
        and measures. Also upscales the archive if conditions are met.

        The function does the following:
            1. Adds ``solution``, ``objective``, and ``measures`` to
               :attr:`_dataset`.
            2. Updates :attr:`_gp` with :attr:`_dataset`.
            3. For each solution whose EJIE attribution exceeds 50%, checks
               whether its predicted cell is different from the cell it is
               actually assigned according to its evaluated measures. If so,
               increments :attr:`_misspec`.
            4. If :attr:`upscale_schedule` is not ``None``, and if the archive
               upcale conditions have been met, sends an upscale signal
               upstream by returning the next resolution to upscale to.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solutions generated by this emitter's :meth:`ask()` method.
            objective (array-like): 1D array containing the objective function
                value of each solution.
            measures (array-like): (batch_size, :attr:`measure_dim`) array
                with the measure values of each solution.
            add_info (dict): Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.

        Returns:
            np.ndarray: A 1D array of shape (:attr:`measure_dim`,) containing
                the next resolution to upscale to. The actual upscaling will
                be done in the scheduler, through
                :meth:`~ribs.schedulers.BayesianOptimizationScheduler.tell`. If
                no upscaling is needed in the current step, returns ``None``.
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
        self._gp.fit(
            X=self._dataset["solution"],
            y=np.hstack(
                (self._dataset["objective"], self._dataset["measures"])
            ),
        )

        # Checks upscale conditions and upscales if needed
        # NOTE: BOP-Elites Algorithm 1 implements a slightly different upscale
        # condition, in which the archive upscale is triggered if either all
        # its cells have been filled or if num_evals > 2*cells. However, the
        # old condition may struggle with applications where some cells are not
        # feasible.We implement an improved condition here as recommended by
        # the original author. The new condition triggers the upscale when no
        # new cell has been found for multiple iterations.
        self._update_no_coverage_progress()
        if np.any(np.all(self.upscale_schedule > self.archive.dims, axis=1)):
            if self._numitrs_noprogress > self.upscale_trigger_threshold:
                # The next resolution on the schedule that is higher than the
                # current resolution along all measure dims
                next_res = self.upscale_schedule[
                    np.all(self.upscale_schedule > self.archive.dims, axis=1)
                ][0]

                return next_res
