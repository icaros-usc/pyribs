import numpy as np

from ribs._utils import validate_batch
from ribs.emitters._emitter_base import EmitterBase

from ribs.archives import GridArchive
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# TODO: Implement pattern search so we don't need to add pymoo as a dependency
#   ...though if we plan to add moqd in the future, we might need pymoo
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem


class BayesianOptimizationEmitter(EmitterBase):
    """A sample-efficient emitter that returns candidate solutions maximizing the
    EJIE acquisition function (as described in BOP-Elites
    <https://ieeexplore.ieee.org/abstract/document/10472301> Sec.IV-A).
    """

    def __init__(
        self,
        archive,
        init_solutions,
        init_objectives,
        init_measures,
        # bounds are necessary here and cannot be +-inf because of SOBOL sampling
        bounds,
        *,
        # number of starting points for pattern search
        search_nrestarts=5,
        upscale_schedule=None,
        batch_size=1,
        seed=None,
    ):
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        if not (isinstance(archive, GridArchive)):
            raise NotImplementedError(
                f"archive type {type(archive)} not implemented "
                "for BayesianOptimizationEmitter. Expected GridArchive"
            )

        if batch_size > self.num_sobol_samples:
            raise ValueError(
                f"batch_size {batch_size} cannot be larger than SOBOL sample size {self.num_sobol_samples}"
            )
        else:
            self._batch_size = batch_size

        self._rng = np.random.default_rng(seed)
        self._sobol = Sobol(d=self.solution_dim, scramble=True, seed=self._rng)

        # Initializes a multi-output GP. 1 output for objective function, plus 1 output for each measure function
        self._gp = GaussianProcessRegressor(
            kernel=Matern(), normalize_y=True, n_targets=1 + self.measure_dim
        )

        # Trains GP with initial data
        self._dataset = {
            "solution": init_solutions,
            "objective": init_objectives,
            "measures": init_measures,
        }
        self._gp.fit(
            X=self._dataset["solution"],
            y=np.stack(
                (self._dataset["objective"], self._dataset["measures"]), axis=1
            ),
        )

        self._search_nrestarts = search_nrestarts

        self._misspec = 0
        self._overspec = 0
        # Saves info on solutions returned by ask()
        self._asked_info = {
            "asked_solutions": np.zeros(
                (self.batch_size, self.solution_dim),
                dtype=self.dtype,
            ),
            "ejie_proportions": np.zeros((self.batch_size,), dtype=np.float64),
            "predicted_cells": np.zeros(
                (self.batch_size, self.measure_dim), dtype=np.int32
            ),
        }

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def cell_prob_cutoff(self):
        """Implements EJIE cutoff (ohm) as described in BOP-Elites Sec.IV-D"""
        return (
            0.5
            * (2 / self.archive.cells)
            ** (
                (10 * self.solution_dim)
                / (
                    self.num_evals
                    - 2 * self._nopointsfound
                    + self._mispredicted
                )
            )
            ** 0.5
        )

    @property
    def num_evals(self):
        return self._dataset["solution"].shape[0]

    @property
    def measure_dim(self):
        return self.archive.measure_dim

    @property
    def num_sobol_samples(self):
        """Implements the sobol num_sample calculation in <https://github.com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD.py#L285>"""
        m = 10 if self.solution_dim < 2 else 1
        return np.clip(
            m * (self.solution_dim**2) * np.prod(self.measure_dim),
            10000,
            100000,
        )

    @property
    def dtype(self):
        return self.archive.dtypes["solution"]

    def _sample_n_rescale(self, num_samples):
        """Draws num_samples each of shape self.solution_dim and rescales them to
        search space bounds."""
        # SOBOL samples are in range [0, 1]. Need to rescale to bounds
        sobol_samples = self._sobol.random(n=num_samples)
        rescaled_samples = self.lower_bounds + sobol_samples * (
            self.upper_bounds - self.lower_bounds
        )

        return rescaled_samples

    def _get_expected_improvements(self, obj_mus, obj_stds):
        """Computes expected improvement of a solution over all cells in the archive.
        Please refer to BOP-Elites Sec.IV-A.
        """
        distribution = norm(loc=obj_mus, scale=obj_stds)

        # retrieves objective values of elites currently stored in the archive
        elite_objs = self.archive.data("objective")
        # empty cells are assumed to have 0 objective value as described in BOP-Elites Sec.IV-A2
        # TODO: BOP-Elites source codes seem to set this to average objective across all elites.
        #   (see <https://github.com/kentwar/BOPElites/blob/main/acq_functions/BOP_UKD_beta.py#L266>)
        elite_objs[np.isnan(elite_objs)] = 0

        # returns an EI for each cell, so shape should be (num_solutions, self.archive.cells)
        return (obj_mus[:, None] - elite_objs) * distribution.cdf(
            elite_objs
        ) + obj_stds[:, None] * distribution.pdf(elite_objs)

    def _get_cell_probs(self, meas_mus, meas_stds, normalize=True, cutoff=True):
        """Computes the probabilities of a solution belonging every cell in the archive.
        Please refer to BOP-Elites Sec.IV-A.

        Args:
            cutoff (bool): If True, sets cell_probs < self.cell_prob_cutoff to 0
        """
        num_solutions = meas_mus.shape[0]

        cell_probs = np.ones([(num_solutions, *self.archive.dims)])
        for measure_idx, (mus, stds) in enumerate(zip(meas_mus, meas_stds)):
            distribution = norm(loc=mus, scale=stds)
            # compute the cdf values at each cell boundary, this has shape (num_solutions, num_boundaries)
            cdf_vals = distribution.cdf(self.archive.boundaries[measure_idx])
            # take the difference between each pair of adjacent boundaries,
            #   this has shape (num_solutions, num_boundaries-1) = (num_solutions, measure_resolution)
            cdf_diffs = np.diff(cdf_vals, axis=1)
            # reshape diffs to be compatible with element-wise multiplication
            #   TODO: Make this prettier...
            for i in range(self.measure_dim):
                if i != measure_idx:
                    # axis i+1 because first axis is num_solutions
                    cdf_diffs = np.expand_dims(cdf_diffs, axis=i + 1)

            cell_probs *= cdf_diffs

        # returns a prob for each cell, so shape should be (num_solutions, self.archive.cells)
        cell_probs = cell_probs.reshape((num_solutions, self.archive.cells))

        if cutoff:
            cell_probs[cell_probs < self.cell_prob_cutoff] = 0

        if normalize:
            cell_probs /= np.sum(cell_probs, axis=1)

        return cell_probs

    def _get_ejie_values(
        self, samples, return_by_cell=False, return_cell_probs=False
    ):
        """Computes EJIE values of samples.

        Args:
            samples (np.ndarray): Size (num_samples, solution_dim) array containing samples whose EJIE
                values need to be computed.
            return_by_cell (bool): If True, returns individual EJIE values for all cells in a
                (num_solutions, num_cells) np.array. This option is mainly used for determining when
                more than half of all EJIE is attributed to a single cell (for updating
                mis-specification count).
            return_cell_probs (bool): If True, returns the predicted probabilities of each sample
                belonging to every cell in a (num_solutions, num_cells) np.array. This option is mainly
                used for identifying predicted archive indices.
        """
        mus, stds = self._gp.predict(samples, return_std=True)

        expected_improvements = self._get_expected_improvements(
            mus[:, 0], stds[:, 0]
        )

        cell_probs = self._get_cell_probs(
            mus[:, 1:], stds[:, 1:], normalize=True, cutoff=True
        )

        if return_by_cell:
            if return_cell_probs:
                return expected_improvements * cell_probs, cell_probs
            else:
                return expected_improvements * cell_probs
        else:
            if return_cell_probs:
                return (
                    np.sum(expected_improvements * cell_probs, axis=1),
                    cell_probs,
                )
            else:
                return np.sum(expected_improvements * cell_probs, axis=1)

    def ask(self):
        """Returns batch_size solutions with the largest EJIE values among SOBOL samples taken
        throughout search space.

        Please refer to BOP-Elites Algorithm 1
        """

        samples = self._sample_n_rescale(self.num_sobol_samples)
        ejie_values = self._get_ejie_values(
            samples, return_by_cell=False, return_cell_probs=False
        )

        # Summary of the select points process:
        #   1. sample self.num_sobol_samples sobol samples uniformly in search space
        #       - load_previous_points(??)
        #   2. add num_cells additional samples by mutating every stored elite
        #   3. for each cell, keeps top 5 samples with the largest EJIE
        #       - if a cell doesn't have any positive EJIE sample, don't consider it for starting point
        #   4. the 50-20-30 split
        # TODO (Questions):
        #   1. Do we need all these? Would it work to just have self.num_sobol_samples and choose the top
        #       EJIEs as starting points
        #       - in order to not confuse users, we want to keep only the core `necessary` components of
        #           BOP-Elites. It is fine to sacrifice some performance.

        # starting points are currently chosen to be the self._sample_n_rescale largest EJIE
        search_starting_points = samples[
            np.argsort(ejie_values)[self._sample_n_rescale :]
        ]

        # pymoo only minimizes so need to negate
        pymoo_problem = FunctionalProblem(
            n_var=self.solution_dim,
            objs=lambda x: -self._get_ejie_values(
                x, return_by_cell=False, return_cell_probs=False
            ),
            xl=self.lower_bounds,
            xu=self.upper_bounds,
        )

        # optimizes ejie values of starting points
        found_positive_ejie = False
        while not found_positive_ejie:
            optimized_samples = np.zeros(
                (self._search_nrestarts, self.solution_dim), dtype=self.dtype
            )
            optimized_ejies = np.zeros(
                (self._search_nrestarts,), dtype=np.float64
            )

            # TODO: Dask this(?)
            for i, x0 in enumerate(search_starting_points):
                optimizer = PatternSearch(x0=x0)

                result = minimize(
                    problem=pymoo_problem,
                    algorithm=optimizer,
                    # PatternSearch termination defaults to SingleObjectiveSpaceTermination(tol=1e-8)
                    # We are currently re-initializing optimizer for each starting point so no need to copy
                    copy_algorithm=False,
                    seed=self._rng,
                )

                optimized_samples[i, :] = result.X
                optimized_ejies[i] = -result.F

            # if didn't find any positive ejie even after optimization, increment over-specification count
            if np.any(optimized_ejies > 0):
                found_positive_ejie = True
            else:
                self._overspec += 1

        ejie_by_cell, cell_probs = self._get_ejie_values(
            optimized_samples, return_by_cell=True, return_cell_probs=True
        )
        # Highest prob. cell for each optimized solution
        best_cell_idx = np.argmax(cell_probs, axis=1)
        # EJIE at highest prob. cell, divided by total EJIE across all cells
        ejie_proportions = ejie_by_cell[
            range(self._search_nrestarts), best_cell_idx
        ] / np.sum(ejie_by_cell, axis=1)

        # Sort by EJIE, take the top batch_size samples
        sorted_idx = np.argsort(optimized_ejies)[::-1][: self.batch_size]

        # Saves asked info for later tell() updates
        self._asked_info["asked_solutions"] = optimized_samples[sorted_idx]
        self._asked_info["ejie_proportions"] = ejie_proportions[sorted_idx]
        self._asked_info["predicted_cells"] = self.archive.int_to_grid_index(
            sorted_idx
        )[sorted_idx]

        return self._asked_info["asked_solutions"].copy()

    def tell(self, solution, objective, measures, add_info, **fields):
        """Given solutions and their evaluated objective and measure values, tell() does the following:
        - Adds new data to self._dataset
        - Updates self._gp
        - Updates self._misspec and self._overspec
        - Upscales the archive to higher resolutions if either
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

        try:
            assert np.all(self._asked_info["asked_solutions"] == solution)
        except:
            __import__("pdb").set_trace()

        self._dataset["solution"] = np.stack(
            (self._dataset["solution"], data["solution"]), axis=0
        )
        self._dataset["objective"] = np.stack(
            (self._dataset["objective"], data["objective"]), axis=0
        )
        self._dataset["measures"] = np.stack(
            (self._dataset["measures"], data["measures"]), axis=0
        )

        self._gp.fit(
            X=self._dataset["solution"],
            y=np.stack(
                (self._dataset["objective"], self._dataset["measures"]), axis=1
            ),
        )

        # Adds 1 to self._misspec if GP attributes more than 50% of EJIE to a single cell,
        #   which turns out to be a wrong cell prediction after evaluated.
        # Only needs to check if the largest EJIE is at over 50%
        # TODO (Questions):
        #   1. If evaluate a batch of multiple solutions, is it still okay to only check the largest EJIE?
        #   2. Source codes (see <https://github.com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD.py#L151>)
        #       also seems to check for no_improvement on top of cell mismatch and attribute > 50%.
        eval_cells = self.archive.index_of(measures)
        if (
            np.any(self._asked_info["predicted_cells"][0] != eval_cells[0])
            and self._asked_info["ejie_proportions"][0] > 0.5
        ):
            self._misspec += 1

        # Upscale if QD score has not improved for multiple iterations
        # converged_by_fitness = self.noFitProgress > 2 * np.sqrt(
        #     np.prod(self.QDarchive.feature_resolution)
        # )
