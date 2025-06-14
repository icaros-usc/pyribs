"""Contains the DensityArchive."""
import numpy as np

from ribs._utils import check_batch_shape, check_finite
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._utils import parse_dtype


def gkern(x):
    gauss = np.exp(-0.5 * np.square(x))
    return gauss / np.sqrt(2 * np.pi)


# TODO: Which of these functions do we use?
def gaussian_kde_measures(m, all_measures, h):
    """Evaluates kernel density estimation.

    Args:
        m (np.ndarray): (dim, ) a single point in measure space.
        all_measures (np.ndarray): (batch_size, dim) batch of measures that
            parameterizes the KDE.
        h (float): The bandwidth of the kernel.
    Returns:
        sol (float): The evaluation of KDE(m).
    """
    diffs = m - all_measures
    norms = np.linalg.norm(diffs, axis=1) / h
    t = np.sum(gkern(norms))
    sol = t / (all_measures.shape[0] * h)
    return sol


def gaussian_kde_measures_batch(m_batch, all_measures, h):
    """Evaluates kernel density estimation.

    Args:
        m_batch (np.ndarray): (batch_size, dim) a batch of solutions in measure space.
        all_measures (np.ndarray): (batch_size, dim) batch of measures that
            parameterizes the KDE.
        h (float): The bandwidth of the kernel.
    Returns:
        sol (float): The evaluation of KDE(m).
    """
    # all_measures: batch_size_2, dim
    # diffs = m_batch - all_measures[None, :, :] # Play around with this one
    distances = np.expand_dims(m_batch, axis=1) - all_measures
    # end dim: (batch_size, batch_size_2, dim)
    # diffs[i] contains distances to all_measures

    # (batch_size, batch_size_2)
    norms = np.linalg.norm(distances, axis=-1) / h

    # expand gkern to take in the above batch size
    t = np.sum(gkern(norms), axis=1)  # (batch_size,)

    sol = t / (all_measures.shape[0] * h)

    return sol


# TODO: Comment out CNF code for now.
# TODO: There's an overflow error somewhere? Try tracing the warning.


class DensityArchive(ArchiveBase):
    """An archive that models the density of solutions in measure space.

    This archive originates in Density Descent Search in `Lee 2023
    <https://dl.acm.org/doi/10.1145/3638529.3654001>`_. It maintains a buffer of
    measures, and using that buffer, it builds a density estimator such as a
    KDE. The density estimator indicates which areas of measure space have, for
    instance, a high density of solutions -- to improve exploration, an
    algorithm would need to target areas with a low density of solutions.

    Incoming solutions are added to the buffer with `reservoir sampling
    <https://en.wikipedia.org/wiki/Reservoir_sampling>`_, specifically the
    algorithm described in `Li 1994
    <https://dl.acm.org/doi/abs/10.1145/198429.198435>`_. Reservoir sampling
    enables sampling uniformly from the incoming stream of solutions generated
    by the emitters.

    Unlike other archives, this archive does not store any elites, and as such,
    most methods from :class:`ArchiveBase` are not implemented. Rather, it is
    assumed that a separate ``result_archive`` (see
    :class:`~ribs.schedulers.Scheduler`) will store solutions when using this
    archive.

    Args:
        measure_dim (int): Dimension of the measure space.
        buffer_size (int): Size of the buffer of measures.
        density_method (str): Method for computing density. Currently supports
            ``"kde"`` (KDE -- kernel density estimator).
        bandwidth (float): Bandwidth when using ``kde`` as the density
            estimator.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type or dict): Data type of the measures.
            This can be ``"f"`` / ``np.float32``, ``"d"`` / ``np.float64``. For
            consistency with other archives, this can also be a dict specifying
            separate dtypes, of the form ``{"solution": <dtype>, "objective":
            <dtype>, "measures": <dtype>}``.
    Raises:
        ValueError: Unknown ``density_method`` provided.
    """

    def __init__(
        self,
        *,
        measure_dim,
        buffer_size=10000,
        density_method="kde",
        bandwidth=None,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
    ):
        self._rng = np.random.default_rng(seed)
        dtypes = parse_dtype(dtype)
        self.measure_dtype = dtypes["measures"]

        ArchiveBase.__init__(
            self,
            solution_dim=0,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        # Buffer for storing the measures.
        self._buffer = np.empty((buffer_size, measure_dim),
                                dtype=self.measure_dtype)
        # Number of occupied entries in the buffer.
        self._n_occupied = 0
        # The acceptance threshold for the buffer.
        self._w = np.exp(np.log(self._rng.uniform()) / buffer_size)
        # Number of solutions to skip.
        self._n_skip = int(np.log(self._rng.uniform()) / np.log(1 - self._w))

        # Set up density estimator.
        self._density_method = density_method
        if self._density_method == "kde":
            # Kernel density estimation
            self._bandwidth = bandwidth
        # TODO
        #  elif self._density_method == "fm":
        #      self._device = "cuda" if torch.cuda.is_available() else "cpu"
        #      print("device ", self._device)
        #      # Flow Matching
        #      self._fm = CNF(measure_dim,
        #                     hidden_features=[256] * 3).to(self._device)
        #      self._fm_loss = FlowMatchingLoss(self._fm)
        #      self._fm_opt = torch.optim.AdamW(self._fm.parameters(), lr=1e-3)

    ## Properties inherited from ArchiveBase ##

    # Necessary to implement this since `Scheduler` calls it.
    @property
    def empty(self):
        """bool: Whether the archive is empty. Since the archive does not store
        elites, we always mark it as not empty."""
        return False

    ## Properties that are not in ArchiveBase ##

    # TODO: Expose the buffer?

    ## Utilities ##

    def compute_density(self, measures):
        # TODO: docstring
        """Calculates density."""
        measures = np.asarray(measures)

        if self._density_method == "kde":
            # TODO: implement bandwidth selection rules
            bandwidth = self._bandwidth
            # For some reason this is faster
            density = np.empty((measures.shape[0],))
            for j in range(measures.shape[0]):
                density[j] = gaussian_kde_measures(measures[j], self._buffer,
                                                   bandwidth)
            return density
            # density = gaussian_kde_measures_batch(measures_batch,
            #                                       self._buffer, bandwidth)
            # kernel = stats.gaussian_kde(self._buffer.T, bw_method=bandwidth)
            # return kernel.evaluate(measures_batch.T)
        # TODO
        #  elif self._density_method == "fm":
        #      density = self._fm.log_prob(
        #          torch.from_numpy(measures_batch).to(self._device,
        #                                              torch.float32))
        #      return density.cpu().detach().numpy()
        else:
            # TODO: Better error.
            raise ValueError("density_method not found")

    ## Methods for writing to the archive ##

    def add(
        self,
        solution,
        objective,
        measures,
        **fields,
    ):
        # TODO: docstring
        """

        Args:
            solution (None or array-like): Included for API consistency.
            objective (None or array-like): Included for API consistency.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with
                measure space coordinates of all the solutions.
            fields (keyword arguments): Included for API consistency.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`int`): An array of
              integers that represent the "status" obtained when attempting to
              insert each solution in the batch. Each item has the following
              possible values:

              - ``0``: The solution was not added to the archive.
              - ``1``: The solution improved the objective value of a cell
                which was already in the archive.
              - ``2``: The solution discovered a new cell in the archive.

              All statuses (and values, below) are computed with respect to the
              *current* archive. For example, if two solutions both introduce
              the same new archive cell, then both will be marked with ``2``.

              The alternative is to depend on the order of the solutions in the
              batch -- for example, if we have two solutions ``a`` and ``b``
              that introduce the same new cell in the archive, ``a`` could be
              inserted first with status ``2``, and ``b`` could be inserted
              second with status ``1`` because it improves upon ``a``. However,
              our implementation does **not** do this.

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus`, e.g., with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"value"`` (:class:`numpy.ndarray` of
              :attr:`dtypes` ["objective"]): An array with values for each
              solution in the batch. With the default values of ``learning_rate
              = 1.0`` and ``threshold_min = -np.inf``, the meaning of each value
              depends on the corresponding ``status`` and is identical to that
              in CMA-ME (`Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_):

              - ``0`` (not added): The value is the "negative improvement,"
                i.e., the objective of the solution passed in minus the
                objective of the elite still in the archive (this value is
                negative because the solution did not have a high enough
                objective to be added to the archive).
              - ``1`` (improve existing cell): The value is the "improvement,"
                i.e., the objective of the solution passed in minus the
                objective of the elite previously in the archive.
              - ``2`` (new cell): The value is just the objective of the
                solution.

              In contrast, for other values of ``learning_rate`` and
              ``threshold_min``, each value is equivalent to the objective value
              of the solution minus the threshold of its corresponding cell in
              the archive.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf
                or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim",
                          "")
        check_finite(measures, "measures")

        input_measures = measures  # TODO: Get rid
        batch_size = measures.shape[0]
        buffer_size = self._buffer.shape[0]

        input_density = self.compute_density(input_measures)

        # Downsampling the buffer using reservoir sampling.
        # https://dl.acm.org/doi/pdf/10.1145/198429.198435

        # Fill the buffer.
        n_fill = 0
        if buffer_size > self._n_occupied:
            n_fill = min(buffer_size - self._n_occupied, batch_size)
            self._buffer[self._n_occupied:self._n_occupied +
                         n_fill] = (measures[:n_fill])
            # TODO: avoid mutating measures
            measures = measures[n_fill:]
            self._n_occupied += n_fill

        # Replace measures in the buffer using reservoir sampling.
        n_remaining = measures.shape[0]
        while n_remaining > 0:
            # Done with skipping, replace measures.
            if self._n_skip < n_remaining:
                replace = self._rng.integers(buffer_size)
                self._buffer[replace] = measures[self._n_skip]
                self._w *= np.exp(np.log(self._rng.uniform()) / buffer_size)
                self._n_skip = int(
                    np.log(self._rng.uniform()) / np.log(1 - self._w))
            skip = min(self._n_skip, n_remaining)
            n_remaining -= skip
            self._n_skip -= skip

        # Training CNF.
        if self._density_method == "fm":
            for _ in range(20):
                samples = np.random.randint(0, self._n_occupied, (256,))
                x = torch.from_numpy(self._buffer[samples]).to(
                    self._device, torch.float32)

                self._fm_opt.zero_grad()
                self._fm_loss(x).backward()
                self._fm_opt.step()

        return {
            "status": np.full(batch_size, 2, dtype=np.int32),
            # TODO: Should density be calculated here instead?
            "density": input_density,
        }
