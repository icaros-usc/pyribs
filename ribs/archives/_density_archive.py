"""Contains the DensityArchive."""
import numpy as np

from ribs._utils import check_batch_shape, check_finite, readonly
from ribs.archives._archive_base import ArchiveBase, parse_dtype
from ribs.archives._array_store import ArrayStore


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


# TODO: Check ProximityArchive and ArchiveBase for method order.
# TODO: Comment out CNF code for now.
# TODO: sample_elites


class DensityArchive(ArchiveBase):
    # TODO: description
    # TODO: Note qd_score_offset is not used.
    # TODO: Stats?
    # TODO: There's an overflow error somewhere? Try tracing the warning.
    """

    Args:
        solution_dim (int): Dimension of the solution space.
        measure_dim (int): Dimension of the measure space.
        buffer_size (int): Size of the buffer of measure values.
        density_method (str): Method for computing density. Currently supports
            ``"kde"`` (KDE -- kernel density estimator).
        bandwidth (float): Bandwidth when using ``kde`` as the density
            estimator.
        qd_score_offset (float): Archives often contain negative objective
            values, and if the QD score were to be computed with these negative
            objectives, the algorithm would be penalized for adding new cells
            with negative objectives. Thus, a standard practice is to normalize
            all the objectives so that they are non-negative by introducing an
            offset. This QD score offset will be *subtracted* from all
            objectives in the archive, e.g., if your objectives go as low as
            -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
    Raises:
        ValueError: Unknown ``density_method`` provided.
    """

    def __init__(
        self,
        *,
        solution_dim,
        measure_dim,
        buffer_size=10000,
        density_method="kde",
        bandwidth=None,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
    ):
        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            # We don't use cells in this archive, so this value does not matter.
            cells=0,
            measure_dim=measure_dim,
            # learning_rate and threhsold_min take on default values since we do
            # not use CMA-MAE threshold in this archive.
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        self.all_measures = np.empty((buffer_size, measure_dim))
        self._num_occupied = 0

        # The acceptance threshold for the buffer.
        self._w = np.exp(np.log(self._rng.uniform()) / buffer_size)
        # Number of solutions to skip.
        self._n_skip = int(np.log(self._rng.uniform()) / np.log(1 - self._w))

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

    # TODO
    @property
    def empty(self):
        """bool: Whether the buffer is empty."""
        return self._num_occupied == 0

    # TODO
    def __len__(self):
        """Number of solutions in the buffer."""
        return self._num_occupied

    def add(
        self,
        solution=None,
        measures=None,
        objective=None,
        metadata_batch=None,
    ):
        input_measures = measures  # TODO: Get rid
        batch_size = measures.shape[0]
        buffer_size = self.all_measures.shape[0]

        input_density = self.calculate_density(input_measures)

        # Downsampling the buffer using reservoir sampling.
        # https://dl.acm.org/doi/pdf/10.1145/198429.198435

        # Fill the buffer.
        n_fill = 0
        if buffer_size > self._num_occupied:
            n_fill = min(buffer_size - self._num_occupied, batch_size)
            self.all_measures[self._num_occupied:self._num_occupied +
                              n_fill] = measures[:n_fill]
            # TODO: avoid mutating measures
            measures = measures[n_fill:]
            self._num_occupied += n_fill

        # Replace measures in the buffer using reservoir sampling.
        n_remaining = measures.shape[0]
        while n_remaining > 0:
            # Done with skipping, replace measures.
            if self._n_skip < n_remaining:
                replace = self._rng.integers(buffer_size)
                self.all_measures[replace] = measures[self._n_skip]
                self._w *= np.exp(np.log(self._rng.uniform()) / buffer_size)
                self._n_skip = int(
                    np.log(self._rng.uniform()) / np.log(1 - self._w))
            skip = min(self._n_skip, n_remaining)
            n_remaining -= skip
            self._n_skip -= skip

        # Training CNF.
        if self._density_method == "fm":
            for _ in range(20):
                samples = np.random.randint(0, self._num_occupied, (256,))
                x = torch.from_numpy(self.all_measures[samples]).to(
                    self._device, torch.float32)

                self._fm_opt.zero_grad()
                self._fm_loss(x).backward()
                self._fm_opt.step()

        return {
            # TODO
            "status": np.full(batch_size, 2),
            #  "objective": np.ones(batch_size),
            #  "measures": np.ones(batch_size),

            # TODO: Should density be calculated here instead?
            "density": input_density,
        }

    def calculate_density(self, measures_batch):
        """Calculates density."""
        if self._density_method == "kde":
            # TODO: implement bandwidth selection rules
            bandwidth = self._bandwidth
            # For some reason this is faster
            density = np.empty((measures_batch.shape[0],))
            for j in range(measures_batch.shape[0]):
                density[j] = gaussian_kde_measures(measures_batch[j],
                                                   self.all_measures, bandwidth)
            return density
            # density = gaussian_kde_measures_batch(measures_batch,
            #                                       self.all_measures, bandwidth)
            # kernel = stats.gaussian_kde(self.all_measures.T, bw_method=bandwidth)
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

    def index_of(self, measures):
        """Not implemented for DDS since it is not used here."""
        return NotImplemented

    def sample_elites(self, n):
        """Randomly samples elites from the archive.

        Currently, this sampling is done uniformly at random. Furthermore, each
        sample is done independently, so elites may be repeated in the sample.
        Additional sampling methods may be supported in the future.

        Since :namedtuple:`EliteBatch` is a namedtuple, the result can be
        unpacked (here we show how to ignore some of the fields)::

            solution_batch, objective_batch, measures_batch, *_ = \\
                archive.sample_elites(32)

        Or the fields may be accessed by name::

            elite = archive.sample_elites(16)
            elite.solution_batch
            elite.objective_batch
            ...

        Args:
            n (int): Number of elites to sample.
        Returns:
            EliteBatch: A batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.all_measures.shape[0] < 1:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(self._num_occupied, size=n)

        # TODO: Pick a random solution from the buffer.
        return {
            "solution": readonly(np.zeros((n, self._solution_dim))),
            "objective": np.zeros(n),
            "measures": self.all_measures[random_indices],
            "index": random_indices,
        }

    # -----------------------------------------------------------------------------------

    #  def __init__(
    #      self,
    #      *,
    #      solution_dim,
    #      measure_dim,
    #      buffer_size=10000,
    #      density_method="kde",
    #      bandwidth=None,
    #      qd_score_offset=0.0,
    #      seed=None,
    #      dtype=np.float64,
    #      extra_fields=None,
    #  ):
    #      ArchiveBase.__init__(
    #          self,
    #          solution_dim=solution_dim,
    #          # We don't use cells in this archive, so this value does not matter.
    #          cells=0,
    #          measure_dim=measure_dim,
    #          # learning_rate and threhsold_min take on default values since we do
    #          # not use CMA-MAE threshold in this archive.
    #          qd_score_offset=qd_score_offset,
    #          seed=seed,
    #          dtype=dtype,
    #          extra_fields=extra_fields,
    #      )

    #      # TODO: Seems there was partial conversion to using
    #      # self._measure_buffer; check if it was done.
    #      self.all_measures = np.empty((buffer_size, measure_dim))
    #      self._num_occupied = 0

    #      self._measure_buffer = ArrayStore(
    #          field_desc={
    #              "measures": ((self._measure_dim,), dtype),
    #              # Must be same dtype as the objective since they share
    #              # calculations.
    #              **extra_fields,
    #          },
    #          capacity=buffer_size,
    #      )

    #      # The acceptance threshold for the buffer.
    #      self._w = np.exp(np.log(self._rng.uniform()) / buffer_size)
    #      # Number of solutions to skip.
    #      self._n_skip = int(
    #          np.log(self._rng.uniform()) / np.log(1 - self._w)) + 1

    #      # TODO: property
    #      # TODO: test property
    #      self._density_method = density_method
    #      if self._density_method == "kde":
    #          # Kernel density estimation
    #          self._bandwidth = bandwidth
    #      #  elif self._density_method == "fm":
    #      #      self._device = "cuda" if torch.cuda.is_available() else "cpu"
    #      #      # Flow Matching
    #      #      self._fm = CNF(len(ranges),
    #      #                     hidden_features=[256] * 3).to(self._device)
    #      #      self._fm_loss = FlowMatchingLoss(self._fm)
    #      #      self._fm_opt = torch.optim.AdamW(self._fm.parameters(), lr=1e-3)
    #      else:
    #          raise ValueError(f"Unknown density_method {self._density_method}")

    #  # TODO: Add back? See ProximityArchive.
    #  def index_of(self, measures):
    #      """Not implemented for DDS since it is not used here."""
    #      return NotImplemented

    #  # def index_of(self, measures) -> np.ndarray:
    #  #     """Returns the index of the closest solution to the given measures.

    #  #     Unlike the structured archives like :class:`~ribs.archives.GridArchive`,
    #  #     this archive does not have indexed cells where each measure "belongs."
    #  #     Thus, this method instead returns the index of the solution with the
    #  #     closest measure to each solution passed in.

    #  #     This means that :meth:`retrieve` will return the solution with the
    #  #     closest measure to each measure passed into that method.

    #  #     Args:
    #  #         measures (array-like): (batch_size, :attr:`measure_dim`) array of
    #  #             coordinates in measure space.
    #  #     Returns:
    #  #         numpy.ndarray: (batch_size,) array of integer indices representing
    #  #           the location of the solution in the archive.
    #  #     Raises:
    #  #         RuntimeError: There were no entries in the archive.
    #  #         ValueError: ``measures`` is not of shape (batch_size,
    #  #             :attr:`measure_dim`).
    #  #         ValueError: ``measures`` has non-finite values (inf or NaN).
    #  #     """
    #  #     measures = np.asarray(measures)
    #  #     check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
    #  #     check_finite(measures, "measures")

    #  #     if self.empty:
    #  #         raise RuntimeError(
    #  #             "There were no solutions in the archive. "
    #  #             "`DensityArchive.index_of` computes the nearest neighbor to "
    #  #             "the input measures, so there must be at least one solution "
    #  #             "present in the archive.")

    #  #     _, indices = self._cur_kd_tree.query(measures)
    #  #     return indices.astype(np.int32)

    #  # TODO: Check ProximityArchive.
    #  def add(self, solution, objective, measures, **fields):
    #      batch_size = measures.shape[0]
    #      buffer_size = self.all_measures.shape[0]

    #      # Downsampling the buffer using reservoir sampling.
    #      # https://dl.acm.org/doi/pdf/10.1145/198429.198435

    #      # Fill the buffer.
    #      n_occupied = len(self._measure_buffer)
    #      if buffer_size > n_occupied:
    #          n_fill = min(buffer_size - n_occupied, measures.shape[0])
    #          indices = np.arange(n_occupied, n_occupied + n_fill)
    #          self._measure_buffer.add(indices, measures[:n_fill], {}, [])

    #      # Replace measures in the buffer using reservoir sampling.
    #      i = n_fill + 1
    #      while i < measures.shape[0]:
    #          skip = min(self._n_skip, measures.shape[0])
    #          i += skip
    #          self._n_skip -= skip
    #          if i < measures.shape[0]:
    #              # Replace a random measure from the buffer with the measure i.
    #              replace = self._rng.integers(buffer_size)
    #              self.all_measures[replace] = measures[i]
    #              self._measure_buffer.add([replace], measures[i], {}, [])
    #              self._w *= np.exp(np.log(self._rng.uniform()) / buffer_size)
    #              # Compute the number of skips again.
    #              self._n_skip = int(
    #                  np.log(self._rng.uniform()) / np.log(1 - self._w)) + 1

    #      # Training CNF.
    #      if self._density_method == "fm":
    #          for _ in range(20):
    #              samples = np.random.randint(0, self._num_occupied, (256,))
    #              x = torch.from_numpy(self.all_measures[samples]).to(
    #                  self._device, torch.float32)

    #              self._fm_opt.zero_grad()
    #              self._fm_loss(x).backward()
    #              self._fm_opt.step()

    #      # TODO: New return values.
    #      return np.ones(batch_size), np.ones(batch_size)

    #  # TODO: Docstring; see ProximityArchive.
    #  def compute_density(self, measures_batch):
    #      """Computes density."""
    #      density = np.empty((measures_batch.shape[0],))
    #      if self._density_method == "kde":
    #          bandwidth = self._bandwidth
    #          # For some reason this is faster
    #          for j in range(measures_batch.shape[0]):
    #              density[j] = gaussian_kde_measures(measures_batch[j],
    #                                                 self.all_measures, bandwidth)
    #          # density = gaussian_kde_measures_batch(measures_batch,
    #          #                                       self.all_measures, bandwidth)
    #          # kernel = stats.gaussian_kde(self.all_measures.T, bw_method=bandwidth)
    #          # density = kernel.evaluate(measures_batch.T)
    #      # TODO
    #      #  elif self._density_method == "fm":
    #      #      density = self._fm.log_prob(
    #      #          torch.from_numpy(measures_batch).to(self._device,
    #      #                                              torch.float32))
    #      #      density = density.cpu().detach().numpy()
    #      else:
    #          raise ValueError(f"Unknown density_method {self._density_method}")
    #      return density

    #  def sample_elites(self, n):
    #      """Randomly samples elites from the archive.

    #      Currently, this sampling is done uniformly at random. Furthermore, each
    #      sample is done independently, so elites may be repeated in the sample.
    #      Additional sampling methods may be supported in the future.

    #      Since :namedtuple:`EliteBatch` is a namedtuple, the result can be
    #      unpacked (here we show how to ignore some of the fields)::

    #          solution_batch, objective_batch, measures_batch, *_ = \\
    #              archive.sample_elites(32)

    #      Or the fields may be accessed by name::

    #          elite = archive.sample_elites(16)
    #          elite.solution_batch
    #          elite.objective_batch
    #          ...

    #      Args:
    #          n (int): Number of elites to sample.
    #      Returns:
    #          EliteBatch: A batch of elites randomly selected from the archive.
    #      Raises:
    #          IndexError: The archive is empty.
    #      """
    #      if self.all_measures.shape[0] < 1:
    #          raise IndexError("No elements in archive.")

    #      random_indices = self._rng.integers(self._num_occupied, size=n)

    #      return EliteBatch(
    #          readonly(np.zeros(self._solution_dim)),
    #          [0.0],
    #          readonly(self.all_measures[random_indices]),
    #          [random_indices],
    #          [0.0],
    #      )
