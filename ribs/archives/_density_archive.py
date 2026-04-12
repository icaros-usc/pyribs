"""Contains the DensityArchive."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity

from ribs._utils import arr_readonly, check_batch_shape, check_finite
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._utils import parse_all_dtypes
from ribs.typing import BatchData, Float, Int

_FLOWS_EXTRA_HINT = (
    "DDS-CNF requires the 'flows' optional dependency group. "
    "Install it with `pip install ribs[flows]` (which brings in torch and zuko)."
)


def gkern(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel."""
    gauss = np.exp(-0.5 * np.square(x))
    return gauss / np.sqrt(2 * np.pi)


def gaussian_kde_measures(
    measures: np.ndarray, buffer: np.ndarray, h: float
) -> np.ndarray:
    """Evaluates kernel density estimation with a Gaussian kernel.

    The density is defined as zero if the buffer is empty.

    Args:
        measures: (measures_batch_size, measure_dim) array of points at which to
            estimate density.
        buffer: (buffer_batch_size, measure_dim) batch of measures that parameterize the
            KDE.
        h: Kernel bandwidth.

    Returns:
        Evaluation of KDE(m).
    """
    if buffer.shape[0] == 0:
        return np.zeros(measures.shape[0], dtype=measures.dtype)

    # (measures_batch_size, buffer_batch_size)
    norms = cdist(measures, buffer) / h

    # (measures_batch_size,)
    t = np.sum(gkern(norms), axis=1)

    return t / (buffer.shape[0] * h)


class _CNFDensityEstimator:
    """Continuous normalizing flow density estimator for DDS-CNF.

    This is a small, self-contained helper that owns a
    :class:`zuko.flows.CNF` and an :class:`torch.optim.Adam` optimizer, and
    knows how to incrementally fine-tune the flow on a feature buffer via
    maximum likelihood. It is intentionally not exposed in the public API --
    construct one implicitly by passing ``density_method="cnf"`` to
    :class:`DensityArchive`.

    The flow is built lazily on first training so that importing this module
    does not require torch or zuko to be installed.

    Args:
        measure_dim: Dimensionality of the feature space that the CNF models.
        lr: Learning rate for the Adam optimizer used during fine-tuning.
        train_steps: Number of gradient steps taken every time the estimator
            is asked to refit on a new buffer.
        batch_size: Mini-batch size used for stochastic gradient steps. If the
            buffer is smaller than ``batch_size``, the entire buffer is used
            as the batch.
        min_buffer_size: Minimum number of points in the buffer before the
            flow is trained at all. Calls before this threshold is reached
            are no-ops and density queries return zeros.
        device: Torch device on which the CNF lives. Strings like ``"cpu"``,
            ``"cuda"``, or an instance of :class:`torch.device` are accepted.
        seed: Seed for the torch random number generator controlling the
            CNF's weight initialization and any stochastic training steps.
        cnf_kwargs: Additional keyword arguments forwarded to
            :class:`zuko.flows.CNF`. ``features`` is set automatically from
            ``measure_dim`` and cannot be overridden. Defaults are chosen to
            match the small-MLP architecture used in the DDS paper
            experiments (``hidden_features=(64, 64)``).
    """

    def __init__(
        self,
        *,
        measure_dim: int,
        lr: float,
        train_steps: int,
        batch_size: int,
        min_buffer_size: int,
        device: Any,
        seed: int | None,
        cnf_kwargs: dict | None,
    ) -> None:
        # Lazy import keeps torch/zuko optional. If either is missing, surface a
        # clear install hint instead of a bare ModuleNotFoundError.
        try:
            import torch  # pylint: disable = import-outside-toplevel
            import zuko  # pylint: disable = import-outside-toplevel
        except ImportError as exc:
            raise ImportError(_FLOWS_EXTRA_HINT) from exc

        self._torch = torch
        self._zuko = zuko

        self._measure_dim = int(measure_dim)
        self._lr = float(lr)
        self._train_steps = int(train_steps)
        self._batch_size = int(batch_size)
        self._min_buffer_size = int(min_buffer_size)
        self._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

        # Disallow passing features via cnf_kwargs since it is determined by the
        # archive's measure_dim and would silently produce a shape mismatch.
        user_kwargs = dict(cnf_kwargs or {})
        if "features" in user_kwargs:
            raise ValueError(
                "`features` cannot be provided in cnf_kwargs; it is set "
                "automatically from measure_dim."
            )
        # Paper does not specify a precise CNF architecture. We pick a small
        # MLP that works well across the sphere/arm benchmarks in the DDS
        # experiments and let users override via cnf_kwargs.
        defaults = {"hidden_features": (64, 64)}
        defaults.update(user_kwargs)
        self._cnf_kwargs = defaults

        # zuko seeds its weight init from torch's global generator. Drop a
        # local generator here so that reproducible runs in pyribs don't leak
        # into the global torch state.
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self._flow: Any = None
        self._optimizer: Any = None
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        """Whether the flow has been trained on any data yet."""
        return self._fitted

    def _build(self) -> None:
        """Construct the CNF and its optimizer on first use."""
        torch = self._torch
        zuko = self._zuko
        # Seed weight init from our local generator to avoid polluting the
        # global torch RNG state used by the rest of the user's program.
        state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(int(self._generator.initial_seed()))
            self._flow = zuko.flows.CNF(
                features=self._measure_dim, **self._cnf_kwargs
            ).to(self._device)
        finally:
            torch.random.set_rng_state(state)
        self._optimizer = torch.optim.Adam(self._flow.parameters(), lr=self._lr)

    def fit(self, buffer: np.ndarray) -> None:
        """Fine-tune the flow on the current feature buffer.

        This takes ``train_steps`` Adam steps, each on a random mini-batch of
        ``batch_size`` rows drawn (with replacement) from ``buffer``. If the
        buffer is smaller than ``min_buffer_size``, the call is a no-op --
        the flow remains untrained and :meth:`log_density` returns zeros.

        Args:
            buffer: ``(n, measure_dim)`` array of features to fit on.
        """
        if buffer.shape[0] < self._min_buffer_size:
            return
        if self._flow is None:
            self._build()

        torch = self._torch
        # `DensityArchive.buffer` is a read-only view; torch emits a warning if
        # we hand it to `as_tensor` directly, so copy into a writable array.
        data = torch.as_tensor(
            np.array(buffer, copy=True), dtype=torch.float32, device=self._device
        )
        n = data.shape[0]
        effective_batch = min(self._batch_size, n)

        self._flow.train()
        for _ in range(self._train_steps):
            # Sample with replacement so that very small buffers still train.
            idx = torch.randint(0, n, (effective_batch,), generator=self._generator).to(
                self._device
            )
            batch = data[idx]
            loss = -self._flow().log_prob(batch).mean()
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()

        self._fitted = True

    def log_density(self, measures: np.ndarray) -> np.ndarray:
        """Evaluate log-density of the flow at a batch of measure-space points.

        If the flow has never been trained, returns zeros.

        Args:
            measures: ``(batch_size, measure_dim)`` array.

        Returns:
            ``(batch_size,)`` array of log-density values in float64.
        """
        if not self._fitted or self._flow is None:
            return np.zeros(measures.shape[0], dtype=np.float64)

        torch = self._torch
        data = torch.as_tensor(measures, dtype=torch.float32, device=self._device)
        self._flow.eval()
        with torch.no_grad():
            log_prob = self._flow().log_prob(data)
        return log_prob.detach().cpu().numpy().astype(np.float64)


# Developer Note: The documentation for this class is hacked. To list new methods,
# manually modify the template in docs/_templates/autosummary/class.rst


class DensityArchive(ArchiveBase):
    """An archive that models the density of solutions in measure space.

    This archive originates in Density Descent Search in `Lee 2024
    <https://dl.acm.org/doi/10.1145/3638529.3654001>`_. It maintains a buffer of
    measures, and using that buffer, it builds a density estimator such as a KDE. The
    density estimator indicates which areas of measure space have, for instance, a high
    density of solutions -- to improve exploration, an algorithm would need to target
    areas with a low density of solutions.

    Incoming solutions are added to the buffer with `reservoir sampling
    <https://en.wikipedia.org/wiki/Reservoir_sampling>`_, specifically as described in
    `Li 1994 <https://dl.acm.org/doi/abs/10.1145/198429.198435>`_. Reservoir sampling
    enables sampling uniformly from the incoming stream of solutions generated by the
    emitters.

    Unlike other archives, this archive does not store any elites, and as such, most
    methods from :class:`ArchiveBase` are not implemented. Rather, it is assumed that a
    separate ``result_archive`` (see :class:`~ribs.schedulers.Scheduler`) will store
    solutions when using this archive.

    Args:
        measure_dim: Dimension of the measure space.
        buffer_size: Size of the buffer of measures.
        density_method: Method for computing density. Supports ``"kde"`` (KDE
            -- kernel density estimator), ``"kde_sklearn"`` (KDE using
            :class:`sklearn.neighbors.KernelDensity`), and ``"cnf"`` (continuous
            normalizing flow, i.e. DDS-CNF from `Lee 2024
            <https://arxiv.org/abs/2312.11331>`_). When ``"kde_sklearn"`` is used,
            this archive computes *log density* rather than density; see
            :meth:`sklearn.neighbors.KernelDensity.score_samples`. When ``"cnf"``
            is used, this archive also returns *log density* since that is what
            the flow models directly. ``"cnf"`` requires the ``flows`` optional
            dependency group (``pip install ribs[flows]``), which brings in
            ``torch`` and ``zuko``.
        bandwidth: Bandwidth when using ``kde`` or ``kde_sklearn`` as the
            ``density_method``. Ignored for ``"cnf"``.
        sklearn_kwargs: kwargs for :class:`sklearn.neighbors.KernelDensity` when using
            ``"kde_sklearn"`` as the ``density_method``. Note that bandwidth is already
            passed in via the ``bandwidth`` parameter above.
        cnf_kwargs: Additional keyword arguments forwarded to
            :class:`zuko.flows.CNF` when ``density_method="cnf"``. ``features`` is
            set automatically from ``measure_dim`` and cannot be overridden.
            Defaults to ``{"hidden_features": (64, 64)}``.
        cnf_lr: Adam learning rate used to fine-tune the CNF during each call to
            :meth:`add` when ``density_method="cnf"``. Defaults to ``1e-3``.
        cnf_train_steps: Number of Adam steps taken every time the CNF is
            fine-tuned on the buffer. Defaults to ``100``.
        cnf_batch_size: Mini-batch size used when fine-tuning the CNF. Defaults
            to ``256``. If the buffer has fewer points, the entire buffer is used.
        cnf_min_buffer_size: Minimum number of points in the buffer before the
            CNF is trained. Before this threshold, the flow stays untrained and
            density queries return zeros. Defaults to ``128``.
        cnf_device: Torch device on which the CNF lives when
            ``density_method="cnf"``. Defaults to ``"cpu"``.
        seed: Value to seed the random number generator. Set to None to avoid a fixed
            seed.
        measures_dtype: Data type of the measures. Defaults to float64 (numpy's default
            floating point type).
        dtype: Alternative for providing data type of the measures. Included for API
            compatibility. Cannot be used at the same time as ``measures_dtype``.

    Raises:
        ValueError: Unknown ``density_method`` provided.
        ImportError: ``density_method="cnf"`` is requested but torch or zuko is
            not installed.
    """

    def __init__(
        self,
        *,
        measure_dim: Int,
        buffer_size: Int = 10000,
        density_method: Literal["kde", "kde_sklearn", "cnf"] = "kde",
        bandwidth: Float | None = None,
        sklearn_kwargs: dict | None = None,
        cnf_kwargs: dict | None = None,
        cnf_lr: Float = 1e-3,
        cnf_train_steps: Int = 100,
        cnf_batch_size: Int = 256,
        cnf_min_buffer_size: Int = 128,
        cnf_device: Any = "cpu",
        seed: Int | None = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        _, _, self._measures_dtype = parse_all_dtypes(
            dtype, None, None, measures_dtype, np
        )

        ArchiveBase.__init__(
            self,
            solution_dim=0,
            objective_dim=(),
            measure_dim=measure_dim,
        )

        # Buffer for storing the measures.
        self._buffer = np.empty((buffer_size, measure_dim), dtype=self._measures_dtype)
        # Number of occupied entries in the buffer.
        self._n_occupied = 0
        # Acceptance threshold for the buffer.
        self._w = np.exp(np.log(self._rng.uniform()) / buffer_size)
        # Number of solutions to skip.
        self._n_skip = int(np.log(self._rng.uniform()) / np.log(1 - self._w))

        # Set up density estimator.
        self._density_method = density_method
        self._cnf_estimator: _CNFDensityEstimator | None = None
        if self._density_method == "kde":
            # Kernel density estimation
            self._bandwidth = bandwidth
        elif self._density_method == "kde_sklearn":
            self._bandwidth = bandwidth
            self._sklearn_kwargs = (
                {} if sklearn_kwargs is None else sklearn_kwargs.copy()
            )
        elif self._density_method == "cnf":
            # Continuous normalizing flow density estimator (DDS-CNF,
            # Lee et al. 2024). The estimator starts untrained; the flow is
            # fitted at the end of each add() call once the buffer has at
            # least `cnf_min_buffer_size` points.
            self._cnf_estimator = _CNFDensityEstimator(
                measure_dim=int(measure_dim),
                lr=float(cnf_lr),
                train_steps=int(cnf_train_steps),
                batch_size=int(cnf_batch_size),
                min_buffer_size=int(cnf_min_buffer_size),
                device=cnf_device,
                seed=seed,
                cnf_kwargs=cnf_kwargs,
            )
        else:
            raise ValueError(f"Unknown density_method '{self._density_method}'")

    ## Properties inherited from ArchiveBase ##

    # Necessary to implement this since `Scheduler` calls it.
    @property
    def empty(self) -> bool:
        """Whether the archive is empty; always ``False``.

        Since the archive does not store elites, we always mark it as not empty.
        """
        return False

    ## Properties that are not in ArchiveBase ##

    @property
    def buffer(self) -> np.ndarray:
        """Buffer of measures considered in the density estimator.

        Shape (n, :attr:`measure_dim`).
        """
        return arr_readonly(self._buffer[: self._n_occupied])

    ## Utilities ##

    def compute_density(self, measures: ArrayLike) -> np.ndarray:
        """Computes density at the given points in measure space.

        Args:
            measures: (batch_size, :attr:`measure_dim`) array with measure space
                coordinates of all the solutions.

        Returns:
            ``(batch_size,)`` array of density values of the input solutions.
        """
        measures = np.asarray(measures, dtype=self._measures_dtype)

        if self._density_method == "kde":
            # Use self.buffer instead of self._buffer since self.buffer only contains
            # the valid entries of the buffer.
            return gaussian_kde_measures(
                measures,
                self.buffer,
                self._bandwidth,
            ).astype(self._measures_dtype)
        elif self._density_method == "kde_sklearn":
            if self.buffer.shape[0] == 0:
                return np.zeros(measures.shape[0], dtype=measures.dtype)
            # Note that this is log density with some normalization too.
            kde = KernelDensity(
                bandwidth=self._bandwidth,
                **self._sklearn_kwargs,
            ).fit(self.buffer)
            return kde.score_samples(measures).astype(self._measures_dtype)
        elif self._density_method == "cnf":
            # The CNF returns log-density (this matches kde_sklearn's
            # semantics). Before the estimator has been fit -- i.e., during the
            # first add() call or before the buffer crosses
            # `cnf_min_buffer_size` -- this returns zeros, mirroring how KDE
            # returns zeros on an empty buffer.
            assert self._cnf_estimator is not None
            return self._cnf_estimator.log_density(measures).astype(
                self._measures_dtype
            )
        else:
            raise ValueError(f"Unknown density_method '{self._density_method}'")

    ## Methods for writing to the archive ##

    def add(
        self,
        solution: ArrayLike | None,
        objective: ArrayLike | None,
        measures: ArrayLike,
        **fields: ArrayLike | None,
    ) -> BatchData:
        """Adds measures to the buffer and updates the density estimator if necessary.

        The measures are added to the buffer with reservoir sampling to enable sampling
        uniformly from the incoming solutions.

        Args:
            solution: Included for API consistency. Any value is ignored.
            objective: Included for API consistency. Any value is ignored.
            measures: (batch_size, :attr:`measure_dim`) array with measure space
                coordinates of all the solutions.
            fields: Included for API consistency. Any value is ignored.

        Returns:
            Information describing the result of the add operation. The dict contains
            the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`np.int32`): An array of
              integers that represent the "status" obtained when attempting to insert
              each solution in the batch. Since this archive does not store any elites,
              all statuses are set to 2 (which normally indicates the solution
              discovered a new cell in the archive -- see :class:`AddStatus`).

            - ``"density"`` (:class:`numpy.ndarray` of the dtype passed in at init): The
              density values of the measure passed in, before the buffer or density
              estimator was updated. Note that when ``"kde_sklearn"`` or ``"cnf"`` is
              used as the ``density_method``, *log density* is computed rather than
              density; see :meth:`sklearn.neighbors.KernelDensity.score_samples` for
              the ``kde_sklearn`` case and the class-level docstring for the
              ``cnf`` case.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures, dtype=self._measures_dtype)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim", "")
        check_finite(measures, "measures")
        batch_size = len(measures)
        buffer_size = len(self._buffer)

        add_info = {
            # Make all statuses be 2 as a placeholder value.
            "status": np.full(batch_size, 2, dtype=np.int32),
            # Note that density should be computed _before_ updating the buffer or
            # density estimator.
            "density": self.compute_density(measures),
        }

        # Add to the buffer using reservoir sampling as in Li 1994
        # (https://dl.acm.org/doi/pdf/10.1145/198429.198435).

        # First, fill the buffer if there are any slots available.
        n_fill = 0
        if buffer_size > self._n_occupied:
            n_fill = min(buffer_size - self._n_occupied, batch_size)
            self._buffer[self._n_occupied : self._n_occupied + n_fill] = measures[
                :n_fill
            ]
            remaining_measures = measures[n_fill:]
            self._n_occupied += n_fill
        else:
            remaining_measures = measures

        # Replace measures in the buffer using reservoir sampling.
        n_remaining = remaining_measures.shape[0]
        while n_remaining > 0:
            # Done with skipping, replace measures.
            if self._n_skip < n_remaining:
                replace = self._rng.integers(buffer_size)
                self._buffer[replace] = remaining_measures[self._n_skip]
                self._w *= np.exp(np.log(self._rng.uniform()) / buffer_size)
                self._n_skip = int(np.log(self._rng.uniform()) / np.log(1 - self._w))
            skip = min(self._n_skip, n_remaining)
            n_remaining -= skip
            self._n_skip -= skip

        # For DDS-CNF, Algorithm 1 line 12 of Lee et al. 2024 calls for the
        # density estimator to be refit on the updated buffer at the end of
        # every iteration. We fine-tune the CNF here so that the next
        # compute_density() call uses an up-to-date flow. KDE methods do not
        # need this step because they are non-parametric.
        if self._density_method == "cnf":
            assert self._cnf_estimator is not None
            self._cnf_estimator.fit(self.buffer)

        return add_info
