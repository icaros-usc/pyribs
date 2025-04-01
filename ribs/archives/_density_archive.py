"""Contains the DensityArchive."""
import numpy as np
import torch
from zuko.utils import odeint

from ribs._utils import check_batch_shape, check_finite
from ribs.archives._archive_base import ArchiveBase, parse_dtype
from ribs.archives._array_store import ArrayStore


def gkern(x):
    gauss = np.exp(-0.5 * np.square(x))
    return gauss / np.sqrt(2 * np.pi)


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


class MLP(torch.nn.Sequential):

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
                hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])

        super().__init__(*layers[:-1])


class CNF(torch.nn.Module):

    def __init__(
        self,
        features,
        freqs=3,
        **kwargs,
    ):
        super().__init__()

        self.net = MLP(2 * freqs + features, features, **kwargs)

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t, x):
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(torch.cat((t, x), dim=-1))

    def encode(self, x):
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z):
        return odeint(self, z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x):
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t, x, ladj):
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx,
                                           x,
                                           I,
                                           is_grads_batched=True,
                                           create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return torch.distribution.Normal(
            0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FlowMatchingLoss(torch.nn.Module):

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y) - u).square().mean()


class DensityArchive(ArchiveBase):
    """An archive that divides each dimension into uniformly-sized cells.

    Args:
        solution_dim (int): Dimension of the solution space.
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        bw (float):
        buffer_size (int):
        density_method ("kde" or "fm"):
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
        ValueError: ``dims`` and ``ranges`` are not the same length.
    """

    def __init__(
            self,
            *,
            solution_dim,
            ranges,
            bw=10,
            buffer_size=10000,
            density_method="kde",  # kde or fm
            qd_score_offset=0.0,
            seed=None,
            dtype=np.float64,
            extra_fields=None,
            ckdtree_kwargs=None):

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=10,  # This value shouldn't matter
            measure_dim=len(ranges),
            # learning_rate and threhsold_min take on default values since we do
            # not use CMA-MAE threshold in this archive.
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
            extra_fields=extra_fields,
        )

        self._rng = np.random.default_rng(seed=seed)

        self.all_measures = np.empty((buffer_size, len(ranges)))
        self._num_occupied = 0

        self._measure_buffer = ArrayStore(
            field_desc={
                "measures": ((self._measure_dim,), dtype),
                # Must be same dtype as the objective since they share
                # calculations.
                **extra_fields,
            },
            capacity=buffer_size,
        )

        # The acceptance threshold for the buffer.
        self._w = np.exp(np.log(self._rng.uniform()) / buffer_size)
        # Number of solutions to skip.
        self._n_skip = int(
            np.log(self._rng.uniform()) / np.log(1 - self._w)) + 1

        self._density_method = density_method
        if self._density_method == "kde":
            # Kernel density estimation
            self._bw = bw
        elif self._density_method == "fm":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # Flow Matching
            self._fm = CNF(len(ranges),
                           hidden_features=[256] * 3).to(self._device)
            self._fm_loss = FlowMatchingLoss(self._fm)
            self._fm_opt = torch.optim.AdamW(self._fm.parameters(), lr=1e-3)

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)

    # def index_of(self, measures) -> np.ndarray:
    #     """Returns the index of the closest solution to the given measures.

    #     Unlike the structured archives like :class:`~ribs.archives.GridArchive`,
    #     this archive does not have indexed cells where each measure "belongs."
    #     Thus, this method instead returns the index of the solution with the
    #     closest measure to each solution passed in.

    #     This means that :meth:`retrieve` will return the solution with the
    #     closest measure to each measure passed into that method.

    #     Args:
    #         measures (array-like): (batch_size, :attr:`measure_dim`) array of
    #             coordinates in measure space.
    #     Returns:
    #         numpy.ndarray: (batch_size,) array of integer indices representing
    #           the location of the solution in the archive.
    #     Raises:
    #         RuntimeError: There were no entries in the archive.
    #         ValueError: ``measures`` is not of shape (batch_size,
    #             :attr:`measure_dim`).
    #         ValueError: ``measures`` has non-finite values (inf or NaN).
    #     """
    #     measures = np.asarray(measures)
    #     check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
    #     check_finite(measures, "measures")

    #     if self.empty:
    #         raise RuntimeError(
    #             "There were no solutions in the archive. "
    #             "`DensityArchive.index_of` computes the nearest neighbor to "
    #             "the input measures, so there must be at least one solution "
    #             "present in the archive.")

    #     _, indices = self._cur_kd_tree.query(measures)
    #     return indices.astype(np.int32)

    def add(self, solution, objective, measures, **fields):
        batch_size = measures.shape[0]
        buffer_size = self.all_measures.shape[0]

        # Downsampling the buffer using reservoir sampling.
        # https://dl.acm.org/doi/pdf/10.1145/198429.198435

        # Fill the buffer.
        n_occupied = len(self._measure_buffer)
        if buffer_size > n_occupied:
            n_fill = min(buffer_size - n_occupied, measures.shape[0])
            indices = np.arange(n_occupied, n_occupied + n_fill)
            self._measure_buffer.add(indices, measures[:n_fill], {}, [])

        # Replace measures in the buffer using reservoir sampling.
        i = n_fill + 1
        while i < measures.shape[0]:
            skip = min(self._n_skip, measures.shape[0])
            i += skip
            self._n_skip -= skip
            if i < measures.shape[0]:
                # Replace a random measure from the buffer with the measure i.
                replace = self._rng.integers(buffer_size)
                self.all_measures[replace] = measures[i]
                self._measure_buffer.add([replace], measures[i], {}, [])
                self._w *= np.exp(np.log(self._rng.uniform()) / buffer_size)
                # Compute the number of skips again.
                self._n_skip = int(
                    np.log(self._rng.uniform()) / np.log(1 - self._w)) + 1

        # Training CNF.
        if self._density_method == "fm":
            for _ in range(20):
                samples = np.random.randint(0, self._num_occupied, (256,))
                x = torch.from_numpy(self.all_measures[samples]).to(
                    self._device, torch.float32)

                self._fm_opt.zero_grad()
                self._fm_loss(x).backward()
                self._fm_opt.step()

        return np.ones(batch_size), np.ones(batch_size)

    def compute_density(self, measures_batch):
        """Computes density."""
        density = np.empty((measures_batch.shape[0],))
        if self._density_method == "kde":
            bw = self._bw
            # For some reason this is faster
            for j in range(measures_batch.shape[0]):
                density[j] = gaussian_kde_measures(measures_batch[j],
                                                   self.all_measures, bw)
            # density = gaussian_kde_measures_batch(measures_batch,
            #                                       self.all_measures, bw)
            # kernel = stats.gaussian_kde(self.all_measures.T, bw_method=bw)
            # density = kernel.evaluate(measures_batch.T)
        elif self._density_method == "fm":
            density = self._fm.log_prob(
                torch.from_numpy(measures_batch).to(self._device,
                                                    torch.float32))
            density = density.cpu().detach().numpy()
        else:
            raise ValueError("density_method not found")
        return density
