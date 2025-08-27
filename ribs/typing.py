"""Custom data types for pyribs."""

# DEVELOPER NOTE: When adding new types, make sure to update the API listing in
# `docs/api/ribs.typing.rst`.

from __future__ import annotations

from typing import Any

import numpy as np

#: Represents data about a batch of solutions. The first dimension of each entry should
#: be the batch dimension.
BatchData = dict[str, np.ndarray]

#: Represents data about a single solution.
SingleData = dict[str, Any]
