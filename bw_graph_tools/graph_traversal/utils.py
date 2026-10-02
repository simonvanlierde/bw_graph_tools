import numpy as np
from bw2calc import LCA, spsolve
from scipy.sparse import spmatrix

from bw_graph_tools.graph_traversal.graph_objects import Node


class CachingSolver:
    """Class which caches cumulative LCA scores during graph traversal.

    ``_score_cache`` stores per-unit *cumulative LCA scores* (scalars) keyed by product index.
    The graph traversal only needs these scores, not full supply vectors. The score of one unit
    of product ``i`` is ``score_row @ A^-1 e_i``, which is entry ``i`` of ``A^-T score_row``. So
    one solve with the transposed technosphere matrix gives the scores of all products at once,
    instead of one solve per product.
    """

    def __init__(self, lca: LCA):
        self.lca = lca
        self._score_cache = {}
        # 1-D array of per-activity characterized scores (column sums of the characterized
        # biosphere matrix). Set by `set_score_row` before `scores` is called.
        self.score_row = None
        # (score_row it was solved for, unit scores of all products)
        self._all_unit_scores = None

    def in_cache(self, indices: set[int]) -> set[int]:
        """Return all `indices` values which already have a cached score."""
        return indices & self._score_cache.keys()

    def add_to_cache(self, index: int, unit_score: float) -> None:
        """Store a pre-computed per-unit cumulative score (for a demand amount of 1)."""
        self._score_cache[index] = float(unit_score)

    def set_score_row(self, characterized_biosphere: spmatrix) -> None:
        """Pre-compute the per-activity score row used to reduce supply vectors to scores.

        ``characterized_biosphere`` is the characterization-times-biosphere matrix (biosphere
        flows by activities). Its column sums give, for each activity, the cumulative score per
        unit of supply, so that ``score_row @ supply`` equals
        ``(characterized_biosphere * supply).sum()``.
        """
        self.score_row = np.asarray(characterized_biosphere.sum(axis=0)).ravel()

    def scores(self, indices: list[int], amounts: list[float]) -> list[float]:
        """Compute cumulative LCA scores for several products.

        Parameters
        ----------
        indices : list[int]
            Product (technosphere row) indices to demand, one unit each.
        amounts : list[float]
            Demanded amount for each product index, in the same order.

        Returns
        -------
        list[float]
            Cumulative LCA score for each `(index, amount)` pair, in input order.
        """
        missing = [index for index in indices if index not in self._score_cache]
        if missing:
            for index, score in zip(missing, self._unit_scores(missing)):
                self._score_cache[index] = float(score)
        return [
            self._score_cache[index] * amount for index, amount in zip(indices, amounts)
        ]

    def _unit_scores(self, indices: list[int]) -> np.ndarray:
        """Unit scores of `indices`, from one solve with the transposed technosphere matrix."""
        if self._all_unit_scores is None or self._all_unit_scores[0] is not self.score_row:
            all_unit_scores = spsolve(
                self.lca.technosphere_matrix.T.tocsr(),
                np.asarray(self.score_row, dtype=float),
            )
            self._all_unit_scores = (self.score_row, np.asarray(all_unit_scores).ravel())
        return self._all_unit_scores[1][indices]


class Counter:
    """Custom counter to have easy access to current value"""

    def __init__(self):
        self.value = -1

    def __next__(self):
        self.value += 1
        return self.value

    def __gt__(self, other):
        return self.value > other


def get_demand_vector_for_activity(
    node: Node,
    skip_coproducts: bool,
    matrix: spmatrix,
) -> (list[int], list[float]):
    """
    Get input matrix indices and amounts for a given activity. Ignores the reference production
    exchanges and optionally other co-production exchanges.

    Parameters
    ----------
    node : `Node`
        Activity whose inputs we are iterating over
    skip_coproducts : bool
        Whether or not to ignore positive production exchanges other than the reference
        product, which is always ignored
    matrix : scipy.sparse.spmatrix
        Technosphere matrix

    Returns
    -------

    row indices : list
        Integer row indices for products consumed by `Node`
    amounts : list
        The amount of each product consumed, scaled to `Node.supply_amount`. Same order as row
        indices.

    """
    matrix = (-1 * node.supply_amount * matrix[:, node.activity_index]).tocoo()

    rows, vals = [], []
    for x, y in zip(matrix.row, matrix.data):
        if x == node.reference_product_index:
            continue
        elif y == 0:
            continue
        elif y < 0 and skip_coproducts:
            continue
        rows.append(x)
        vals.append(y)
    return rows, vals
