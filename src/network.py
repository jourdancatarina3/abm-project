"""Network builders.

The model can be run on three classes of network for sensitivity analysis:
- Barabasi--Albert (scale-free, hubs)         -- default
- Watts--Strogatz  (small-world, high clustering)
- Erdos--Renyi     (random, no preferential attachment)
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def build_network(
    kind: str,
    n: int,
    rng: np.random.Generator,
    m: int = 3,
    ws_k: int = 6,
    ws_p: float = 0.1,
    er_p: float = 0.006,
) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    if kind == "ba":
        g = nx.barabasi_albert_graph(n, m=m, seed=seed)
    elif kind == "ws":
        g = nx.watts_strogatz_graph(n, k=ws_k, p=ws_p, seed=seed)
    elif kind == "er":
        g = nx.erdos_renyi_graph(n, p=er_p, seed=seed)
    else:
        raise ValueError(f"Unknown network kind: {kind!r}")
    return g


def degree_sequence(g: nx.Graph) -> list[int]:
    return [g.degree(i) for i in range(g.number_of_nodes())]


def top_k_hubs(g: nx.Graph, k: int) -> list[int]:
    """Return the indices of the top-k nodes by degree."""
    deg = degree_sequence(g)
    return list(np.argsort(deg)[::-1][:k])
