"""Hypergraph representation for pre-geometric physics models.

Tier C module (see docs/HYPERGRAPH_NOTES.md for the full triage). This is not
a claim that space and time emerge from hypergraph rewriting -- it is
infrastructure for testing that hypothesis computationally, following the
same discipline as the rest of this repository: definitions and algorithms
are Tier B (exactly what they compute, no more); physical interpretation is
Tier C and never load-bearing.

A hypergraph here is a multiset of ordered hyperedges over an implicit node
set. Following the Wolfram Model / SetReplace convention, edges are ordered
tuples (not sets) because the order of nodes in a relation carries meaning
(e.g. a 3-edge (a, b, c) is not interchangeable with (c, b, a) -- it encodes
directed adjacency structure that the rewrite rules can exploit).

References
----------
Wolfram (2020), "A Project to Find the Fundamental Theory of Physics".
Gorard, Namuduri, Arsiwalla (2020), "Zenna: ..." (SetReplace formalism).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import count

Node = int
Edge = tuple[Node, ...]


@dataclass(frozen=True)
class Hypergraph:
    """An immutable, hashable multiset of ordered hyperedges.

    Nodes are plain integers. The node set is inferred from the edges (there
    are no isolated nodes in this representation, matching the Wolfram Model
    convention where nodes only exist by virtue of participating in a
    relation).
    """

    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(tuple(e) for e in self.edges))

    @classmethod
    def of(cls, *edges: Edge) -> Hypergraph:
        return cls(tuple(edges))

    @property
    def nodes(self) -> frozenset[Node]:
        return frozenset(n for e in self.edges for n in e)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def arities(self) -> Counter[int]:
        """Distribution of edge arities (edge lengths) present."""
        return Counter(len(e) for e in self.edges)

    def adjacency(self) -> dict[Node, set[Node]]:
        """Node -> set of nodes co-occurring in some edge with it.

        This is the graph one gets by "flattening" every hyperedge into a
        clique on its constituent nodes -- the standard construction used to
        define distance and ball-volume growth on a hypergraph (Wolfram
        Model's "spatial graph").

        Cached per (immutable, hashable) `Hypergraph` instance: `ball()` was
        calling this once per radius queried, and `local_dimension` queries
        several radii per node and is itself called many times by
        `mean_dimension` -- rebuilding an O(edges) dict from scratch on every
        one of those calls was the dominant cost in a dimension-vs-n
        convergence sweep (measured: a 3200-point, k=10 sweep across a
        6-point n-grid took over two minutes before this fix). Safe because
        every `Hypergraph` is frozen; nothing can invalidate the cache
        without producing a different (differently-hashed) instance.
        """
        return _adjacency_cached(self)

    def relabeled(self, start: int = 0) -> Hypergraph:
        """Canonically relabel nodes 0..n-1 in first-appearance order.

        Two hypergraphs that are isomorphic via a node relabeling produce the
        same relabeled form here only if the edges also appear in the same
        order -- this is a canonicalization for *display and hashing of a
        specific edge sequence*, not a graph-isomorphism certificate. Do not
        use this to test isomorphism; see `core.py` docstring caveat.
        """
        mapping: dict[Node, int] = {}
        counter = count(start)
        new_edges = []
        for edge in self.edges:
            new_edge = []
            for n in edge:
                if n not in mapping:
                    mapping[n] = next(counter)
                new_edge.append(mapping[n])
            new_edges.append(tuple(new_edge))
        return Hypergraph(tuple(new_edges))

    def fresh_node(self) -> Node:
        """A node id guaranteed not to collide with any existing node."""
        return (max(self.nodes) + 1) if self.nodes else 0

    def __len__(self) -> int:
        return self.num_edges

    def __iter__(self):
        return iter(self.edges)


@lru_cache(maxsize=64)
def _adjacency_cached(hg: Hypergraph) -> dict[Node, set[Node]]:
    """Module-level cache keyed by the (immutable, hashable) Hypergraph itself.

    Returns the SAME dict object on repeated calls with an equal hypergraph
    -- callers must treat the result as read-only (this is why `Hypergraph`
    exposes it only via the `.adjacency()` method, not this function
    directly). `maxsize=64` bounds memory for workloads that build many
    distinct hypergraphs (e.g. a rewriting evolution history); it does not
    limit how many times any single hypergraph's adjacency can be reused.
    """
    adj: dict[Node, set[Node]] = defaultdict(set)
    for edge in hg.edges:
        for i, u in enumerate(edge):
            for v in edge[i + 1 :]:
                adj[u].add(v)
                adj[v].add(u)
    return dict(adj)


def ball(hg: Hypergraph, source: Node, radius: int) -> set[Node]:
    """Nodes reachable from `source` within `radius` hops of the adjacency graph.

    The fundamental primitive for estimating emergent dimension (Wolfram
    Model's volume-growth definition): count |ball(r)| as a function of r.
    """
    if source not in hg.nodes:
        raise ValueError(f"node {source} is not present in this hypergraph")
    adj = hg.adjacency()
    frontier = {source}
    visited = {source}
    for _ in range(radius):
        next_frontier: set[Node] = set()
        for u in frontier:
            next_frontier |= adj.get(u, set()) - visited
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier
    return visited
