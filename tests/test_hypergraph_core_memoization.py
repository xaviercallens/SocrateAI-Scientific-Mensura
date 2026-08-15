"""`Hypergraph.nodes` and `__hash__` are memoized per instance.

Profiling a scale-aware dimension run (n=1600 uniform square,
docs/MENSURA_BENCH_V2.md section 8.6) found that 25.1s of a 44.9s run was
the `nodes` property rebuilding its frozenset over ~20k edges on each of 824
calls -- 20.4M generator steps -- and a further 2.0s was the frozen
dataclass's generated `__hash__` re-walking the entire nested edge tuple on
every `adjacency()` lru_cache *hit*. The cache was paying an O(total nodes)
key cost to avoid an O(total nodes) rebuild.

Both are now computed once per instance. This is only safe because
`Hypergraph` is frozen -- a cached value cannot go stale without producing a
different instance -- so these tests pin the frozen-ness and the dataclass
semantics (equality, hashing, repr, dict-key usability) that the caches must
not disturb, alongside the caching itself.
"""

from __future__ import annotations

import dataclasses

from socrates.hypergraph.core import Hypergraph


def _triangle() -> Hypergraph:
    return Hypergraph.of((0, 1), (1, 2), (2, 0))


# --------------------------------------------------------------------------
# The caching itself
# --------------------------------------------------------------------------


def test_nodes_returns_the_identical_object_on_repeat_calls():
    """The point of the fix: the second call must not rebuild anything."""
    hg = _triangle()
    assert hg.nodes is hg.nodes


def test_nodes_is_still_correct_and_instance_independent():
    hg, same = _triangle(), _triangle()
    assert hg.nodes == frozenset({0, 1, 2})
    assert hg.nodes == same.nodes


def test_nodes_is_computed_lazily_not_in_post_init():
    """Construction must stay cheap -- building many hypergraphs (a rewriting
    evolution history) should not pay for node sets nobody asks for."""
    hg = _triangle()
    assert "_nodes_cache" not in hg.__dict__
    _ = hg.nodes
    assert "_nodes_cache" in hg.__dict__


def test_hash_is_memoized_and_matches_the_generated_value():
    """Equal in value to the frozen dataclass's own `hash((edges,))`, so no
    behaviour depends on which implementation is in play."""
    hg = _triangle()
    assert hash(hg) == hash((hg.edges,))
    assert "_hash_cache" in hg.__dict__
    assert hash(hg) == hash(hg)


def test_empty_hypergraph_caches_correctly():
    """An empty node set is falsy -- a cache guarded on truthiness rather
    than on `None` would recompute it forever."""
    empty = Hypergraph(())
    assert empty.nodes == frozenset()
    assert empty.nodes is empty.nodes
    assert "_nodes_cache" in empty.__dict__


# --------------------------------------------------------------------------
# The dataclass semantics the caches must not disturb
# --------------------------------------------------------------------------


def test_equal_hypergraphs_still_hash_equal_after_caching():
    """The one invariant that must never break: eq implies hash-eq."""
    a, b = _triangle(), _triangle()
    _ = a.nodes, hash(a)  # populate a's caches, leave b's cold
    assert a == b
    assert hash(a) == hash(b)


def test_distinct_hypergraphs_remain_distinct():
    a = _triangle()
    b = Hypergraph.of((0, 1), (1, 2))
    assert a != b
    _ = a.nodes, b.nodes
    assert a != b


def test_caches_are_not_dataclass_fields():
    """They must stay out of `fields()`, `repr`, and therefore out of
    equality -- otherwise a cached and an uncached instance would differ."""
    hg = _triangle()
    _ = hg.nodes, hash(hg)
    assert [f.name for f in dataclasses.fields(hg)] == ["edges"]
    assert "_nodes_cache" not in repr(hg)
    assert "_hash_cache" not in repr(hg)


def test_still_usable_as_a_dict_key_across_cache_states():
    """`adjacency()`'s lru_cache depends on this exact property."""
    cold, warm = _triangle(), _triangle()
    _ = warm.nodes, hash(warm)
    table = {warm: "value"}
    assert table[cold] == "value"


def test_num_nodes_agrees_with_the_cached_node_set():
    hg = _triangle()
    assert hg.num_nodes == len(hg.nodes) == 3


def test_adjacency_cache_still_returns_a_consistent_result():
    """The hash memoization sits underneath `adjacency()`'s lru_cache; the
    cached adjacency must be unaffected by which instance asks for it."""
    a, b = _triangle(), _triangle()
    assert a.adjacency() == b.adjacency()
    assert a.adjacency()[0] == {1, 2}
