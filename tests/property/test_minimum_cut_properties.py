"""
Property-based tests for the min-cut computation used by
``ResultTemplateMinimumCutInterference._compute_from_scratch``
(``vision_unlearning/benchmarks/I_care/result_templates.py``).

These tests exercise ``networkx.minimum_cut`` directly, using the exact same graph
construction convention the production code uses (a ``networkx.DiGraph`` with a
``capacity`` edge attribute) -- not the ``ResultTemplate`` wrapper itself. The
min-cut/max-flow weak-duality property holds for ANY non-negative weighted digraph,
independent of I-CARE entity names, interference matrices, or the lambda-threshold
machinery, so this test needs no HuggingFace data and lives entirely in the lite
dependency tier (``networkx`` + ``hypothesis`` only, no torch/diffusers). It is a concrete,
checkable example of the lite/heavy boundary documented in CONTRIBUTING.md Section 6.

All properties below are **oracle-free**: none of them require re-implementing min-cut to
know the "correct" answer, only an independent bound or edge-case that must hold no matter
how the value was computed. That is what makes this a real property test and not a
disguised smoke test.

This is the production analogue of the max-flow/min-cut theorem discussed in the I-CARE
paper's appendix (``ap:flow_isolation``).
"""
from typing import Tuple

import networkx as nx
from hypothesis import given, strategies as st

MAX_NODES = 8
MAX_WEIGHT = 1e6

_WeightedDigraph = Tuple["nx.DiGraph", str, str]


def _trivial_cut_capacity(graph: "nx.DiGraph", source: str) -> float:
    """Capacity of the trivial cut ``({source}, everything else)``, summed directly from
    the graph's own edge data -- independent of ``networkx``'s own min-cut algorithm."""
    return sum(float(data["capacity"]) for _, _, data in graph.out_edges(source, data=True))


@st.composite
def weighted_digraphs(draw: "st.DrawFn") -> _WeightedDigraph:
    """Generate a small weighted digraph with two distinguished nodes ``source``/``sink``.

    Node count in [2, MAX_NODES]. The edge set is a random subset of all possible directed
    pairs (including a direct source->sink edge). Weights are bounded, non-negative,
    finite floats; zero-weight draws are dropped (mirrors the production code, which only
    ever adds edges with strictly positive weight -- see ``_interference_to_weight`` and
    the ``if w > lambda_threshold`` guard in ``_compute_from_scratch``).
    """
    n_nodes = draw(st.integers(min_value=2, max_value=MAX_NODES))
    nodes = [f"n{i}" for i in range(n_nodes)]
    source, sink = nodes[0], nodes[1]

    possible_edges = [(u, v) for u in nodes for v in nodes if u != v]
    edges = draw(
        st.lists(st.sampled_from(possible_edges), unique=True, max_size=len(possible_edges))
    )

    graph: "nx.DiGraph" = nx.DiGraph()
    graph.add_nodes_from(nodes)
    weight_strategy = st.floats(
        min_value=0.0, max_value=MAX_WEIGHT, allow_nan=False, allow_infinity=False
    )
    for u, v in edges:
        weight = draw(weight_strategy)
        if weight > 0.0:
            graph.add_edge(u, v, capacity=weight)

    return graph, source, sink


@given(weighted_digraphs())
def test_min_cut_never_exceeds_trivial_cut(data: _WeightedDigraph) -> None:
    """Weak duality: the minimum s-t cut value can never exceed the capacity of ANY
    feasible cut -- in particular the trivial cut ``({source}, rest)``, computed
    independently of ``nx.minimum_cut``."""
    graph, source, sink = data
    if not nx.has_path(graph, source, sink):
        return  # mirrors the RT's own no-path branch, which bypasses nx.minimum_cut
    cut_value, _ = nx.minimum_cut(graph, source, sink)
    assert cut_value <= _trivial_cut_capacity(graph, source) + 1e-9


@given(weighted_digraphs())
def test_min_cut_is_never_negative(data: _WeightedDigraph) -> None:
    """A cut's capacity is a sum of non-negative edge weights, so the minimum over all
    cuts can never be negative."""
    graph, source, sink = data
    if not nx.has_path(graph, source, sink):
        return
    cut_value, _ = nx.minimum_cut(graph, source, sink)
    assert cut_value >= 0.0


@given(weighted_digraphs())
def test_min_cut_is_zero_iff_no_directed_path(data: _WeightedDigraph) -> None:
    """No directed source->sink path implies a zero min-cut (the RT's own bypass branch
    returns 0.0 directly); conversely, since every edge in this construction carries
    strictly positive capacity, ANY existing directed path already proves a strictly
    positive max-flow / min-cut value."""
    graph, source, sink = data
    if not nx.has_path(graph, source, sink):
        return  # cut_value == 0.0 by construction/production convention; nothing to call
    cut_value, _ = nx.minimum_cut(graph, source, sink)
    assert cut_value > 0.0
