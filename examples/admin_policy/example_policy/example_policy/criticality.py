"""Dependency-criticality scores for the tasks of a pipeline.

Admin policies see one task at a time, so a policy cannot look at the
pipeline graph. This helper runs before submission: it scores every task of
a ``sky.Dag`` and writes the resulting tier into a resource label, which
``CriticalityTierPolicy`` then reads on the API server.

The score follows the Centrality-Entropy Index (P. Pokharel, "Federated
Deployment and Empirical Evaluation of the Centrality-Entropy Index Framework
for Enterprise Cloud Governance in Multi-Tier Distributed Environments",
FTC 2026):

    score = alpha * C + beta * H + gamma * R

- C: centrality of the task in the dependency graph (consumer -> producer),
  0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness, divided
  by the largest value in the pipeline.
- H: Shannon entropy of the task's past runtimes (10 bins), in [0, 1].
- R: an operator-declared priority in [0, 1].

alpha = 0.40 and gamma = 0.25 are the paper's weights. beta defaults to 0
because the entropy term did not change which tasks were protected in
simulations of spot pipelines; pass ``weights={'beta': 0.35}`` together with
``runtime_history`` to use it. A part with no input (no runtime history or no
priorities for any task) is dropped and the remaining weights are rescaled
to sum to 1. Tasks scoring at least CRITICAL_THRESHOLD are ``critical`` and
the rest ``standard``.

Example (see the admin policy docs for launching the labelled tasks):

    from example_policy import criticality
    with sky.Dag() as dag:
        prep = sky.Task.from_yaml('prep.yaml')
        train = sky.Task.from_yaml('train.yaml')
        prep >> train
    criticality.label_pipeline_criticality(dag, priorities={'train': 1.0})
"""
import math
import typing
from typing import Dict, Iterable, Mapping, Optional

import networkx as nx

if typing.TYPE_CHECKING:
    import sky

LABEL_KEY = 'criticality'
DEFAULT_WEIGHTS = {'alpha': 0.40, 'beta': 0.0, 'gamma': 0.25}
CRITICAL_THRESHOLD = 0.75


def pagerank(graph: nx.DiGraph,
             alpha: float = 0.85,
             tol: float = 1e-10,
             max_iter: int = 500) -> Dict[str, float]:
    """PageRank by power iteration, as ``nx.pagerank`` with default options.

    ``nx.pagerank`` needs scipy, which SkyPilot does not install.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    rank = {v: 1.0 / n for v in graph}
    out_deg = {v: graph.out_degree(v) for v in graph}
    for _ in range(max_iter):
        dangling = alpha * sum(rank[v] for v in graph if out_deg[v] == 0) / n
        new = {v: (1.0 - alpha) / n + dangling for v in graph}
        for u, v in graph.edges:
            new[v] += alpha * rank[u] / out_deg[u]
        err = sum(abs(new[v] - rank[v]) for v in graph)
        rank = new
        if err < n * tol:
            break
    return rank


def centrality(graph: nx.DiGraph) -> Dict[str, float]:
    """Composite centrality of each node, scaled so the largest is 1."""
    if graph.number_of_nodes() == 0:
        return {}
    pr = pagerank(graph)
    betweenness = nx.betweenness_centrality(graph)
    degree = nx.degree_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    composite = {
        n: 0.35 * pr[n] + 0.30 * betweenness[n] + 0.20 * degree[n] +
        0.15 * closeness[n] for n in graph
    }
    top = max(composite.values())
    if top <= 0:
        return {n: 0.0 for n in graph}
    return {n: v / top for n, v in composite.items()}


def runtime_entropy(history: Iterable[float], bins: int = 10) -> float:
    """Entropy of past runtimes, scaled by the longest run, in [0, 1]."""
    values = [float(v) for v in history if v is not None and v >= 0]
    if len(values) < 2 or max(values) <= 0:
        return 0.0
    longest = max(values)
    counts = [0] * bins
    for v in values:
        counts[min(int(v / longest * bins), bins - 1)] += 1
    total = len(values)
    entropy = -sum(c / total * math.log2(c / total) for c in counts if c > 0)
    return min(1.0, entropy / math.log2(bins))


def scores(graph: nx.DiGraph,
           priorities: Optional[Mapping[str, float]] = None,
           runtime_history: Optional[Mapping[str, Iterable[float]]] = None,
           weights: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """Scores the nodes of a pipeline graph.

    Args:
        graph: pipeline graph whose edges point from producer to consumer,
            as in ``sky.Dag.graph``. Nodes are task names.
        priorities: operator priority per node, in [0, 1].
        runtime_history: past runtimes per node, in any time unit.
        weights: overrides for alpha, beta and gamma.
    """
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    if not runtime_history:
        w['beta'] = 0.0
    if not priorities:
        w['gamma'] = 0.0
    total = w['alpha'] + w['beta'] + w['gamma']
    if total <= 0:
        raise ValueError('At least one weight must be positive.')
    # A consumer depends on its producer, so score the dependency graph.
    cent = centrality(graph.reverse(copy=True))
    out = {}
    for node in graph:
        c = cent.get(node, 0.0)
        h = runtime_entropy((runtime_history or {}).get(node, []))
        r = float((priorities or {}).get(node, 0.0))
        if not 0.0 <= r <= 1.0:
            raise ValueError(f'Priority of {node!r} must be in [0, 1].')
        out[node] = (w['alpha'] * c + w['beta'] * h + w['gamma'] * r) / total
    return out


def tier(score: float) -> str:
    """Maps a score in [0, 1] to ``critical`` or ``standard``."""
    return 'critical' if score >= CRITICAL_THRESHOLD else 'standard'


def label_pipeline_criticality(dag: 'sky.Dag',
                               priorities: Optional[Mapping[str, float]] = None,
                               runtime_history: Optional[Mapping[
                                   str, Iterable[float]]] = None,
                               weights: Optional[Mapping[str, float]] = None,
                               label_key: str = LABEL_KEY) -> Dict[str, str]:
    """Writes a criticality tier label into every task of ``dag``.

    Tasks that already have the label keep it, so an operator can always
    override the computed tier. Returns the tier of each task by name.
    """
    names = {}
    for i, task in enumerate(dag.tasks):
        names[task] = task.name or f'task-{i}'
    graph = nx.DiGraph()
    graph.add_nodes_from(names.values())
    graph.add_edges_from((names[u], names[v]) for u, v in dag.graph.edges)
    task_scores = scores(graph, priorities, runtime_history, weights)

    tiers = {}
    for task, name in names.items():
        resources = list(task.resources)
        existing = [(r.labels or {}).get(label_key) for r in resources]
        if any(existing):
            tiers[name] = next(t for t in existing if t)
            continue
        tiers[name] = tier(task_scores[name])
        labelled = [
            r.copy(labels={
                **(r.labels or {}), label_key: tiers[name]
            }) for r in resources
        ]
        task.set_resources(
            labelled if isinstance(task.resources, list) else set(labelled))
    return tiers
