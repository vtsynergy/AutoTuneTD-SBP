import argparse
import os
import json
from multiprocessing import Pool
import cudf
import cugraph
import numpy as np
import pandas as pd


DEGREE_MODES = [
    "uniform",
    "lognormal",
    "powerlaw",
    "bimodal"
]

ASSORTATIVITY_MODES = [
    "neutral",
    "positive",
    "negative"
]


class MutableEdgeListGraph:
    """Mutable simple graph used while constructing a synthetic edge list.

    cuGraph graphs are initialized from complete edge lists and are not intended
    for thousands of individual edge mutations. This small builder keeps the
    generation loop efficient; :func:`graph_metrics` materializes the finished
    edge list as a cuGraph graph for all topology calculations.
    """

    def __init__(self, num_nodes):
        self._nodes = tuple(range(num_nodes))
        self._edges = set()
        self._neighbors = [set() for _ in self._nodes]

    @staticmethod
    def _edge(u, v):
        u = int(u)
        v = int(v)
        return (u, v) if u < v else (v, u)

    def add_edge(self, u, v):
        if u == v:
            return
        edge = self._edge(u, v)
        if edge in self._edges:
            return
        self._edges.add(edge)
        self._neighbors[edge[0]].add(edge[1])
        self._neighbors[edge[1]].add(edge[0])

    def remove_edge(self, u, v):
        edge = self._edge(u, v)
        self._edges.remove(edge)
        self._neighbors[edge[0]].remove(edge[1])
        self._neighbors[edge[1]].remove(edge[0])

    def has_edge(self, u, v):
        return self._edge(u, v) in self._edges

    def neighbors(self, node):
        return self._neighbors[int(node)]

    def nodes(self):
        return self._nodes

    def edges(self):
        return self._edges

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)


def make_communities(num_nodes, num_communities, rng, imbalance=0.0):
    """
    Creates K communities.

    imbalance = 0.0 -> nearly equal community sizes
    imbalance > 0.0 -> more uneven community sizes
    """

    if imbalance <= 0.05:
        sizes = [num_nodes // num_communities] * num_communities
        sizes[-1] += num_nodes - sum(sizes)
    else:
        alpha = np.ones(num_communities) * max(0.2, 2.0 - imbalance)
        props = rng.dirichlet(alpha)
        sizes = np.maximum(2, (props * num_nodes).astype(int))

        while sizes.sum() < num_nodes:
            sizes[rng.integers(0, num_communities)] += 1

        while sizes.sum() > num_nodes:
            idx = rng.integers(0, num_communities)
            if sizes[idx] > 2:
                sizes[idx] -= 1

        sizes = sizes.tolist()

    communities = {}
    node = 0

    for c, size in enumerate(sizes):
        for _ in range(size):
            communities[node] = c
            node += 1

    return communities, sizes


def make_node_weights(num_nodes, mode, rng):
    """
    Controls degree-distribution style.

    uniform   -> narrow degree distribution
    lognormal -> moderately skewed
    powerlaw  -> highly skewed / hub-heavy
    bimodal   -> low-degree and high-degree groups
    """

    if mode == "uniform":
        weights = np.ones(num_nodes)

    elif mode == "lognormal":
        sigma = rng.uniform(0.5, 1.5)
        weights = rng.lognormal(mean=0.0, sigma=sigma, size=num_nodes)

    elif mode == "powerlaw":
        alpha = rng.uniform(1.5, 3.0)
        weights = rng.pareto(alpha, size=num_nodes) + 1.0

    elif mode == "bimodal":
        weights = np.ones(num_nodes)
        high_frac = rng.uniform(0.05, 0.25)
        high_nodes = rng.choice(
            num_nodes,
            size=max(1, int(high_frac * num_nodes)),
            replace=False
        )
        weights[high_nodes] = rng.uniform(5.0, 20.0)

    else:
        raise ValueError(f"Unknown degree mode: {mode}")

    weights = weights / weights.sum()
    return weights


def weighted_choice(nodes, weights, rng):
    local_weights = weights[nodes]
    local_weights = local_weights / local_weights.sum()
    return int(rng.choice(nodes, p=local_weights))


def pick_pair(nodes, weights, assortativity_mode, rng):
    """
    Picks a node pair.

    assortativity_mode:
        neutral  -> normal weighted sampling
        positive -> high-weight nodes tend to connect to high-weight nodes
        negative -> high-weight nodes tend to connect to low-weight nodes
    """

    if len(nodes) < 2:
        return None, None

    nodes = np.array(nodes)

    if assortativity_mode == "neutral":
        u = weighted_choice(nodes, weights, rng)
        v = weighted_choice(nodes, weights, rng)

        tries = 0
        while v == u and tries < 20:
            v = weighted_choice(nodes, weights, rng)
            tries += 1

        if u == v:
            return None, None

        return u, v

    sorted_nodes = nodes[np.argsort(weights[nodes])]
    midpoint = len(sorted_nodes) // 2

    low = sorted_nodes[:midpoint]
    high = sorted_nodes[midpoint:]

    if len(low) == 0 or len(high) == 0:
        return pick_pair(nodes, weights, "neutral", rng)

    if assortativity_mode == "positive":
        if rng.random() < 0.5:
            group = high
        else:
            group = low

        if len(group) < 2:
            return pick_pair(nodes, weights, "neutral", rng)

        u = weighted_choice(group, weights, rng)
        v = weighted_choice(group, weights, rng)

    elif assortativity_mode == "negative":
        u = weighted_choice(high, weights, rng)
        v = weighted_choice(low, weights, rng)

    else:
        raise ValueError(f"Unknown assortativity mode: {assortativity_mode}")

    if u == v:
        return None, None

    return int(u), int(v)


def add_edges_with_target_mu(
    G,
    num_edges,
    communities,
    weights,
    target_mu,
    assortativity_mode,
    rng
):
    """
    target_mu is approximately the fraction of edges between communities.

    mu close to 0.0 -> strong communities
    mu close to 1.0 -> weak/random communities
    """

    comm_to_nodes = {}

    for node, c in communities.items():
        comm_to_nodes.setdefault(c, []).append(node)

    comm_ids = list(comm_to_nodes.keys())

    max_attempts = num_edges * 200
    attempts = 0

    while G.number_of_edges() < num_edges and attempts < max_attempts:
        attempts += 1

        make_external = rng.random() < target_mu

        if make_external and len(comm_ids) > 1:
            c1, c2 = rng.choice(comm_ids, size=2, replace=False)

            u = weighted_choice(np.array(comm_to_nodes[c1]), weights, rng)
            v = weighted_choice(np.array(comm_to_nodes[c2]), weights, rng)

        else:
            c = int(rng.choice(comm_ids))
            nodes = comm_to_nodes[c]
            u, v = pick_pair(nodes, weights, assortativity_mode, rng)

            if u is None or v is None:
                continue

        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    # Fallback: fill remaining edges randomly while preserving simple graph
    all_nodes = list(G.nodes())

    while G.number_of_edges() < num_edges:
        u, v = rng.choice(all_nodes, size=2, replace=False)
        if not G.has_edge(int(u), int(v)):
            G.add_edge(int(u), int(v))

    return G


def increase_clustering(G, strength, rng):
    """
    Rewires some edges to close triangles.

    strength = 0.0 -> no extra triadic closure
    strength = 1.0 -> many triangle-closing attempts
    """

    if strength <= 0:
        return G

    num_attempts = int(strength * G.number_of_edges() * 2)

    nodes = list(G.nodes())

    for _ in range(num_attempts):
        v = int(rng.choice(nodes))
        neighbors = list(G.neighbors(v))

        if len(neighbors) < 2:
            continue

        u, w = rng.choice(neighbors, size=2, replace=False)
        u = int(u)
        w = int(w)

        if u == w or G.has_edge(u, w):
            continue

        # Remove a random edge that is not involved in this closing edge
        edges = tuple(G.edges())
        if len(edges) == 0:
            continue

        for _ in range(20):
            a, b = edges[int(rng.integers(0, len(edges)))]
            if len({a, b, u, w}) >= 3:
                G.remove_edge(a, b)
                G.add_edge(u, w)
                break

    return G


def graph_metrics(G, communities):
    """Calculate synthetic-graph topology metrics with RAPIDS cuGraph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_edges == 0:
        return {
            "num_nodes": num_nodes,
            "num_edges": 0,
            "num_communities": len(set(communities.values())),
            "actual_mu": 0.0,
            "avg_degree": 0.0,
            "degree_std": 0.0,
            "degree_min": 0.0,
            "degree_p25": 0.0,
            "degree_p50": 0.0,
            "degree_p75": 0.0,
            "degree_max": 0.0,
            "avg_clustering": 0.0,
            "clustering_std": 0.0,
            "clustering_p25": 0.0,
            "clustering_p50": 0.0,
            "clustering_p75": 0.0,
            "assortativity": 0.0,
            "density": 0.0,
            "largest_cc_fraction": 0.0,
        }

    edge_pairs = sorted(G.edges())
    edge_df = cudf.DataFrame(
        {
            "source": [source for source, _target in edge_pairs],
            "target": [target for _source, target in edge_pairs],
        }
    ).astype({"source": "int64", "target": "int64"})
    vertices = cudf.Series(G.nodes(), dtype="int64")
    gpu_graph = cugraph.Graph(directed=False)
    gpu_graph.from_cudf_edgelist(
        edge_df,
        source="source",
        destination="target",
        renumber=False,
        vertices=vertices,
    )

    degree_df = gpu_graph.degree().rename(columns={"degree": "degree"})
    degrees = degree_df["degree"].astype("float64")
    degree_quantiles = degrees.quantile([0.25, 0.50, 0.75]).to_pandas().to_dict()

    triangle_df = cugraph.triangle_count(gpu_graph)
    clustering_df = triangle_df.merge(degree_df, on="vertex", how="right")
    clustering_df["clustering"] = (
        2.0 * clustering_df["counts"]
        / (clustering_df["degree"] * (clustering_df["degree"] - 1.0))
    )
    clustering_df["clustering"] = clustering_df["clustering"].where(
        clustering_df["degree"] > 1,
        0.0,
    ).fillna(0.0)
    clustering = clustering_df["clustering"].astype("float64")
    clustering_quantiles = (
        clustering.quantile([0.25, 0.50, 0.75]).to_pandas().to_dict()
    )

    components = cugraph.connected_components(gpu_graph)
    largest_component_size = int(components.groupby("labels").size().max())
    largest_cc_fraction = largest_component_size / num_nodes

    endpoint_degrees = edge_df.merge(
        degree_df,
        left_on="source",
        right_on="vertex",
        how="left",
    ).rename(columns={"degree": "source_degree"}).drop(columns=["vertex"])
    endpoint_degrees = endpoint_degrees.merge(
        degree_df,
        left_on="target",
        right_on="vertex",
        how="left",
    ).rename(columns={"degree": "target_degree"}).drop(columns=["vertex"])
    endpoint_mean = float(
        (endpoint_degrees["source_degree"].sum()
         + endpoint_degrees["target_degree"].sum())
        / (2.0 * num_edges)
    )
    source_delta = endpoint_degrees["source_degree"] - endpoint_mean
    target_delta = endpoint_degrees["target_degree"] - endpoint_mean
    variance_sum = float((source_delta**2).sum() + (target_delta**2).sum())
    assortativity = (
        float(2.0 * (source_delta * target_delta).sum() / variance_sum)
        if variance_sum > 0.0
        else 0.0
    )

    community_df = cudf.DataFrame(
        {
            "vertex": list(communities.keys()),
            "cluster": list(communities.values()),
        }
    )
    classified_edges = edge_df.merge(
        community_df,
        left_on="source",
        right_on="vertex",
    ).drop(columns=["vertex"])
    classified_edges = classified_edges.merge(
        community_df,
        left_on="target",
        right_on="vertex",
        suffixes=("_source", "_target"),
    )
    actual_mu = float(
        (classified_edges["cluster_source"] != classified_edges["cluster_target"]).mean()
    )

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_communities": len(set(communities.values())),

        "actual_mu": actual_mu,

        "avg_degree": float(degrees.mean()),
        "degree_std": float(degrees.std(ddof=0)),
        "degree_min": float(degrees.min()),
        "degree_p25": float(degree_quantiles[0.25]),
        "degree_p50": float(degree_quantiles[0.50]),
        "degree_p75": float(degree_quantiles[0.75]),
        "degree_max": float(degrees.max()),

        "avg_clustering": float(clustering.mean()),
        "clustering_std": float(clustering.std(ddof=0)),
        "clustering_p25": float(clustering_quantiles[0.25]),
        "clustering_p50": float(clustering_quantiles[0.50]),
        "clustering_p75": float(clustering_quantiles[0.75]),

        "assortativity": assortativity,
        "density": float(gpu_graph.density()),
        "largest_cc_fraction": float(largest_cc_fraction)
    }


def generate_candidate_graph(
    num_nodes,
    num_edges,
    num_communities,
    seed,
    planned_config=None,
    fixed_degree_mode=None,
    fixed_target_mu=None
):
    rng = np.random.default_rng(seed)

    if planned_config is None:
        target_mu = fixed_target_mu if fixed_target_mu is not None else rng.uniform(0.02, 0.90)
        imbalance = rng.uniform(0.0, 1.0)
        degree_mode = fixed_degree_mode if fixed_degree_mode is not None else rng.choice(DEGREE_MODES)
        assortativity_mode = rng.choice(ASSORTATIVITY_MODES)
        clustering_strength = rng.uniform(0.0, 1.0)
    else:
        target_mu = fixed_target_mu if fixed_target_mu is not None else planned_config["target_mu"]
        imbalance = planned_config["community_imbalance"]
        degree_mode = fixed_degree_mode if fixed_degree_mode is not None else planned_config["degree_mode"]
        assortativity_mode = planned_config["assortativity_mode"]
        clustering_strength = planned_config["clustering_strength"]

    communities, sizes = make_communities(
        num_nodes=num_nodes,
        num_communities=num_communities,
        rng=rng,
        imbalance=imbalance
    )

    weights = make_node_weights(
        num_nodes=num_nodes,
        mode=degree_mode,
        rng=rng
    )

    G = MutableEdgeListGraph(num_nodes)

    G = add_edges_with_target_mu(
        G=G,
        num_edges=num_edges,
        communities=communities,
        weights=weights,
        target_mu=target_mu,
        assortativity_mode=assortativity_mode,
        rng=rng
    )

    G = increase_clustering(
        G=G,
        strength=clustering_strength,
        rng=rng
    )

    metrics = graph_metrics(G, communities)

    config = {
        "target_mu": float(target_mu),
        "community_imbalance": float(imbalance),
        "degree_mode": str(degree_mode),
        "assortativity_mode": str(assortativity_mode),
        "clustering_strength": float(clustering_strength),
        "seed": int(seed),
        "community_sizes": sizes
    }

    return G, communities, metrics, config


def latin_hypercube_values(count, low, high, rng):
    """
    Creates one value from each equally sized interval in [low, high].

    This gives deliberate coverage of the range without requiring a large
    candidate pool.
    """

    if count <= 0:
        return []

    edges = np.linspace(low, high, count + 1)
    values = [
        rng.uniform(edges[i], edges[i + 1])
        for i in range(count)
    ]

    rng.shuffle(values)
    return values


def balanced_categories(options, count, rng):
    values = []

    while len(values) < count:
        block = list(options)
        rng.shuffle(block)
        values.extend(block)

    return values[:count]


def planned_graph_configs(count, seed, fixed_degree_mode=None, fixed_target_mu=None):
    """
    Builds exactly count graph settings with planned diversity.

    Continuous parameters are spread with Latin-hypercube-style sampling.
    Categorical parameters are balanced so every mode appears regularly.
    """

    rng = np.random.default_rng(seed)

    if fixed_target_mu is None:
        target_mu_values = latin_hypercube_values(count, 0.02, 0.90, rng)
    else:
        target_mu_values = [fixed_target_mu] * count

    imbalance_values = latin_hypercube_values(count, 0.0, 1.0, rng)
    clustering_values = latin_hypercube_values(count, 0.0, 1.0, rng)
    if fixed_degree_mode is None:
        degree_modes = balanced_categories(DEGREE_MODES, count, rng)
    else:
        degree_modes = [fixed_degree_mode] * count

    assortativity_modes = balanced_categories(ASSORTATIVITY_MODES, count, rng)

    configs = []

    for i in range(count):
        configs.append({
            "target_mu": float(target_mu_values[i]),
            "community_imbalance": float(imbalance_values[i]),
            "degree_mode": degree_modes[i],
            "assortativity_mode": assortativity_modes[i],
            "clustering_strength": float(clustering_values[i])
        })

    return configs


def generate_candidate_from_task(task):
    (
        candidate_id,
        num_nodes,
        num_edges,
        num_communities,
        seed,
        planned_config,
        fixed_degree_mode,
        fixed_target_mu
    ) = task

    G, communities, metrics, config = generate_candidate_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_communities=num_communities,
        seed=seed,
        planned_config=planned_config,
        fixed_degree_mode=fixed_degree_mode,
        fixed_target_mu=fixed_target_mu
    )

    row = {}
    row.update(metrics)
    row.update(config)
    row["candidate_id"] = candidate_id

    return {
        "G": G,
        "communities": communities,
        "metrics": metrics,
        "config": config,
        "row": row
    }


def normalize_matrix(X):
    X = np.asarray(X, dtype=float)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)

    return (X - mins) / denom


def select_diverse_graphs(df, x):
    """
    Selects X graphs that maximize coverage of measured graph properties.

    This is greedy farthest-point sampling in normalized metric space.
    """

    coverage_cols = [
        "actual_mu",
        "degree_std",
        "degree_p25",
        "degree_p50",
        "degree_p75",
        "degree_max",
        "avg_clustering",
        "clustering_std",
        "clustering_p25",
        "clustering_p50",
        "clustering_p75",
        "assortativity",
        "largest_cc_fraction"
    ]

    Xmat = normalize_matrix(df[coverage_cols].values)

    selected = []

    # Start with graph closest to center
    center = np.ones(Xmat.shape[1]) * 0.5
    first = int(np.argmax(np.linalg.norm(Xmat - center, axis=1)))
    selected.append(first)

    while len(selected) < x:
        remaining = [i for i in range(len(df)) if i not in selected]

        best_idx = None
        best_dist = -1

        for idx in remaining:
            dists = [
                np.linalg.norm(Xmat[idx] - Xmat[j])
                for j in selected
            ]
            min_dist = min(dists)

            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = idx

        selected.append(best_idx)

    return selected, coverage_cols


def coverage_report(df, coverage_cols, bins=10):
    Xmat = normalize_matrix(df[coverage_cols].values)

    marginal_coverage = {}

    for i, col in enumerate(coverage_cols):
        hist, _ = np.histogram(Xmat[:, i], bins=bins, range=(0, 1))
        marginal_coverage[col] = float(np.sum(hist > 0) / bins)

    binned = np.floor(Xmat * bins).astype(int)
    binned = np.clip(binned, 0, bins - 1)

    unique_bins = set(tuple(row) for row in binned)
    max_possible = min(len(df), bins ** len(coverage_cols))

    joint_coverage = len(unique_bins) / max_possible

    _, counts = np.unique(binned, axis=0, return_counts=True)
    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
    entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0

    overall = (
        0.50 * np.mean(list(marginal_coverage.values()))
        + 0.30 * joint_coverage
        + 0.20 * entropy_score
    )

    return {
        "overall_coverage_score": float(overall),
        "marginal_coverage": marginal_coverage,
        "joint_coverage": float(joint_coverage),
        "entropy_score": float(entropy_score),
        "bins": bins,
        "coverage_metrics": coverage_cols
    }


def write_graph(G, communities, name, out_dir):
    edge_file = os.path.join(out_dir, f"{name}.tsv")
    gt_file = os.path.join(out_dir, f"{name}_truePartition.tsv")

    with open(edge_file, "w") as f:
        for u, v in sorted(G.edges()):
            f.write(f"{u + 1}\t{v + 1}\t1.0\n")

    with open(gt_file, "w") as f:
        for node in G.nodes():
            f.write(f"{node + 1}\t{communities[node]}\n")


def main(args):
    os.makedirs(args.data_dir, exist_ok=True)
    batch_name = (
        f"SBM_N{args.num_nodes}_M{args.num_edges}_K{args.num_communities}"
    )

    if args.target_mu is not None and not 0.0 <= args.target_mu <= 1.0:
        raise ValueError("--target_mu must be in [0, 1]")

    fixed_degree_mode = None if args.degree_mode == "mixed" else args.degree_mode
    fixed_target_mu = args.target_mu

    if args.sampling_strategy == "planned":
        candidate_count = args.num_graphs
        planned_configs = planned_graph_configs(
            args.num_graphs,
            args.seed,
            fixed_degree_mode=fixed_degree_mode,
            fixed_target_mu=fixed_target_mu
        )
    else:
        candidate_count = max(
            args.num_graphs * args.candidate_multiplier,
            args.num_graphs
        )
        planned_configs = [None] * candidate_count

    print(f"Input nodes: {args.num_nodes}")
    print(f"Input edges: {args.num_edges}")
    print(f"Input communities: {args.num_communities}")
    print(f"Requested output graphs X: {args.num_graphs}")
    print(f"Degree mode: {args.degree_mode}")
    print(f"Target mu / clustering difficulty: {args.target_mu if args.target_mu is not None else 'mixed'}")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print(f"Generating candidate graphs: {candidate_count}")
    if args.sampling_strategy == "planned":
        print("Candidate multiplier ignored in planned mode.")

    workers = max(1, min(args.workers, candidate_count))
    print(f"Generation workers: {workers}")

    tasks = [
        (
            i,
            args.num_nodes,
            args.num_edges,
            args.num_communities,
            args.seed + i,
            planned_configs[i],
            fixed_degree_mode,
            fixed_target_mu
        )
        for i in range(candidate_count)
    ]

    if workers == 1:
        candidates = [generate_candidate_from_task(task) for task in tasks]
    else:
        with Pool(processes=workers) as pool:
            candidates = pool.map(generate_candidate_from_task, tasks)

    df_all = pd.DataFrame([c["row"] for c in candidates])

    if candidate_count == args.num_graphs:
        coverage_cols = [
            "actual_mu",
            "degree_std",
            "degree_p25",
            "degree_p50",
            "degree_p75",
            "degree_max",
            "avg_clustering",
            "clustering_std",
            "clustering_p25",
            "clustering_p50",
            "clustering_p75",
            "assortativity",
            "largest_cc_fraction"
        ]
        selected_indices = list(range(args.num_graphs))
    else:
        selected_indices, coverage_cols = select_diverse_graphs(
            df_all,
            args.num_graphs
        )

    print(f"Selected graphs: {len(selected_indices)}")

    selected_rows = []

    for output_id, idx in enumerate(selected_indices):
        candidate = candidates[idx]

        name = (
            f"SBPGraph_N{args.num_nodes}"
            f"_M{args.num_edges}"
            f"_K{args.num_communities}"
            f"_id{output_id}"
        )

        write_graph(
            G=candidate["G"],
            communities=candidate["communities"],
            name=name,
            out_dir=args.data_dir
        )

        row = candidate["row"].copy()
        row["output_id"] = output_id
        row["dataset"] = name

        selected_rows.append(row)

    df_selected = pd.DataFrame(selected_rows)

    all_metrics_file = os.path.join(
        args.data_dir, f"{batch_name}_all_candidate_metrics.csv"
    )
    selected_metrics_file = os.path.join(
        args.data_dir, f"{batch_name}_selected_graph_metrics.csv"
    )
    report_file = os.path.join(
        args.data_dir, f"{batch_name}_coverage_report.json"
    )

    df_all.to_csv(all_metrics_file, index=False)
    df_selected.to_csv(selected_metrics_file, index=False)

    report = coverage_report(
        df_selected,
        coverage_cols=coverage_cols,
        bins=args.bins
    )

    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    print("\nGeneration complete.")
    print(f"Exactly {args.num_graphs} graphs written.")
    print(f"Output directory: {args.data_dir}")
    print(f"Selected metrics: {selected_metrics_file}")
    print(f"Candidate metrics: {all_metrics_file}")
    print(f"Coverage report: {report_file}")
    print(f"Coverage score: {report['overall_coverage_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate diverse SBM graphs with fixed N, M, and K."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./Data/SBM"
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        required=True,
        help="Fixed number of nodes."
    )

    parser.add_argument(
        "--num_edges",
        type=int,
        required=True,
        help="Fixed number of edges."
    )

    parser.add_argument(
        "--num_communities",
        type=int,
        required=True,
        help="Fixed number of ground-truth communities."
    )

    parser.add_argument(
        "--num_graphs",
        type=int,
        required=True,
        help="Number of final graphs to output. This is X."
    )

    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=10,
        help=(
            "Used by --sampling_strategy random. Generate "
            "X * candidate_multiplier candidates, then keep the best X."
        )
    )

    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["planned", "random"],
        default="planned",
        help=(
            "planned generates exactly X deliberately spread graph settings; "
            "random uses the older oversampling-and-selection approach."
        )
    )

    parser.add_argument(
        "--degree_mode",
        type=str,
        choices=["mixed"] + DEGREE_MODES,
        default="mixed",
        help="Degree-distribution style for this batch, or mixed for planned variation."
    )

    parser.add_argument(
        "--target_mu",
        type=float,
        default=None,
        help="Fixed clustering difficulty as the target fraction of cross-community edges."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes used to generate graphs."
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Bins used for coverage calculation."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    main(args)
