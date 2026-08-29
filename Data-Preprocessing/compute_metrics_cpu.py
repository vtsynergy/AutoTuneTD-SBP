import argparse
import os
import json
import re
import cudf
import cugraph
import pandas as pd
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ==========================================
# 1. HELPER FUNCTIONS (Combinatorial Math)
# ==========================================
def fastlbinom(n, k):
    n = np.maximum(n, 0)
    k = np.clip(k, 0, n)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def log_q_approx(n, k):
    n = np.maximum(n, 1)
    k = np.clip(k, 1, n)
    return fastlbinom(n - 1, k - 1) - gammaln(k + 1)

# ==========================================
# 2. METRIC FUNCTIONS (RAPIDS cuDF/cuGraph)
# ==========================================
def as_cudf(frame):
    """Return a cuDF DataFrame without copying an existing GPU frame."""
    return frame if isinstance(frame, cudf.DataFrame) else cudf.from_pandas(frame)


def calculate_modularity(edges, clusters):
    edges = as_cudf(edges)
    clusters = as_cudf(clusters)
    m = edges['weight'].sum()
    if float(m) == 0.0:
        return 0.0
    out_deg = edges.groupby('source', as_index=False)['weight'].sum().rename(columns={'source':'vertex', 'weight':'k_out'})
    in_deg = edges.groupby('target', as_index=False)['weight'].sum().rename(columns={'target':'vertex', 'weight':'k_in'})
    nodes = clusters.merge(out_deg, on='vertex', how='left').merge(in_deg, on='vertex', how='left').fillna(0)
    comm = nodes.groupby('cluster').agg({'k_out':'sum', 'k_in':'sum'}).reset_index()
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex').merge(clusters, left_on='target', right_on='vertex')
    e_internal = e_comm[e_comm['cluster_x'] == e_comm['cluster_y']].groupby('cluster_x')['weight'].sum().reset_index().rename(columns={'cluster_x':'cluster', 'weight':'E_c'})
    final = comm.merge(e_internal, on='cluster', how='left').fillna(0)
    final['term'] = (final['E_c']/m) - ((final['k_out']/m) * (final['k_in']/m))
    return float(final['term'].sum())

def calculate_conductance(edges, clusters):
    edges = as_cudf(edges)
    clusters = as_cudf(clusters)
    out_deg = edges.groupby('source', as_index=False)['weight'].sum().rename(columns={'source':'vertex', 'weight':'k_out'})
    nodes = clusters.merge(out_deg, on='vertex', how='left').fillna(0)
    comm_out = nodes.groupby('cluster')['k_out'].sum().reset_index()
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex').merge(clusters, left_on='target', right_on='vertex')
    e_internal = e_comm[e_comm['cluster_x'] == e_comm['cluster_y']].groupby('cluster_x')['weight'].sum().reset_index().rename(columns={'cluster_x':'cluster', 'weight':'E_c'})
    final = comm_out.merge(e_internal, on='cluster', how='left').fillna(0)
    final['cond'] = (final['k_out'] - final['E_c']) / final['k_out']
    return float(final['cond'].fillna(0).mean())

def calculate_inverse_hnorm(edges, clusters):
    """Inverse H-norm using the Peixoto nonparametric SBM log_q description length.
    hnorm       = normalized MDL = Σ_r log_q(e_r, n_r) / log_q(m, n)
    inverse_hnorm = 1 - hnorm  (higher = partition is more compressive than baseline)
    """
    edges = as_cudf(edges)
    clusters = as_cudf(clusters)
    m = int(edges['weight'].sum())         # total weighted edges
    n = int(clusters['vertex'].nunique())  # total nodes

    # Internal edge weight per community
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex') \
                  .merge(clusters, left_on='target', right_on='vertex')
    e_internal = e_comm[e_comm['cluster_x'] == e_comm['cluster_y']] \
                       .groupby('cluster_x')['weight'].sum() \
                       .reset_index().rename(columns={'cluster_x': 'cluster', 'weight': 'e_r'})

    # Node count per community
    n_comm = clusters.groupby('cluster').size().reset_index(name='n_r')
    comm = n_comm.merge(e_internal, on='cluster', how='left').fillna(0)

    # Sum log_q over all communities
    sum_log_q = sum(
        log_q_approx(int(row.e_r), int(row.n_r))
        for row in comm.to_pandas().itertuples(index=False)
    )

    # Baseline: entire graph as a single block
    baseline_log_q = log_q_approx(m, n)

    if baseline_log_q == 0:
        return 0.0
    hnorm = sum_log_q / baseline_log_q
    return 1.0 - hnorm

def calculate_f1_nmi(pred_clusters, true_clusters, num_vertices):
    pred_clusters = as_cudf(pred_clusters)
    true_clusters = as_cudf(true_clusters)
    merged = pred_clusters.merge(true_clusters, on='vertex', suffixes=('_pred', '_true'))
    if len(merged) == 0:
        return 0.0, 0.0

    contingency = merged.groupby(['cluster_pred', 'cluster_true']).size().reset_index(name='n_ij')
    row_sums = merged.groupby('cluster_pred').size().reset_index(name='n_i')
    col_sums = merged.groupby('cluster_true').size().reset_index(name='n_j')
    
    contingency_pdf = contingency.to_pandas()
    row_sums_pdf = row_sums.to_pandas()
    col_sums_pdf = col_sums.to_pandas()
    n_ij = contingency_pdf['n_ij'].to_numpy()
    n_i = row_sums_pdf['n_i'].to_numpy()
    n_j = col_sums_pdf['n_j'].to_numpy()
    
    # F1
    cell_pairs = np.sum(n_ij * (n_ij - 1)) / 2.0
    row_pairs = np.sum(n_i * (n_i - 1)) / 2.0
    col_pairs = np.sum(n_j * (n_j - 1)) / 2.0
    
    precision = cell_pairs / row_pairs if row_pairs > 0 else 0.0
    recall = cell_pairs / col_pairs if col_pairs > 0 else 0.0
    f1_score = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # NMI
    N = float(num_vertices)
    p_i = n_i / N
    p_j = n_j / N
    H_pred = -np.sum(p_i * np.log(p_i))
    H_true = -np.sum(p_j * np.log(p_j))
    
    c_df = contingency_pdf.merge(row_sums_pdf, on='cluster_pred').merge(col_sums_pdf, on='cluster_true')
    p_ij = c_df['n_ij'].to_numpy() / N
    p_i_aligned = c_df['n_i'].to_numpy() / N
    p_j_aligned = c_df['n_j'].to_numpy() / N
    
    valid = p_ij > 0
    p_ij, p_i_aligned, p_j_aligned = p_ij[valid], p_i_aligned[valid], p_j_aligned[valid]
    
    mi = np.sum(p_ij * np.log(p_ij / (p_i_aligned * p_j_aligned)))
    nmi = mi / np.sqrt(H_pred * H_true) if (H_pred * H_true) > 0 else 0.0
    
    return f1_score, nmi

def calculate_graph_structure_profile(edges_df, num_vertices):
    """Compute graph-only structural features once per graph with cuGraph."""
    profile = {}
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if edges_df.empty:
        profile["Max In-Degree"] = 0
        profile["Avg In-Degree"] = 0
        profile["Max Out-Degree"] = 0
        profile["Avg Out-Degree"] = 0
        profile["Avg CC"] = 0
        profile["Max Assortativity (knn)"] = 0
        profile["Avg Assortativity (knn)"] = 0
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            profile[f"In-Degree {p}th"] = 0
            profile[f"Out-Degree {p}th"] = 0
            profile[f"CC {p}th"] = 0
            profile[f"Assortativity {p}th"] = 0
        return profile

    gpu_edges = cudf.from_pandas(
        edges_df[["source", "target", "weight"]]
        .drop_duplicates(subset=["source", "target"])
        .loc[lambda frame: frame["source"] != frame["target"]]
        .astype({"source": "int64", "target": "int64", "weight": "float64"})
    )
    observed_vertices = sorted(
        set(edges_df["source"].astype(int)) | set(edges_df["target"].astype(int))
    )
    vertex_ids = list(observed_vertices)
    vertex_set = set(vertex_ids)
    candidate_vertex = 0 if min(observed_vertices) == 0 else 1
    while len(vertex_ids) < num_vertices:
        if candidate_vertex not in vertex_set:
            vertex_ids.append(candidate_vertex)
            vertex_set.add(candidate_vertex)
        candidate_vertex += 1
    vertices = cudf.Series(vertex_ids, dtype="int64")

    directed_graph = cugraph.Graph(directed=True)
    directed_graph.from_cudf_edgelist(
        gpu_edges,
        source="source",
        destination="target",
        edge_attr="weight",
        vertices=vertices,
    )
    in_degree = directed_graph.in_degree()["degree"].astype("float64")
    out_degree = directed_graph.out_degree()["degree"].astype("float64")
    in_degree_percentiles = in_degree.quantile(percentiles).to_pandas().to_dict()
    out_degree_percentiles = out_degree.quantile(percentiles).to_pandas().to_dict()

    profile["Max In-Degree"] = float(in_degree.max())
    profile["Avg In-Degree"] = float(in_degree.mean())
    profile["Max Out-Degree"] = float(out_degree.max())
    profile["Avg Out-Degree"] = float(out_degree.mean())

    undirected_graph = cugraph.Graph(directed=False)
    undirected_graph.from_cudf_edgelist(
        gpu_edges[["source", "target"]],
        source="source",
        destination="target",
        vertices=vertices,
    )
    triangle_df = cugraph.triangle_count(undirected_graph)
    degree_df = undirected_graph.degree()
    cc_df = triangle_df.merge(degree_df, on="vertex", how="right")
    cc_df["cc"] = (
        2.0 * cc_df["counts"]
        / (cc_df["degree"] * (cc_df["degree"] - 1.0))
    )
    cc_df["cc"] = cc_df["cc"].where(cc_df["degree"] > 1, 0.0).fillna(0.0)
    cc_series = cc_df["cc"].astype("float64")
    cc_percentiles = cc_series.quantile(percentiles).to_pandas().to_dict()

    edges_sym = cudf.concat([
        gpu_edges[["source", "target"]],
        gpu_edges[["target", "source"]].rename(
            columns={"target": "source", "source": "target"}
        ),
    ]).drop_duplicates()
    edges_sym = edges_sym.merge(
        degree_df,
        left_on="source",
        right_on="vertex",
        how="inner",
    ).rename(columns={"degree": "degree_source"}).drop(columns=["vertex"])
    edges_sym = edges_sym.merge(
        degree_df,
        left_on="target",
        right_on="vertex",
        how="inner",
    ).rename(columns={"degree": "degree_target"}).drop(columns=["vertex"])
    node_knn = edges_sym.groupby("source")["degree_target"].mean()
    knn_percentiles = node_knn.quantile(percentiles).to_pandas().to_dict()

    profile["Avg CC"] = float(cc_series.mean())
    profile["Max Assortativity (knn)"] = float(node_knn.max())
    profile["Avg Assortativity (knn)"] = float(node_knn.mean())
    for p in percentiles:
        label = int(p * 100)
        profile[f"In-Degree {label}th"] = float(in_degree_percentiles[p])
        profile[f"Out-Degree {label}th"] = float(out_degree_percentiles[p])
        profile[f"CC {label}th"] = float(cc_percentiles[p])
        profile[f"Assortativity {label}th"] = float(knn_percentiles[p])

    return profile


def as_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=np.nan):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def first_numeric_value(data, keys, default=np.nan):
    for key in keys:
        if key in data:
            value = as_float(data.get(key), default=np.nan)
            if not np.isnan(value):
                return value
    return default


def first_integer_value(data, keys, default=np.nan):
    for key in keys:
        if key in data:
            value = as_int(data.get(key), default=np.nan)
            if not pd.isna(value):
                return value
    return default


def parse_sbp_out_metrics(out_path):
    metrics = {
        "runtime_seconds": np.nan,
        "iterations": np.nan,
    }

    if not os.path.exists(out_path):
        return metrics

    with open(out_path, "r", errors="replace") as f:
        text = f.read()

    runtime_patterns = [
        r"(?i)\bruntime(?:\s*\(s\)|\s+seconds?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?i)\belapsed(?:\s+time)?(?:\s*\(s\)|\s+seconds?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?i)\btime(?:\s+taken)?(?:\s*\(s\)|\s+seconds?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in runtime_patterns:
        matches = re.findall(pattern, text)
        if matches:
            metrics["runtime_seconds"] = as_float(matches[-1])
            break

    iteration_patterns = [
        r"(?i)\b(?:number\s+of\s+)?(?:total\s+)?(?:async\s+)?iterations?\s*[:=]\s*([0-9]+)",
        r"(?i)\b(?:completed|ran|finished)\s+([0-9]+)\s+(?:async\s+)?iterations?\b",
        r"(?i)\b([0-9]+)\s+(?:async\s+)?iterations?\b",
    ]
    for pattern in iteration_patterns:
        matches = re.findall(pattern, text)
        if matches:
            metrics["iterations"] = as_int(matches[-1])
            break

    return metrics


def load_performance_metrics(run_dir, result_json):
    performance_path = os.path.join(run_dir, "performance.json")
    out_path = os.path.join(run_dir, "sbp.out")
    performance_data = {}

    if os.path.exists(performance_path):
        with open(performance_path, "r") as f:
            performance_data = json.load(f)
        out_path = performance_data.get("sbp_out_file", out_path)

    out_metrics = parse_sbp_out_metrics(out_path)

    runtime = first_numeric_value(
        result_json,
        ["Runtime (s)", "Runtime", "runtime", "runtime_seconds", "elapsed_seconds"],
    )
    if np.isnan(runtime):
        runtime = first_numeric_value(
            performance_data,
            ["runtime_seconds", "Runtime (s)", "Runtime", "runtime", "elapsed_seconds"],
        )
    if np.isnan(runtime):
        runtime = out_metrics["runtime_seconds"]

    iterations = first_integer_value(
        result_json,
        ["Iterations", "iterations", "Num Iterations", "Number of Iterations", "ASYNC_ITERS"],
    )
    if pd.isna(iterations):
        iterations = first_integer_value(
            performance_data,
            ["iterations", "Iterations", "num_iterations"],
        )
    if pd.isna(iterations):
        iterations = out_metrics["iterations"]

    return {
        "runtime_seconds": runtime,
        "iterations": iterations,
        "performance_file": performance_path if os.path.exists(performance_path) else np.nan,
        "sbp_out_file": out_path if os.path.exists(out_path) else np.nan,
    }

def load_json_document(path):
    """Load a JSON file, tolerating concatenated JSON objects by using the last one."""
    with open(path, 'r') as f:
        text = f.read()

    decoder = json.JSONDecoder()
    index = 0
    documents = []
    text_length = len(text)

    while index < text_length:
        while index < text_length and text[index].isspace():
            index += 1
        if index >= text_length:
            break

        document, next_index = decoder.raw_decode(text, index)
        documents.append(document)
        index = next_index

    if len(documents) > 1:
        print(f"Warning: found {len(documents)} concatenated JSON documents in {path}; using the last one.")

    return documents[-1]

def load_graph_generation_metadata(data_dir):
    metrics_path = os.path.join(data_dir, "selected_graph_metrics.csv")
    if not os.path.exists(metrics_path):
        return {}

    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        print(f"Warning: could not load graph metadata from {metrics_path}: {exc}")
        return {}

    if "dataset" not in df.columns:
        return {}

    return {
        str(row["dataset"]): row.to_dict()
        for _, row in df.iterrows()
    }

# ==========================================
# 3. VISUALIZATION AND EXECUTION
# ==========================================
def build_undirected_cugraph(edges_df, vertices):
    """Build an undirected cuGraph graph from a host edge-list DataFrame."""
    gpu_edges = cudf.from_pandas(
        edges_df[["source", "target"]].astype(
            {"source": "int64", "target": "int64"}
        )
    )
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(
        gpu_edges,
        source="source",
        destination="target",
        vertices=cudf.Series(sorted(vertices), dtype="int64"),
    )
    return graph


def draw_cugraph_panel(ax, graph, positions, colors, title):
    """Render a cuGraph graph with Matplotlib without a graph-library shim."""
    edge_list = graph.view_edge_list().to_pandas()
    source_col, target_col = edge_list.columns[:2]
    segments = [
        (positions[int(source)], positions[int(target)])
        for source, target in edge_list[[source_col, target_col]].itertuples(index=False)
        if int(source) in positions and int(target) in positions
    ]
    if segments:
        ax.add_collection(
            LineCollection(segments, colors="black", linewidths=0.5, alpha=0.1)
        )
    nodes = sorted(positions)
    ax.scatter(
        [positions[node][0] for node in nodes],
        [positions[node][1] for node in nodes],
        c=[colors.get(node, 0) for node in nodes],
        cmap=plt.cm.Set1,
        s=50,
        alpha=0.9,
        linewidths=0.0,
    )
    ax.autoscale_view()
    ax.set_title(title, fontsize=14)
    ax.axis("off")


def visualize_comparison(G, true_clusters, pred_clusters, dataset_name, metrics, out_dir="."):
    """Draw side-by-side cuGraph ForceAtlas2 layouts."""
    layout = cugraph.force_atlas2(G, max_iter=50, random_state=42).to_pandas()
    positions = {
        int(row.vertex): (float(row.x), float(row.y))
        for row in layout.itertuples(index=False)
    }
    true_colors = true_clusters.set_index("vertex")["cluster"].astype(int).to_dict()
    pred_colors = pred_clusters.set_index("vertex")["cluster"].astype(int).to_dict()
    metric_str = (f"Modularity: {metrics['Mod']:.3f} | Conductance: {metrics['Cond']:.3f}\n"
                  f"F1: {metrics['F1']:.3f} | NMI: {metrics['NMI']:.3f} | Inv.Hnorm: {metrics['InvHnorm']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    draw_cugraph_panel(
        axes[0], G, positions, true_colors, f"Ground Truth ({dataset_name})"
    )
    draw_cugraph_panel(
        axes[1], G, positions, pred_colors, f"SBP Prediction\n{metric_str}"
    )
    fig.tight_layout()
    viz_path = os.path.join(out_dir, f"{dataset_name}_viz.png")
    fig.savefig(viz_path, dpi=200)
    print(f"Saved visualization to {viz_path}")
    plt.close(fig)

def add_parameter_rankings(df_results):
    """Rank parameters independently for each graph.

    Primary objective is ground-truth recovery, with runtime used only as a
    final tie-breaker. Rank 1 is best for that graph.
    """
    df_ranked = df_results.copy()
    df_ranked["Parameter ID"] = (
        df_ranked["Parameter"].astype(str).str.replace("Param", "", regex=False).astype(int)
    )
    df_ranked["Rank Score"] = (
        df_ranked["F1 Score"].fillna(-np.inf) * 1_000_000
        + df_ranked["NMI"].fillna(-np.inf) * 1_000
        + df_ranked["Inverse H_norm"].fillna(-np.inf)
        - df_ranked["Directed Conductance"].fillna(0) * 0.001
        - df_ranked["Runtime"].fillna(0) * 0.000001
    )
    df_ranked["Parameter Rank"] = (
        df_ranked.groupby("Graph Name")["Rank Score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df_ranked.sort_values(["Graph Name", "Parameter Rank"])

def visualize_parameter_rankings(df_ranked, result_dir):
    rank_csv = os.path.join(result_dir, "SBM_parameter_rankings.csv")
    avg_csv = os.path.join(result_dir, "SBM_average_parameter_ranking.csv")
    heatmap_path = os.path.join(result_dir, "SBM_parameter_rank_heatmap.png")
    avg_plot_path = os.path.join(result_dir, "SBM_average_parameter_rank.png")

    rank_cols = [
        "Graph Name", "Parameter", "Parameter ID", "Parameter Rank", "Rank Score",
        "F1 Score", "NMI", "Inverse H_norm", "Directed Modularity",
        "Directed Conductance", "Runtime", "Iterations", "TEPS"
    ]
    df_ranked[rank_cols].to_csv(rank_csv, index=False)

    avg_rank = (
        df_ranked.groupby(["Parameter", "Parameter ID"], as_index=False)
        .agg(
            Average_Rank=("Parameter Rank", "mean"),
            Median_Rank=("Parameter Rank", "median"),
            Wins=("Parameter Rank", lambda s: int((s == 1).sum())),
            Average_F1=("F1 Score", "mean"),
            Average_NMI=("NMI", "mean"),
            Average_Runtime=("Runtime", "mean"),
            Average_Iterations=("Iterations", "mean"),
            Average_TEPS=("TEPS", "mean"),
        )
        .sort_values(["Average_Rank", "Median_Rank", "Parameter ID"])
    )
    avg_rank.to_csv(avg_csv, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(
        avg_rank["Parameter"],
        avg_rank["Average_Rank"],
        color="#3b82f6",
        edgecolor="#1f2937",
        linewidth=0.4
    )
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Rank (lower is better)")
    plt.title("Average SBP Parameter Rank Across Graphs")
    plt.tight_layout()
    plt.savefig(avg_plot_path, dpi=200)
    plt.close()

    heatmap_df = (
        df_ranked.pivot(index="Graph Name", columns="Parameter ID", values="Parameter Rank")
        .sort_index()
    )
    height = max(8, min(40, 0.03 * len(heatmap_df)))
    plt.figure(figsize=(14, height))
    plt.imshow(heatmap_df, aspect="auto", cmap="viridis_r", interpolation="nearest")
    plt.colorbar(label="Parameter Rank (1 = best)")
    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)),
        labels=[f"P{int(c)}" for c in heatmap_df.columns],
        rotation=90
    )
    plt.yticks([])
    plt.xlabel("Parameter")
    plt.ylabel("Graphs")
    plt.title("Per-Graph SBP Parameter Rankings")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"Saved parameter rankings to {rank_csv}")
    print(f"Saved average parameter rankings to {avg_csv}")
    print(f"Saved ranking visualizations to {avg_plot_path} and {heatmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize SBP results on SBM graphs.")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "Data", "SBM"),
                        help="Directory containing the generated SBM graph TSV files.")
    parser.add_argument("--result_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "Results", "SBP"),
                        help="Directory containing the SBP result.json files.")
    parser.add_argument("--param_file", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "TopDown-SBP", "configs", "params.conf"),
                        help="SBP parameter sweep file used for this run.")
    parser.add_argument("--ranks", type=int, default=int(os.environ.get("MPI_RANKS", 2)),
                        help="MPI ranks used for each SBP run.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SBP_THREADS", 6)),
                        help="Threads used per SBP rank.")
    parser.add_argument("--nodes", type=int, default=1,
                        help="Nodes used for each SBP run.")
    parser.add_argument("--run-shard-index", type=int, default=0,
                        help="Evaluate runs where run_index %% run_shard_count equals this index.")
    parser.add_argument("--run-shard-count", type=int, default=1,
                        help="Number of run-level evaluation shards.")
    parser.add_argument("--run-shard-label", type=str, default="",
                        help="Label used in the shard CSV filename.")
    parser.add_argument("--combine-shards", action="store_true",
                        help="Combine SBM_results_shard_*.csv files and write final rankings.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    RESULT_DIR = args.result_dir

    if not os.path.exists(RESULT_DIR):
        print(f"Result directory {RESULT_DIR} not found. Run the bash script first.")
        exit()

    if args.run_shard_count < 1:
        raise ValueError("--run-shard-count must be positive")
    if args.run_shard_index < 0 or args.run_shard_index >= args.run_shard_count:
        raise ValueError("--run-shard-index must be in [0, run_shard_count)")

    CSV_COLUMNS = [
        "Algorithm", "Parameter", "Run", 
        "SUBGRAPHS", "BATCHES", "CACHE_SIZE", "DEGREEPRODUCTSORT", "SPLITINIT", "SPLIT", "MH_PERCENT", "ALGORITHM", "OVERLAP", "NONPARAMETRIC", "NODELTA", "MIX", "GREEDY", "APPROXIMATE", "ASYNC_ITERS",
        "Graph Name", "Configured Degree Distribution", "Configured Clustering Difficulty",
        "Generator Degree Mode", "Generator Target Mu", "Generator Actual Mu",
        "Generator Community Imbalance", "Generator Assortativity Mode",
        "Generator Clustering Strength", "Generator Seed",
        "Vertices", "Edges",
        "Max In-Degree", "Avg In-Degree", "Max Out-Degree", "Avg Out-Degree", "Avg CC", 
        "Max Assortativity (knn)", "Avg Assortativity (knn)", 
        "In-Degree 10th", "Out-Degree 10th", "CC 10th", "Assortativity 10th", 
        "In-Degree 20th", "Out-Degree 20th", "CC 20th", "Assortativity 20th", 
        "In-Degree 30th", "Out-Degree 30th", "CC 30th", "Assortativity 30th", 
        "In-Degree 40th", "Out-Degree 40th", "CC 40th", "Assortativity 40th", 
        "In-Degree 50th", "Out-Degree 50th", "CC 50th", "Assortativity 50th", 
        "In-Degree 60th", "Out-Degree 60th", "CC 60th", "Assortativity 60th", 
        "In-Degree 70th", "Out-Degree 70th", "CC 70th", "Assortativity 70th", 
        "In-Degree 80th", "Out-Degree 80th", "CC 80th", "Assortativity 80th", 
        "In-Degree 90th", "Out-Degree 90th", "CC 90th", "Assortativity 90th", 
        "Directed Modularity", "Directed Conductance", "Normalized MDL", "Inverse H_norm", "F1 Score", "NMI", 
        "Runtime", "Iterations", "TEPS", "TNPS",
        "Parameter ID", "Parameter Rank", "Rank Score",
        "Average Utilization", "Max Utilization", "Max Memory", "Nodes", "Ranks", "Threads"
    ]

    if args.combine_shards:
        shard_files = sorted(
            os.path.join(RESULT_DIR, name)
            for name in os.listdir(RESULT_DIR)
            if name.startswith("SBM_results_shard_") and name.endswith(".csv")
        )
        if not shard_files:
            print(f"No shard CSV files found in {RESULT_DIR}")
            exit(1)

        df_results = pd.concat((pd.read_csv(path) for path in shard_files), ignore_index=True)
        df_results = add_parameter_rankings(df_results)
        out_csv = os.path.join(RESULT_DIR, "SBM_results_combined.csv")
        df_results.to_csv(out_csv, index=False)
        print(f"\nSaved combined CSV to {out_csv}")
        visualize_parameter_rankings(df_results, RESULT_DIR)
        exit(0)

    csv_data = []
    run_index = 0
    
    # Parse params.conf
    params_dict = {}
    param_file = args.param_file
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split('|')
                if len(parts) >= 16:
                    idx = parts[0]
                    params_dict[f"Param{idx}"] = {
                        "SUBGRAPHS": parts[1],
                        "BATCHES": parts[2],
                        "CACHE_SIZE": parts[3],
                        "DEGREEPRODUCTSORT": parts[4],
                        "SPLITINIT": parts[5],
                        "SPLIT": parts[6],
                        "MH_PERCENT": parts[7],
                        "ALGORITHM": parts[8],
                        "OVERLAP": parts[9],
                        "NONPARAMETRIC": parts[10],
                        "NODELTA": parts[11],
                        "MIX": parts[12],
                        "GREEDY": parts[13],
                        "APPROXIMATE": parts[14],
                        "ASYNC_ITERS": parts[15]
                    }

    graph_profile_cache = {}
    graph_generation_metadata = load_graph_generation_metadata(DATA_DIR)

    for dataset_name in sorted(os.listdir(RESULT_DIR)):
        dataset_dir = os.path.join(RESULT_DIR, dataset_name)
        if not os.path.isdir(dataset_dir): continue
        
        for param_dir in sorted(os.listdir(dataset_dir)):
            run_dir = os.path.join(dataset_dir, param_dir)
            if not os.path.isdir(run_dir): continue

            current_run_index = run_index
            run_index += 1
            if current_run_index % args.run_shard_count != args.run_shard_index:
                continue

            json_path = os.path.join(run_dir, "result.json")
            edge_path = os.path.join(DATA_DIR, f"{dataset_name}.tsv")
            gt_path = os.path.join(DATA_DIR, f"{dataset_name}_truePartition.tsv")
            
            if not os.path.exists(json_path) or not os.path.exists(edge_path):
                continue
                
            print(f"\nEvaluating: {dataset_name} ({param_dir})")
            
            # 1. Load Data
            edges_df = pd.read_csv(edge_path, sep='\t', header=None, names=['source', 'target', 'weight'])
            true_clusters_df = pd.read_csv(gt_path, sep='\t', header=None, names=['vertex', 'cluster'])
            
            data = load_json_document(json_path)
            
            # Format SBP prediction into DataFrame
            # Results is a flat list: index=vertex-1 (0-based), value=cluster
            node_comm_list = data.get("Results", [])
            if len(node_comm_list) > 0 and isinstance(node_comm_list[0], (list, tuple)):
                # Old format: list of [vertex, cluster] pairs
                pred_clusters_df = pd.DataFrame(node_comm_list, columns=['vertex', 'cluster'])
            else:
                # New format: flat list of cluster IDs, positionally indexed (1-indexed vertices)
                pred_clusters_df = pd.DataFrame({
                    'vertex': range(1, len(node_comm_list) + 1),
                    'cluster': node_comm_list
                })
            
            # 2. Calculate Metrics
            num_vertices = len(true_clusters_df)
            mod_score = calculate_modularity(edges_df, pred_clusters_df)
            cond_score = calculate_conductance(edges_df, pred_clusters_df)
            f1_score, nmi_score = calculate_f1_nmi(pred_clusters_df, true_clusters_df, num_vertices)
            inv_hnorm = calculate_inverse_hnorm(edges_df, pred_clusters_df)

            metrics = {"Mod": mod_score, "Cond": cond_score, "F1": f1_score, "NMI": nmi_score, "InvHnorm": inv_hnorm}
            print(f"  -> Modularity: {mod_score:.4f} | Conductance: {cond_score:.4f}")
            print(f"  -> F1 Score:   {f1_score:.4f} | NMI: {nmi_score:.4f}")
            print(f"  -> Inv. Hnorm: {inv_hnorm:.4f}")
            
            # 3. Visualize only the 1k-node graph runs; force-directed layouts
            # are expensive for the larger SBM graphs.
            if num_vertices == 1000:
                G = build_undirected_cugraph(
                    edges_df,
                    set(true_clusters_df["vertex"].astype(int)),
                )
                visualize_comparison(G, true_clusters_df, pred_clusters_df, dataset_name, metrics, out_dir=run_dir)
            else:
                print(f"Skipping per-run visualization for {num_vertices}-node graph.")

            # 4. Save to CSV data
            row = {c: np.nan for c in CSV_COLUMNS}
            row["Algorithm"] = "SBP"
            row["Parameter"] = param_dir
            row["Run"] = 1
            row["Graph Name"] = dataset_name

            graph_metadata = graph_generation_metadata.get(dataset_name, {})
            row["Configured Degree Distribution"] = graph_metadata.get("degree_mode", np.nan)
            row["Configured Clustering Difficulty"] = graph_metadata.get("target_mu", np.nan)
            row["Generator Degree Mode"] = graph_metadata.get("degree_mode", np.nan)
            row["Generator Target Mu"] = graph_metadata.get("target_mu", np.nan)
            row["Generator Actual Mu"] = graph_metadata.get("actual_mu", np.nan)
            row["Generator Community Imbalance"] = graph_metadata.get("community_imbalance", np.nan)
            row["Generator Assortativity Mode"] = graph_metadata.get("assortativity_mode", np.nan)
            row["Generator Clustering Strength"] = graph_metadata.get("clustering_strength", np.nan)
            row["Generator Seed"] = graph_metadata.get("seed", np.nan)
            
            if param_dir in params_dict:
                for k, v in params_dict[param_dir].items():
                    row[k] = v

            row["Directed Modularity"] = mod_score
            row["Directed Conductance"] = cond_score
            row["Normalized MDL"] = 1.0 - inv_hnorm
            row["Inverse H_norm"] = inv_hnorm
            row["F1 Score"] = f1_score
            row["NMI"] = nmi_score

            performance_metrics = load_performance_metrics(run_dir, data)
            runtime = performance_metrics["runtime_seconds"]
            row["Runtime"] = runtime
            row["Iterations"] = performance_metrics["iterations"]
            
            edges_count = edges_df['weight'].sum() if not edges_df.empty else 0
            vertices_count = num_vertices
            row["Vertices"] = vertices_count
            row["Edges"] = edges_count
            
            if runtime and not np.isnan(runtime) and runtime > 0:
                row["TEPS"] = edges_count / runtime
                row["TNPS"] = vertices_count / runtime

            row["Ranks"] = args.ranks
            row["Threads"] = args.threads
            row["Nodes"] = args.nodes

            if dataset_name not in graph_profile_cache:
                graph_profile_cache[dataset_name] = calculate_graph_structure_profile(edges_df, num_vertices)
            row.update(graph_profile_cache[dataset_name])

            csv_data.append(row)

    if csv_data:
        df_results = pd.DataFrame(csv_data, columns=CSV_COLUMNS)
        if args.run_shard_count == 1:
            df_results = add_parameter_rankings(df_results)
            out_csv = os.path.join(RESULT_DIR, "SBM_results_combined.csv")
        else:
            shard_label = args.run_shard_label or f"{args.run_shard_index:04d}-of-{args.run_shard_count:04d}"
            out_csv = os.path.join(RESULT_DIR, f"SBM_results_shard_{shard_label}.csv")
        df_results.to_csv(out_csv, index=False)
        print(f"\nSaved CSV to {out_csv}")
        if args.run_shard_count == 1:
            visualize_parameter_rankings(df_results, RESULT_DIR)
    else:
        print(
            f"No evaluation rows for shard {args.run_shard_index}/{args.run_shard_count} "
            f"in {RESULT_DIR}"
        )
