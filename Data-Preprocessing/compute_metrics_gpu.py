import os
import json
import argparse
import cudf
import cugraph
import pandas as pd
import numpy as np
import rmm
from scipy.special import gammaln
import gc
import glob
from pathlib import Path
import fcntl

# Use managed memory when a graph does not fit entirely in GPU memory.
rmm.reinitialize(managed_memory=True)

SUCCESS_MARKER = "Job completed successfully"


def parse_csv_list(value):
    if not value:
        return []
    items = []
    for token in value:
        items.extend([part.strip() for part in token.split(',') if part.strip()])
    return items


def parse_graph_param_limits(value):
    limits = {}
    for item in parse_csv_list(value):
        if ':' not in item:
            raise ValueError(f"Invalid graph limit '{item}'. Expected GRAPH:MAX_PARAM.")
        graph, max_param = item.split(':', 1)
        graph = graph.strip()
        max_param = max_param.strip()
        if max_param.startswith("Param"):
            max_param = max_param.replace("Param", "", 1)
        limits[graph] = int(max_param)
    return limits


def param_number(param):
    if isinstance(param, str) and param.startswith("Param") and param[5:].isdigit():
        return int(param[5:])
    return None


def is_successful_run(run_dir):
    for out_file in glob.glob(f"{run_dir}/*.out"):
        try:
            with open(out_file, 'r') as out_f:
                if SUCCESS_MARKER in out_f.read():
                    return True
        except Exception:
            continue
    return False


def find_first_existing(candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def resolve_graph_file(base_data_dir, dataset, graph_name):
    base = Path(base_data_dir)
    candidates = [
        base / dataset / f"{graph_name}.tsv",
        base / dataset / "StaticGraphs" / f"{graph_name}.tsv",
        base / "CAIDA" / "StaticGraphs" / f"{graph_name}.tsv",
        base / "CAIDAGraphs" / "StaticGraphs" / f"{graph_name}.tsv",
    ]

    if dataset in {"OtherWithGT", "Other", "OtherGraphs"}:
        candidates.extend((base / "OtherGraphs" / "Real-Graphs-With-GroundTruth").glob(f"*/{graph_name}.tsv"))

    return find_first_existing([str(path) for path in candidates])


def resolve_ground_truth_file(base_data_dir, dataset, graph_name):
    base = Path(base_data_dir)
    candidates = [
        base / dataset / f"{graph_name}_truePartition.tsv",
    ]

    if dataset in {"OtherWithGT", "Other", "OtherGraphs"}:
        candidates.extend((base / "OtherGraphs" / "Real-Graphs-With-GroundTruth").glob(f"*/{graph_name}_truePartition.tsv"))

    return find_first_existing([str(path) for path in candidates])


def discover_params(base_result_dir, dataset, algo, graph_name, requested_params, graph_param_limits):
    if requested_params == ['all'] or requested_params == ['all',]:
        algo_graph_dir = f"{base_result_dir}/{dataset}/{algo}/{graph_name}"
        if os.path.exists(algo_graph_dir):
            params = sorted(
                [
                    d for d in os.listdir(algo_graph_dir)
                    if os.path.isdir(os.path.join(algo_graph_dir, d)) and d.startswith("Param")
                ],
                key=lambda x: int(x.replace("Param", "")) if x.replace("Param", "").isdigit() else x
            )
        else:
            params = []
    else:
        params = requested_params

    if graph_name in graph_param_limits:
        max_param = graph_param_limits[graph_name]
        params = [
            param for param in params
            if param_number(param) is not None and param_number(param) <= max_param
        ]

    return params


def has_completed_selected_run(base_result_dir, dataset, algorithms, graph_name, requested_params, graph_param_limits):
    for algo in algorithms:
        for param in discover_params(base_result_dir, dataset, algo, graph_name, requested_params, graph_param_limits):
            param_dir = f"{base_result_dir}/{dataset}/{algo}/{graph_name}/{param}"
            if not os.path.exists(param_dir):
                continue
            for timestamp in os.listdir(param_dir):
                run_dir = f"{param_dir}/{timestamp}"
                if os.path.isdir(run_dir) and os.path.exists(f"{run_dir}/result.json") and is_successful_run(run_dir):
                    return True
    return False

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
# 2. METRIC FUNCTIONS
# ==========================================
def calculate_modularity(edges, clusters):
    m = edges['weight'].sum()
    out_deg = edges.groupby('source', as_index=False)['weight'].sum().rename(columns={'source':'vertex', 'weight':'k_out'})
    in_deg = edges.groupby('target', as_index=False)['weight'].sum().rename(columns={'target':'vertex', 'weight':'k_in'})
    nodes = clusters.merge(out_deg, on='vertex', how='left').merge(in_deg, on='vertex', how='left').fillna(0)
    comm = nodes.groupby('cluster').agg({'k_out':'sum', 'k_in':'sum'}).reset_index()
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex').merge(clusters, left_on='target', right_on='vertex')
    e_internal = e_comm[e_comm['cluster_x'] == e_comm['cluster_y']].groupby('cluster_x')['weight'].sum().reset_index().rename(columns={'cluster_x':'cluster', 'weight':'E_c'})
    final = comm.merge(e_internal, on='cluster', how='left').fillna(0)
    final['term'] = (final['E_c']/m) - ((final['k_out']/m) * (final['k_in']/m))
    return final['term'].sum()

def calculate_conductance(edges, clusters):
    out_deg = edges.groupby('source', as_index=False)['weight'].sum().rename(columns={'source':'vertex', 'weight':'k_out'})
    nodes = clusters.merge(out_deg, on='vertex', how='left').fillna(0)
    comm_out = nodes.groupby('cluster')['k_out'].sum().reset_index()
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex').merge(clusters, left_on='target', right_on='vertex')
    e_internal = e_comm[e_comm['cluster_x'] == e_comm['cluster_y']].groupby('cluster_x')['weight'].sum().reset_index().rename(columns={'cluster_x':'cluster', 'weight':'E_c'})
    final = comm_out.merge(e_internal, on='cluster', how='left').fillna(0)
    final['cond'] = (final['k_out'] - final['E_c']) / final['k_out']
    return final['cond'].fillna(0).mean()

def calculate_mdl(edges, clusters):
    E, V, B = int(edges['weight'].sum()), len(clusters), clusters['cluster'].nunique()
    out_deg = edges.groupby('source')['weight'].sum().reset_index().rename(columns={'source':'vertex', 'weight':'k_out'})
    in_deg = edges.groupby('target')['weight'].sum().reset_index().rename(columns={'target':'vertex', 'weight':'k_in'})
    nodes = clusters.merge(out_deg, on='vertex', how='left').merge(in_deg, on='vertex', how='left').fillna(0)
    comm = nodes.groupby('cluster').agg({'k_out':'sum', 'k_in':'sum', 'vertex':'count'}).reset_index().rename(columns={'vertex':'N_r'})
    e_comm = edges.merge(clusters, left_on='source', right_on='vertex').merge(clusters, left_on='target', right_on='vertex')
    M_rc = e_comm.groupby(['cluster_x', 'cluster_y'])['weight'].sum().to_numpy()
    
    K_out, K_in, N_r = comm['k_out'].to_numpy(), comm['k_in'].to_numpy(), comm['N_r'].to_numpy()
    k_out, k_in = nodes['k_out'].to_numpy(), nodes['k_in'].to_numpy()

    S = np.sum(gammaln(M_rc + 1)) - (np.sum(gammaln(K_out + 1)) + np.sum(gammaln(K_in + 1))) + np.sum(gammaln(k_out + 1)) + np.sum(gammaln(k_in + 1))
    S_part = fastlbinom(V - 1, B - 1) + gammaln(V + 1) - np.sum(gammaln(N_r + 1)) + np.log(V)
    S_edges = fastlbinom((B**2) + E - 1, E)
    S_deg = np.sum(log_q_approx(K_out, N_r)) + np.sum(log_q_approx(K_in, N_r))
    actual_mdl = float(S + S_part + S_edges + S_deg)
    
    S_null = gammaln(E + 1) - (2 * gammaln(E + 1)) + np.sum(gammaln(k_out + 1)) + np.sum(gammaln(k_in + 1))
    S_dl_null = np.log(V) + fastlbinom(E, E) + 2 * float(log_q_approx(E, V))
    null_mdl = float(S_null + S_dl_null)
    
    return actual_mdl, null_mdl, (actual_mdl / null_mdl)

def calculate_f1_nmi(pred_clusters, true_clusters, num_vertices):
    """
    Calculates Pairwise F1 and NMI. 
    Skips the Hungarian matching step as these metrics are invariant to label permutations.
    """
    merged = pred_clusters.merge(true_clusters, on='vertex', suffixes=('_pred', '_true'))
    
    # Return "NA" if no matching vertices are found
    if len(merged) == 0:
        return "NA", "NA"

    # Build Contingency Table on GPU
    contingency = merged.groupby(['cluster_pred', 'cluster_true']).size().reset_index()
    contingency.columns = ['cluster_pred', 'cluster_true', 'n_ij']
    
    row_sums = merged.groupby('cluster_pred').size().reset_index()
    row_sums.columns = ['cluster_pred', 'n_i']
    
    col_sums = merged.groupby('cluster_true').size().reset_index()
    col_sums.columns = ['cluster_true', 'n_j']
    
    # Move to Host for fast numpy calculations
    c_df = contingency.to_pandas()
    r_df = row_sums.to_pandas()
    cl_df = col_sums.to_pandas()
    
    n_ij = c_df['n_ij'].to_numpy()
    n_i = r_df['n_i'].to_numpy()
    n_j = cl_df['n_j'].to_numpy()
    
    # Pairwise F1 Score
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
    
    c_df = c_df.merge(r_df, on='cluster_pred').merge(cl_df, on='cluster_true')
    p_ij = c_df['n_ij'].to_numpy() / N
    p_i_aligned = c_df['n_i'].to_numpy() / N
    p_j_aligned = c_df['n_j'].to_numpy() / N
    
    valid = p_ij > 0
    p_ij, p_i_aligned, p_j_aligned = p_ij[valid], p_i_aligned[valid], p_j_aligned[valid]
    
    mi = np.sum(p_ij * np.log(p_ij / (p_i_aligned * p_j_aligned)))
    nmi = mi / np.sqrt(H_pred * H_true) if (H_pred * H_true) > 0 else 0.0
    
    return f1_score, nmi

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Topology and Community Metrics")
    
    parser.add_argument("--dataset_idx", type=int, default=-1, help="Index of the dataset to run for SLURM arrays")
    parser.add_argument("--algorithms", nargs='+', required=True, help="List of algorithms (e.g., SBP)")
    parser.add_argument("--parameters", nargs='+', required=True, help="List of parameters")
    parser.add_argument("--base_data_dir", type=str, required=True, help="Base directory for datasets")
    parser.add_argument("--base_result_dir", type=str, required=True, help="Base directory for results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV outputs")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    # Pass the path to the 27-configuration params.conf from the user's local
    # Top-down SBP checkout. This repository does not include that checkout.
    parser.add_argument("--params_conf", type=str, required=False, help="Path to params.conf")
    parser.add_argument("--graphs", nargs='*', default=[], help="Optional graph names to process")
    parser.add_argument("--graph_param_limits", nargs='*', default=[], help="Optional GRAPH:MAX_PARAM entries; process Param1..MAX_PARAM for each graph")
    parser.add_argument("--completed_only", action="store_true", help="Only evaluate runs with a result.json and a successful slurm output")
    parser.add_argument("--output_name", type=str, default=None, help="Optional output CSV filename")
    
    args = parser.parse_args()

    param_dict = {}
    if args.params_conf and os.path.exists(args.params_conf):
        with open(args.params_conf, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('|')
                    if len(parts) >= 16:
                        idx = f"Param{parts[0]}"
                        param_dict[idx] = {
                            "SUBGRAPHS": parts[1], "BATCHES": parts[2], "CACHE_SIZE": parts[3],
                            "DEGREEPRODUCTSORT": parts[4], "SPLITINIT": parts[5], "SPLIT": parts[6],
                            "MH_PERCENT": parts[7], "ALGORITHM": parts[8], "OVERLAP": parts[9],
                            "NONPARAMETRIC": parts[10], "NODELTA": parts[11], "MIX": parts[12],
                            "GREEDY": parts[13], "APPROXIMATE": parts[14], "ASYNC_ITERS": parts[15]
                        }

    PERCENTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    requested_graphs = parse_csv_list(args.graphs)
    graph_param_limits = parse_graph_param_limits(args.graph_param_limits)

    graph_names = set()
    for algo in args.algorithms:
        algo_dir = os.path.join(args.base_result_dir, args.dataset, algo)
        if os.path.exists(algo_dir):
            for d in os.listdir(algo_dir):
                if os.path.isdir(os.path.join(algo_dir, d)):
                    graph_names.add(d)

    if requested_graphs:
        graph_names = graph_names.intersection(set(requested_graphs))
    if graph_param_limits:
        graph_names = graph_names.intersection(set(graph_param_limits.keys()))
    
    graph_names = sorted(list(graph_names))
    if not graph_names:
        print(f"No graphs found in {args.base_result_dir}/{args.dataset}/ for algorithms {args.algorithms}")
        exit(0)

    if 0 <= args.dataset_idx < len(graph_names):
        datasets_to_process = [graph_names[args.dataset_idx]]
    else:
        datasets_to_process = graph_names

    final_results = []

    for graph_name in datasets_to_process:
        print(f"\n{'='*70}")
        print(f"Computing graph topology: {graph_name}")
        print(f"{'='*70}")

        if args.completed_only and not has_completed_selected_run(
            args.base_result_dir, args.dataset, args.algorithms, graph_name, args.parameters, graph_param_limits
        ):
            print(f"⏭️ No completed selected runs with result.json for {graph_name}. Skipping graph.")
            continue

        file_path = resolve_graph_file(args.base_data_dir, args.dataset, graph_name)
        if not file_path:
            print(f"Graph file not found for {args.dataset}/{graph_name}; skipping.")
            continue
        
        # --- A. LOAD EDGES ---
        print("Loading edges...")
        edges_df = cudf.read_csv(file_path, sep='\t', header=None, names=['source', 'target', 'weight'])
        edges_df = edges_df.drop_duplicates(subset=['source', 'target'])
        edges_df = edges_df[edges_df['source'] != edges_df['target']]
        if edges_df['weight'].isnull().any() or (edges_df['weight'] == 0).all():
            edges_df['weight'] = 1.0

        num_edges = len(edges_df)
        edge_min = edges_df['source'].min()
        
        # --- B. GRAPH PROPERTIES ---
        print("Computing topological features...")
        G_dir = cugraph.Graph(directed=True)
        G_dir.from_cudf_edgelist(edges_df, source='source', destination='target', edge_attr='weight')
        num_vertices = G_dir.number_of_vertices()
        
        in_deg = G_dir.in_degree()
        out_deg = G_dir.out_degree()
        
        max_in_deg = in_deg['degree'].max()
        max_out_deg = out_deg['degree'].max()
        avg_in_deg = in_deg['degree'].mean()
        avg_out_deg = out_deg['degree'].mean()
        in_deg_pct = in_deg['degree'].quantile(PERCENTILES).to_pandas().to_dict()
        out_deg_pct = out_deg['degree'].quantile(PERCENTILES).to_pandas().to_dict()

        G_undir = cugraph.Graph(directed=False)
        G_undir.from_cudf_edgelist(edges_df, source='source', destination='target')
        
        triangle_df = cugraph.triangle_count(G_undir)
        degree_df = G_undir.in_degree() 
        cc_df = triangle_df.merge(degree_df, on='vertex')
        cc_df['cc'] = (2.0 * cc_df['counts']) / (cc_df['degree'] * (cc_df['degree'] - 1.0))
        cc_df['cc'] = cc_df['cc'].fillna(0.0)
        
        avg_cc = cc_df['cc'].mean()
        cc_pct = cc_df['cc'].quantile(PERCENTILES).to_pandas().to_dict()

        edges_sym = cudf.concat([
            edges_df[['source', 'target']],
            edges_df[['target', 'source']].rename(columns={'target': 'source', 'source': 'target'})
        ]).drop_duplicates()
        edges_sym = edges_sym.merge(degree_df, left_on='source', right_on='vertex', how='inner')
        edges_sym = edges_sym.rename(columns={'degree': 'degree_source'}).drop(columns=['vertex'])
        edges_sym = edges_sym.merge(degree_df, left_on='target', right_on='vertex', how='inner')
        edges_sym = edges_sym.rename(columns={'degree': 'degree_target'}).drop(columns=['vertex'])

        node_knn = edges_sym.groupby('source')['degree_target'].mean().reset_index()
        node_knn = node_knn.rename(columns={'degree_target': 'knn_i'})
        
        avg_knn = node_knn['knn_i'].mean()
        max_knn = node_knn['knn_i'].max()
        knn_pct = node_knn['knn_i'].quantile(PERCENTILES).to_pandas().to_dict()

        base_features = {
            "Dataset": args.dataset,
            "Graph Name": graph_name,
            "Vertices": num_vertices,
            "Edges": num_edges,
            "Max In-Degree": max_in_deg,
            "Avg In-Degree": avg_in_deg,
            "Max Out-Degree": max_out_deg,
            "Avg Out-Degree": avg_out_deg,
            "Avg CC": avg_cc,
            "Max Assortativity (knn)": max_knn,
            "Avg Assortativity (knn)": avg_knn
        }
        for p in PERCENTILES:
            p_label = int(p * 100)
            base_features[f"In-Degree {p_label}th"] = in_deg_pct.get(p, 0)
            base_features[f"Out-Degree {p_label}th"] = out_deg_pct.get(p, 0)
            base_features[f"CC {p_label}th"] = cc_pct.get(p, 0)
            base_features[f"Assortativity {p_label}th"] = knn_pct.get(p, 0)

        del G_dir, G_undir, in_deg, out_deg, triangle_df, degree_df, cc_df, edges_sym, node_knn
        gc.collect()

        # --- LOAD GROUND TRUTH ---
        true_clusters_df = None
        gt_path = resolve_ground_truth_file(args.base_data_dir, args.dataset, graph_name)
        if gt_path and os.path.exists(gt_path):
            print(f"Loading Ground Truth for {graph_name}...")
            true_clusters_df = cudf.read_csv(gt_path, sep='\t', header=None, names=['vertex', 'cluster'])
            true_clusters_df['vertex'] = true_clusters_df['vertex'].astype('int64')
            true_clusters_df['cluster'] = true_clusters_df['cluster'].astype('int32')
            
            if edge_min != true_clusters_df['vertex'].min():
                true_clusters_df['vertex'] += (edge_min - true_clusters_df['vertex'].min())
        else:
            print(f"Ground-truth partition not found at {gt_path}; skipping F1/NMI.")

        # --- C. LOOP OVER ALGORITHMS AND PARAMETERS ---
        for algo in args.algorithms:
            params_to_process = discover_params(
                args.base_result_dir, args.dataset, algo, graph_name, args.parameters, graph_param_limits
            )

            for param in params_to_process:
                param_dir = f"{args.base_result_dir}/{args.dataset}/{algo}/{graph_name}/{param}"
                
                if not os.path.exists(param_dir):
                    continue

                subdirs = [d for d in os.listdir(param_dir) if os.path.isdir(os.path.join(param_dir, d))]
                if not subdirs:
                    continue

                for timestamp in subdirs:
                    print(f"  -> Evaluating Metrics for: {algo} | {param} | Run: {timestamp}")
                    
                    run_dir = f"{param_dir}/{timestamp}"
                    json_path = f"{run_dir}/result.json"
                    resource_path = f"{run_dir}/resource_usage.csv"
                    hpc_config_path = f"{run_dir}/hpc_config.txt"
                    
                    out_files = glob.glob(f"{run_dir}/*.out")
                    err_files = glob.glob(f"{run_dir}/*.err")
                    
                    is_timeout = False

                    run_succeeded = is_successful_run(run_dir)
                    if args.completed_only and (not os.path.exists(json_path) or not run_succeeded):
                        print(f"    ⏭️ Skipping incomplete run (completed_only).")
                        continue

                    if not os.path.exists(json_path):
                        if not out_files and not err_files:
                            print(f"    ⏳ Job Pending (no .out/.err). Skipping.")
                            continue
                        
                        if err_files:
                            try:
                                with open(err_files[0], 'r') as err_f:
                                    err_content = err_f.read()
                                    if "CANCELLED" in err_content and "DUE TO TIME LIMIT" in err_content:
                                        is_timeout = True
                            except Exception as e:
                                print(f"Could not read {err_files[0]}: {e}")
                                
                        if not is_timeout:
                            print(f"    🏃 Job Running (no json, no timeout). Skipping.")
                            continue
                        else:
                            print(f"    ⏱️ Job Timeout detected. Recording as OOT.")

                    num_nodes, num_ranks, num_threads = np.nan, np.nan, np.nan
                    if os.path.exists(hpc_config_path):
                        try:
                            with open(hpc_config_path, 'r') as hpc_f:
                                for line in hpc_f:
                                    if '=' in line:
                                        key, val = line.strip().split('=', 1)
                                        if key == 'NODES': num_nodes = int(val)
                                        elif key == 'NTASKS_PER_NODE': num_ranks = int(val)
                                        elif key == 'CPUS_PER_TASK': num_threads = int(val)
                        except Exception as e:
                            print(f"Could not read HPC configuration data: {e}")

                    if is_timeout:
                        mod_score, cond_score, norm_mdl, inv_h_norm = "OOT", "OOT", "OOT", "OOT"
                        f1_score, nmi_score, teps, tnps = "OOT", "OOT", "OOT", "OOT"
                        run_time, avg_util, max_util, max_mem = "OOT", "OOT", "OOT", "OOT"
                    else:
                        run_time, avg_util, max_util, max_mem = np.nan, np.nan, np.nan, np.nan
                        if os.path.exists(resource_path):
                            try:
                                res_df = pd.read_csv(resource_path)
                                if not res_df.empty:
                                    run_time = res_df['Timestamp'].max() - res_df['Timestamp'].min()
                                    avg_util = 100.0 - res_df['CPU_Idle'].mean()
                                    max_util = 100.0 - res_df['CPU_Idle'].min()
                                    max_mem = res_df['Mem_Used_MB'].max()
                            except Exception as e:
                                print(f"Could not read resource data: {e}")

                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        node_comm_list = json_data.get("Results", [])

                        clustering_pdf = pd.DataFrame(node_comm_list, columns=['cluster']).reset_index()
                        clustering_pdf.columns = ['vertex', 'cluster']
                        clustering_pdf['vertex'] = clustering_pdf['vertex'].astype('int64') 
                        clustering_pdf['cluster'] = clustering_pdf['cluster'].astype('int32')

                        cluster_min = clustering_pdf['vertex'].min()
                        if edge_min != cluster_min:
                            clustering_pdf['vertex'] += (edge_min - cluster_min)

                        clustering_df = cudf.DataFrame(clustering_pdf)
                        
                        mod_score = calculate_modularity(edges_df, clustering_df)
                        cond_score = calculate_conductance(edges_df, clustering_df)
                        _, _, norm_mdl = calculate_mdl(edges_df, clustering_df)
                        inv_h_norm = 1.0 - norm_mdl

                        # Set defaults to "NA"
                        f1_score, nmi_score = "NA", "NA"
                        if true_clusters_df is not None:
                            f1_score, nmi_score = calculate_f1_nmi(clustering_df, true_clusters_df, num_vertices)

                        if pd.isna(run_time) or run_time == "OOT" or float(run_time) <= 0:
                            teps, tnps = "OOT", "OOT"
                        else:
                            teps = (num_edges / float(run_time)) / 1000.0
                            tnps = (num_vertices / float(run_time)) / 1000.0

                        del clustering_df, clustering_pdf
                        gc.collect()

                    row_data = base_features.copy()
                    row_data.update({
                        "Algorithm": algo,
                        "Parameter": param,
                        "Run": timestamp
                    })
                    
                    if param in param_dict:
                        row_data.update(param_dict[param])

                    row_data.update({
                        "Directed Modularity": mod_score,
                        "Directed Conductance": cond_score,
                        "Normalized MDL": norm_mdl,
                        "Inverse H_norm": inv_h_norm,
                        "F1 Score": f1_score,
                        "NMI": nmi_score,
                        "Runtime": run_time,
                        "TEPS": teps,
                        "TNPS": tnps,
                        "Average Utilization": avg_util,
                        "Max Utilization": max_util,
                        "Max Memory": max_mem,
                        "Nodes": num_nodes,
                        "Ranks": num_ranks,
                        "Threads": num_threads
                    })

                    final_results.append(row_data)

        del edges_df
        gc.collect()

    # ==========================================
    # 4. FINAL EXPORT
    # ==========================================
    results_df = pd.DataFrame(final_results)
    if results_df.empty:
        print("\nNo rows produced for this task; output CSV unchanged.")
        exit(0)
    
    cols = list(results_df.columns)
    base_cols = []
    if 'Dataset' in cols: base_cols.append(cols.pop(cols.index('Dataset')))
    if 'Graph Name' in cols: base_cols.append(cols.pop(cols.index('Graph Name')))
    if 'Algorithm' in cols: base_cols.append(cols.pop(cols.index('Algorithm')))
    if 'Parameter' in cols: base_cols.append(cols.pop(cols.index('Parameter')))
    if 'Run' in cols: base_cols.append(cols.pop(cols.index('Run')))
    
    param_keys = ["SUBGRAPHS", "BATCHES", "CACHE_SIZE", "DEGREEPRODUCTSORT", "SPLITINIT", "SPLIT", "MH_PERCENT", "ALGORITHM", "OVERLAP", "NONPARAMETRIC", "NODELTA", "MIX", "GREEDY", "APPROXIMATE", "ASYNC_ITERS"]
    for pk in param_keys:
        if pk in cols:
            base_cols.append(cols.pop(cols.index(pk)))

    results_df = results_df[base_cols + cols]

    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_name = args.output_name if args.output_name else f"{args.dataset}_ml_dataset_combined.csv"
    csv_out = f"{args.output_dir}/{csv_name}"

    lock_path = f"{csv_out}.lock"
    with open(lock_path, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        file_exists = os.path.isfile(csv_out)
        results_df.to_csv(csv_out, mode='a', index=False, header=not file_exists)
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    print(f"\nDataset appended to: {csv_out}")
