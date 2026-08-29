# Data preprocessing

The paper uses 6,043 graphs from four dataset groups:

| Graph dataset | Type | Nodes | Edges | Communities | Graphs |
|---|---:|---:|---:|---:|---:|
| Stochastic Block Model (SBM) | Undirected | 1K, 5K, 10K | 5K, 10K, 50K | 4, 8, 16, 32 | 6,000 |
| MIT Graph Challenge | Directed | 1M, 5M | 23.8M, 118.8M | 125, 221 | 8 |
| SNAP-DBLP | Undirected | 317,080 | 1,049,866 | 13,477 | 1 |
| SNAP-Youtube | Undirected | 1,134,890 | 2,987,624 | 8,385 | 1 |
| SNAP-WikiTopcats | Directed | 1,791,489 | 28,511,807 | 17,364 | 1 |
| SNAP-LiveJournal | Directed | 3,997,962 | 34,681,189 | 287,512 | 1 |
| SNAP-Orkut | Undirected | 3,072,441 | 117,185,083 | 6,288,363 | 1 |
| CAIDA | Directed | approximately 4,414,370 | approximately 16,056,211 | N/A | 30 |

## Required format

Every graph must be an uncompressed TSV edge list without a header:

```text
source<TAB>destination<TAB>weight
```

Vertex IDs are integers and weight is numeric. Use weight `1` for an unweighted edge. Do not pass Matrix Market files, incidence matrices, SNAP community lists, or CAIDA PCAP files directly to Top-down SBP.

When a non-overlapping ground-truth partition is available, store it beside the graph as `<graph>_truePartition.tsv`:

```text
vertex<TAB>community
```

The expected local layout is:

```text
Data/
├── SBM/                 # 6,000 generated graphs and truth partitions
├── MITGraphChallenge/   # 8 downloaded graphs and truth partitions
├── SNAP/                # 5 downloaded and converted edge lists
└── CAIDA/               # 30 authorized daily graph edge lists
```

Print or validate the complete manifest with:

```bash
python Data-Preprocessing/dataset_manifest.py show
python Data-Preprocessing/dataset_manifest.py check --data-root Data --check-format
```

## 1. Stochastic Block Model (SBM)

`generate_sbm_corpus.py` generates the complete paper corpus: three `(nodes, edges)` scales, four community counts, and 500 graphs per combination, for a total of 6,000 graphs.

```bash
python Data-Preprocessing/generate_sbm_corpus.py --dry-run

python Data-Preprocessing/generate_sbm_corpus.py \
  --output-dir Data/SBM \
  --workers 8 \
  --seed 42
```

The generator writes `SBPGraph_N<N>_M<M>_K<K>_id<I>.tsv` and the corresponding `_truePartition.tsv`. It also writes topology and coverage summaries for each of the twelve generation strata. The generator requires the RAPIDS environment declared by the repository.

## 2. MIT Graph Challenge

Download the 2022 static stochastic-block-partition datasets from the [MIT Graph Challenge dataset page](https://graphchallenge.mit.edu/data-sets/) or its public `s3://graphchallenge` bucket. Use the adjacency TSV representation; Graph Challenge TSV matrices contain `row`, `column`, and `value` and use 1-based indexing.

The paper uses the four overlap/block-size-variation settings at 1M and 5M vertices. Store these eight edge lists under `Data/MITGraphChallenge/`:

```text
static_lowOverlap_lowBlockSizeVar_1000000_nodes.tsv
static_lowOverlap_lowBlockSizeVar_5000000_nodes.tsv
static_lowOverlap_highBlockSizeVar_1000000_nodes.tsv
static_lowOverlap_highBlockSizeVar_5000000_nodes.tsv
static_highOverlap_lowBlockSizeVar_1000000_nodes.tsv
static_highOverlap_lowBlockSizeVar_5000000_nodes.tsv
static_highOverlap_highBlockSizeVar_1000000_nodes.tsv
static_highOverlap_highBlockSizeVar_5000000_nodes.tsv
```

Store each known partition using the same stem followed by `_truePartition.tsv`. The 20M-vertex Graph Challenge graphs are not part of the paper corpus.

## 3. SNAP

Download the source edge lists from the [Stanford SNAP dataset collection](https://snap.stanford.edu/data/):

| Paper name | SNAP file | Local path |
|---|---|---|
| SNAP-DBLP | `com-dblp.ungraph.txt.gz` | `Data/SNAP/SNAP-DBLP.tsv` |
| SNAP-Youtube | `com-youtube.ungraph.txt.gz` | `Data/SNAP/SNAP-Youtube.tsv` |
| SNAP-WikiTopcats | `wiki-topcats.txt.gz` | `Data/SNAP/SNAP-WikiTopcats.tsv` |
| SNAP-LiveJournal | `com-lj.ungraph.txt.gz` | `Data/SNAP/SNAP-LiveJournal.tsv` |
| SNAP-Orkut | `com-orkut.ungraph.txt.gz` | `Data/SNAP/SNAP-Orkut.tsv` |

SNAP edge lists contain two whitespace-separated vertex IDs and comments beginning with `#`. Convert each download by adding unit weights:

```bash
python Data-Preprocessing/convert_edge_list.py \
  --input downloads/SNAP/com-dblp.ungraph.txt.gz \
  --output Data/SNAP/SNAP-DBLP.tsv \
  --unit-weight
```

Repeat the command using the mappings above. SNAP community files contain overlapping communities and are not valid `vertex<TAB>community` partitions; they are not used as Top-down SBP truth partitions here.

## 4. CAIDA

The CAIDA files are not redistributed. Request access through the [UCSD Network Telescope dataset page](https://www.caida.org/catalog/datasets/telescope-near-real-time_dataset/) and follow the CAIDA Acceptable Use Agreement and Telescope supplement.

The repository requires 30 authorized, anonymized daily graphs, not raw PCAP. Aggregate each day into a weighted directed edge list and store it as:

```text
Data/CAIDA/YYYYMMDD-HHMMSS.tsv
```

Use names such as `Data/CAIDA/20220114-000000.tsv`. Each row is `source<TAB>destination<TAB>packet_count`. CAIDA graphs have no ground-truth partition. Do not commit packets, addresses, credentials, or sensitive derived data.

## Measurement generation

After Top-down SBP has run, `compute_metrics_cpu.py`, `compute_metrics_gpu.py`, or `evaluate_sweep.py` converts the graph/configuration runs into the measurement CSV consumed by AutoTuneTD-SBP. `TEPS` is reported in thousands of processed edges per second, matching the paper.

For the `theta_*/run_*` result layout:

```bash
python Data-Preprocessing/evaluate_sweep.py \
  --graph Data/MITGraphChallenge/static_lowOverlap_lowBlockSizeVar_1000000_nodes.tsv \
  --ground-truth Data/MITGraphChallenge/static_lowOverlap_lowBlockSizeVar_1000000_nodes_truePartition.tsv \
  --results runs/static_lowOverlap_lowBlockSizeVar_1000000_nodes \
  --dataset MITGraphChallenge \
  --result-batch paper \
  --output outputs/mgc_measurements.csv
```

Omit `--ground-truth` for SNAP and CAIDA.
