# Accelerated Stochastic Block Partitioning (SBP)

This project provides an optimized implementation of Stochastic Block Partitioning (SBP) for community detection in graphs. It integrates 4 complementary acceleration heuristics into a unified, high-performance framework:

1. **Sampling** via the SamBaS framework [[IEEE HPEC Paper @ IEEE Xplore](https://ieeexplore.ieee.org/document/9926331), [Preprint](https://arxiv.org/abs/2209.04905), [Journal Paper Preprint](https://arxiv.org/abs/2209.04905)]
2. **Shared-memory parallelization** via Hybrid SBP [[ICPP Paper @ ACM](https://dl.acm.org/doi/10.1145/3545008.3545091), [Preprint](https://arxiv.org/abs/2206.14709)]
3. **Distributed-memory parallelization** via EDiSt [[IEEE CLUSTER Paper Preprint](https://arxiv.org/abs/2209.04910)]
4. **Top-Down Computation** via Top-Down SBP [[ACM HPDC Paper](https://dl.acm.org/doi/10.1145/3731545.3731589)]

Our SBP implementation is based on the [Graph Challenge](http://graphchallenge.org) reference implementation and optimized for modern HPC systems with support for multi-threaded (OpenMP) and distributed (MPI) execution.

## Installation

### Requirements

- C++17 compatible compiler (e.g., AMD Clang, GCC, Intel)
- CMake 3.10 or higher
- MPI implementation (OpenMPI, Cray MPICH, etc.)
- OpenMP support

### Build Instructions

```bash
git clone --recursive https://github.com/vtsynergy/IntegratedSBP.git
cd IntegratedSBP
mkdir build
cd build
cmake ..
make
```

### Build Options

For testing and debugging purposes:

- `cmake -DSCOREP=ON ..` - Enable Score-P instrumentation
- `cmake -DVALGRIND=ON ..` - Enable Valgrind instrumentation
- `cmake -DDEBUG=ON ..` - Enable AddressSanitizer (note: will cause test failures due to memory leak detection)

## Usage

### Basic Usage

For a single-node run with OpenMP parallelization:

```bash
./SBP --filepath /path/to/graph --threads <num_threads>
```

### Recommended Usage (Integrated Approach)

For best results with the integrated sampling, shared-memory, top-down computation and distributed-memory approach:

```bash
mpiexec -n <num_cluster_nodes> ./TopDownSBP \
  --filepath <path> \
  -a hybrid_mcmc \
  --threads <num_threads> \
  --batches 2 \
  --distribute none-edge-balanced \
  --nonparametric \
  -m 0.1 \
  --samplingalg expansion_snowball \
  --samplesize 0.5 \
  --cachesize 20000
  --split connectivity-snowball \
  --splitinit random
```

Since most of these options have been made the default in newer builds, and you should get similar results using:

```bash
mpiexec -n <num_cluster_nodes> ./TopDownSBP --filepath <path>
```

### Graph File Formats

The code expects graph files in one of these formats:

- **Matrix Market** (`.mtx`): Standard sparse matrix format
- **Tab-Separated Values** (`.tsv`): Edge list format with tab delimiter

**File naming convention**: If `--filepath=/path/to/graph`, the code will look for:
- Graph: `/path/to/graph.mtx` or `/path/to/graph.tsv`
- Ground truth (optional): `/path/to/graph_truePartition.tsv`

### Key Parameters

#### Algorithm Options

- `-a, --algorithm <type>` - MCMC algorithm to use:
  - `hybrid_mcmc` (default, recommended): Hybrid approach balancing parallelism and accuracy
  - `metropolis_hastings`: Classic MH algorithm (no parallel implementation)
  - `async_gibbs`: Fully asynchronous, faster but may reduce quality

#### Performance Tuning

- `--threads <N>` - Number of OpenMP threads (default: 1)
- `--batches <N>` - Number of batches for parallel algorithms (default: 2, recommended ≥ 2)
- `--cachesize <N>` - Size of logarithm cache (default: 20000)
- `-m, --mh_percent <X>` - Fraction of high-influence vertices for sequential processing (default: 0.1). High-influence vertices are identified by edge degree product (or vertex degree if `--vertex-degree-sort` is used)

#### Sampling

- `--samplesize <X>` - Fraction of vertices to sample (0.0 to 1.0, default: 1.0 = no sampling)
- `--samplingalg <type>` - Sampling algorithm:
  - `random`: Uniform random sampling
  - `max_degree`: Select highest degree vertices
  - `expansion_snowball`: Graph expansion from high-degree seeds

#### Entropy Calculation

- `--parametric` - Use parametric entropy instead of nonparametric (not recommended; nonparametric is default and more accurate)
- `--hastings-correction` - Compute Hastings correction (only useful in parametric mode; disabled by default)
- `--approximate` - Use faster, less accurate block merge (not recommended with nonparametric mode)
- `--degreecorrected` - Degree-corrected entropy (**WARNING**: not fully implemented, will produce poor results with nonparametric mode)

#### Distribution (Multi-node)

- `--distribute <scheme>` - Distribution scheme for MPI:
  - `none` (default): No distribution
  - `none-edge-balanced`: Balanced edge distribution (recommended for MPI)
  - `2hop-round-robin`, `2hop-size-balanced`, `2hop-snowball`: Alternative schemes (less tested)

#### Top-Down SBP

- `--split <type>` - Split strategy for top-down SBP:
  - `connectivity-snowball` (default): Connectivity-aware graph partitioning
  - `random`: Random vertex assignment
  - `snowball`, `single-snowball`: Snowball expansion variants
- `--splitinit <type>` - Split initialization:
  - `random` (default): Random seed selection
  - `degree-weighted`: Weighted by vertex degree
  - `high-degree`: Select highest degree vertex

#### Matrix Storage

- `--matrix_type <type>` - Blockmodel matrix storage backend (default: `sparse_transpose`):
  - `sparse_transpose`: Stores matrix and transpose for fast column access (default, recommended)
  - `sparse`: Basic sparse matrix using hash maps
  - `dense`: Dense 2D storage, better cache locality for graphs with many inter-block edges
- `--csrgraph` - Use CSR (Compressed Sparse Row) format for graph adjacency storage. When omitted (default), the original neighbor-list (vector-of-vectors) format is used. The two formats produce identical results; CSR is slightly faster for some workloads and is GPU-mappable for future offloading.

#### Vertex Processing

- `--vertex-degree-sort` - Use vertex degree instead of edge degree product to identify high-influence vertices (may reduce accuracy)
- `--ordered` - Process vertices in order (disabled by default: vertices processed in a random order)
- `--detach` - Detach degree-1 vertices before community detection

#### Input/Output

- `-f, --filepath <path>` - Path to graph file (without extension)
- `-j, --json <dir>` - Output directory for JSON results (default: `output`)
- `--output_file <name>` - Output filename (default: auto-generated from PID and timestamp)
- `--evaluate` - Evaluate and report accuracy metrics immediately (requires ground truth)
- `--tag <string>` - Custom tag for identifying runs

#### Graph Properties

- `--undirected` - Treat graph as undirected
- `--noduplicates` - Check for and remove duplicate edges
- `--delimiter <char>` - Delimiter for TSV files (default: tab)

### Output

Results are saved as JSON files in the specified output directory. Each file contains:

- Detected community assignments
- Runtime statistics
- Algorithm parameters
- Entropy/DL values
- Evaluation metrics (if ground truth provided and `--evaluate` used)

### Notes

1. Nonparametric entropy (default) improves accuracy. Do not use `--parametric` unless you have a specific reason.
2. Set `--batches` to at least 2 (default) for parallel/distributed runs to improve accuracy.
3. Edge degree product sorting (default) with `-m 0.1` (default) improves accuracy in parallel/distributed runs. `--vertex-degree-sort` and/or lower values of `m` can lead to poor accuracy.
4. For distributed runs, use `--distribute none-edge-balanced`.
5. For sampling, `--samplingalg expansion_snowball` with `--samplesize 0.5` provides good speed/accuracy tradeoff. For very sparse datasets, use `--samplingalg max_degree` instead.
6. Hastings correction is disabled by default. Avoid using `--hastings-correction` and `--approximate` together when in nonparametric mode.
7. For top-down SBP, the recommended `--split connectivity-snowball` (default) and `--splitinit random` (default) provide best accuracy.

## Code Structure

```
IntegratedSBP/
├── include/          # Header files
│   ├── args.hpp      # Command-line argument parsing
│   ├── sbp.hpp       # Main SBP algorithm
│   ├── blockmodel/   # Blockmodel data structures
│   ├── distributed/  # EDiSt distributed components
│   └── ...
├── src/              # Implementation files
│   ├── main.cpp      # Entry point for SBP
│   ├── TopDownSBPMain.cpp  # Entry point for top-down SBP
│   ├── sbp.cpp       # Core SBP logic
│   ├── finetune.cpp  # MCMC algorithms
│   ├── block_merge.cpp     # Block merge phase
│   ├── entropy.cpp   # Entropy calculations
│   ├── distributed/  # Distributed memory components
│   └── ...
├── test/             # Test suite
├── extern/           # External dependencies
├── scripts/          # Utility scripts
└── CMakeLists.txt    # Build configuration
```

## Citation

If you use this code, please cite our relevant papers:

**Integrated SBP (IEEE HPEC 2023):**
```bibtex
@INPROCEEDINGS{Wanye2023IntegratedSBP,
  author={Wanye, Frank and Gleyzer, Vitaliy and Kao, Edward and Feng, Wu-chun},
  booktitle={2023 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={An Integrated Approach for Accelerating Stochastic Block Partitioning}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  month={9},
  address={Waltham, MA},
  doi={10.1109/HPEC58863.2023.10363599}
}
```

**Top-Down SBP (ACM HPDC 2024):**
```bibtex
@INPROCEEDINGS{Wanye2024TopDownSBP,
  author={Wanye, Frank and Gleyzer, Vitaliy and Kao, Edward and Feng, Wu-chun},
  booktitle={Proceedings of the 33rd International Symposium on High-Performance Parallel and Distributed Computing (HPDC '24)},
  title={Top-Down Stochastic Block Partitioning for Scalable Community Detection},
  year={2024},
  publisher={Association for Computing Machinery},
  address={New York, NY, USA},
  doi={10.1145/3731545.3731589}
}
```

**SamBaS (Sampling):**
```bibtex
@INPROCEEDINGS{Gleyzer2022SamBaS,
  author={Gleyzer, Vitaliy and Kao, Edward and Feng, Wu-chun},
  booktitle={2022 IEEE High Performance Extreme Computing Conference (HPEC)},
  title={SamBaS: Sampling-Based Stochastic Block Partitioning},
  year={2022}
}
```

**Hybrid SBP (Shared-memory parallelization):**
```bibtex
@INPROCEEDINGS{Kao2022HybridSBP,
  author={Kao, Edward and Gleyzer, Vitaliy and Feng, Wu-chun},
  booktitle={2022 International Conference on Parallel Processing (ICPP)},
  title={Hybrid Stochastic Block Partitioning for Large-Scale Graphs},
  year={2022}
}
```

**EDiSt (Distributed-memory parallelization):**
```bibtex
@INPROCEEDINGS{Gleyzer2022EDiSt,
  author={Gleyzer, Vitaliy and Kao, Edward and Feng, Wu-chun},
  booktitle={2022 IEEE International Conference on Cluster Computing (CLUSTER)},
  title={EDiSt: Edge-Balanced Distribution for Stochastic Block Partitioning},
  year={2022}
}
```

## Known Issues

- The `--degreecorrected` option is not fully implemented and will produce incorrect results in nonparametric mode (default)
- The `--approximate` option may reduce quality in nonparametric mode (default)
- Some distribution schemes other than `none-edge-balanced` are not fully tested

## License

© Virginia Polytechnic Institute and State University, 2023.

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details. Original fork's license may be updated to LGPL v2.1 in the future.

## Acknowledgments

This work is based on the Graph Challenge Stochastic Block Partitioning reference implementation. We thank the Graph Challenge organizers and the original SBP authors for their foundational work.

## Weighted edge input

Graph rows may use either unweighted or weighted form:

```text
src dst
src dst weight
```

Vertex IDs remain 1-indexed in input files. The optional `weight` must be a positive integer. A row such as `1 2 3` is represented as three parallel unit edges from vertex 1 to vertex 2, matching the integer edge-count model used by SBP.

MatrixMarket coordinate files may also provide a third integer value per edge row. For symmetric or `--undirected` MatrixMarket graphs, non-self edges are mirrored with the same weight.
