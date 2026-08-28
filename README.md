# AutoTuneTD-SBP

AutoTuneTD-SBP recommends a Top-down stochastic block partitioning (SBP) configuration for a graph. The tradeoff parameter `alpha` controls the objective: `0` prioritizes performance, `0.5` balances performance and accuracy, and `1` prioritizes accuracy.

Artifact for **ML-Guided Parameter Configuration Selection for Top-Down Stochastic Block Partitioning**, accepted at *IEEE HPEC 2026*.

## Results

The recommended configurations change as `alpha` moves from performance (`0`) to a balance of performance and accuracy (`0.5`) and then accuracy (`1`).

| Graph | `alpha = 0` | `alpha = 0.5` | `alpha = 1` |
| --- | --- | --- | --- |
| CAIDA<br><code>20220114-<br>000000</code> | `SUBGRAPHS=2, BATCHES=2, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=random, SPLIT=single-snowball, MH_PERCENT=0.1, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=0, NODELTA=0, MIX=0, GREEDY=0, APPROXIMATE=0, ASYNC_ITERS=0` | `SUBGRAPHS=2, BATCHES=2, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=random, SPLIT=single-snowball, MH_PERCENT=1.0, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=1, NODELTA=0, MIX=0, GREEDY=0, APPROXIMATE=0, ASYNC_ITERS=0` | `SUBGRAPHS=2, BATCHES=2, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=random, SPLIT=single-snowball, MH_PERCENT=0.9, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=1, NODELTA=0, MIX=0, GREEDY=0, APPROXIMATE=0, ASYNC_ITERS=0` |
| MIT Graph Challenge<br><code>static_highOverlap_<br>highBlockSizeVar_<br>5000000_nodes</code> | `SUBGRAPHS=2, BATCHES=2, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=random, SPLIT=random, MH_PERCENT=0.1, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=1, NODELTA=0, MIX=0, GREEDY=0, APPROXIMATE=0, ASYNC_ITERS=0` | `SUBGRAPHS=2, BATCHES=2, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=random, SPLIT=connectivity-snowball, MH_PERCENT=0.1, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=1, NODELTA=0, MIX=0, GREEDY=1, APPROXIMATE=0, ASYNC_ITERS=0` | `SUBGRAPHS=2, BATCHES=3, CACHE_SIZE=20000, DEGREEPRODUCTSORT=on, SPLITINIT=degree-weighted, SPLIT=connectivity-snowball, MH_PERCENT=0.1, ALGORITHM=hybrid_mcmc, OVERLAP=unk, NONPARAMETRIC=1, NODELTA=0, MIX=0, GREEDY=0, APPROXIMATE=0, ASYNC_ITERS=0` |

See the [CAIDA results](Results/CAIDA/README.md) and [MIT Graph Challenge results](Results/MGC/README.md). The same settings are available in the [CAIDA configuration table](Results/CAIDA/caida_best_observed_by_alpha.csv) and [MIT Graph Challenge configuration table](Results/MGC/mgc_recommended_configurations.csv).

## Requirements

- Python 3.11 or newer
- An NVIDIA GPU and CUDA 12 for RAPIDS preprocessing and metrics
- A C++17 compiler, CMake, OpenMP, and MPI for Top-down SBP

## Installation

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate autotune-tdsbp
```

Alternatively, on a CUDA 12 system:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA 13, replace the CUDA 12 RAPIDS packages in `requirements.txt` with their `-cu13` equivalents.

## Build Top-down SBP

```bash
cmake -S TopDown-SBP -B TopDown-SBP/build -DCMAKE_BUILD_TYPE=Release
cmake --build TopDown-SBP/build --target TopDownSBP -j
```

## Run

Top-down SBP accepts a weighted tab-separated edge list named `<graph>.tsv`:

```text
source_vertex<TAB>target_vertex<TAB>weight
```

Run the paper's parameter sweep:

```bash
TopDown-SBP/scripts/run_parameter_sweep.sh \
  Data/CAIDA/20220114-000000 \
  runs/CAIDA/20220114-000000
```

Compute graph and clustering metrics:

```bash
python Data-Preprocessing/compute_metrics_cpu.py --help
python Data-Preprocessing/compute_metrics_gpu.py --help
```

Train the recommendation model:

```bash
python AutoTuneTD-SBP/train.py \
  --input path/to/training_measurements.csv \
  --output-dir models/xgboost
```

Recommend configurations for a new graph:

```bash
python AutoTuneTD-SBP/recommend.py \
  --model models/xgboost/model.joblib \
  --graph-features path/to/new_graph_features.csv \
  --configurations TopDown-SBP/configs/params.conf \
  --alpha 0.5 \
  --output recommendations.csv
```

See [AutoTuneTD-SBP/README.md](AutoTuneTD-SBP/README.md) and [Data-Preprocessing/README.md](Data-Preprocessing/README.md) for input-column and preprocessing details.

## Parameter configurations

All 27 configurations evaluated in the paper are listed in [params.conf](TopDown-SBP/configs/params.conf). The first row is the reference configuration; every other row changes one setting. Dataset-specific experiments may use only a subset of these configurations.

## Citation

```bibtex
@inproceedings{dey2026autotunetdsbp,
  author    = {Saikat Dey and Wu-chun Feng},
  title     = {ML-Guided Parameter Configuration Selection for Top-Down Stochastic Block Partitioning},
  booktitle = {2026 IEEE High Performance Extreme Computing Conference (HPEC)},
  year      = {2026}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
