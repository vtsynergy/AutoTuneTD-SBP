# AutoTuneTD-SBP

AutoTuneTD-SBP recommends a Top-down stochastic block partitioning (SBP) configuration for a graph. The tradeoff parameter `alpha` controls the objective: `0` prioritizes performance, `0.5` balances performance and accuracy, and `1` prioritizes accuracy.

Artifact for **ML-Guided Parameter Configuration Selection for Top-Down Stochastic Block Partitioning**, accepted at *IEEE HPEC 2026*.

## Reusability and Broader Impact (CAIDA and MIT Graph Challenge Graphs)

In addition to the AutoTuneTD-SBP codebase, we provide recommended parameter configurations for the top-down SBP graph clustering algorithm for publicly available CAIDA graphs and [MIT Graph Challenge datasets](https://graphchallenge.mit.edu/data-sets/), as identified by AutoTuneTD-SBP. Researchers can use these configurations to run top-down SBP on these graphs without first running AutoTuneTD-SBP.

For each of the graphs, we provide three sets of recommended parameter configurations (obtained via AutoTuneTD-SBP experiments): for the case of `alpha = 0` that prioritizes computational performance, for `alpha = 0.5` that balances clustering accuracy and computational performance, and for `alpha = 1` that prioritizes clustering accuracy. More details on alpha can be found in the manuscript.

Each non-reference parameter configuration (Configurations 2-27) differs from the reference parameter configuration (Configuration 1) by exactly one parameter.

The complete definitions of all 27 parameter configurations considered in this work are available in [params.conf](TopDown-SBP/configs/params.conf).

The table below shows one CAIDA graph and one MIT Graph Challenge graph with the parameter configuration recommended by AutoTuneTD-SBP for each `alpha` value:

| Example Graph | Recommended Configuration<br>(for `alpha = 0`) | Recommended Configuration<br>(for `alpha = 0.5`) | Recommended Configuration<br>(for `alpha = 1`) |
| --- | --- | --- | --- |
| CAIDA<br>`20220114-`<br>`000000` | **Configuration 24**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=random<br>SPLIT=connectivity-snowball<br>MH_PERCENT=0.1<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=0<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> | **Configuration 22**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=random<br>SPLIT=connectivity-snowball<br>MH_PERCENT=1.0<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=1<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> | **Configuration 21**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=random<br>SPLIT=connectivity-snowball<br>MH_PERCENT=0.9<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=1<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> |
| MIT Graph Challenge<br>`static_lowOverlap_`<br>`highBlockSizeVar_`<br>`20000000_nodes` | **Configuration 12**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=random<br>SPLIT=single-snowball<br>MH_PERCENT=0.1<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=1<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> | **Configuration 9**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=degree-weighted<br>SPLIT=connectivity-snowball<br>MH_PERCENT=0.1<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=1<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> | **Configuration 1**<br><sub><code>SUBGRAPHS=2<br>BATCHES=2<br>CACHE_SIZE=20000<br>DEGREEPRODUCTSORT=1<br>SPLITINIT=random<br>SPLIT=connectivity-snowball<br>MH_PERCENT=0.1<br>ALGORITHM=hybrid_mcmc<br>OVERLAP=unk<br>NONPARAMETRIC=1<br>NODELTA=0<br>MIX=0<br>GREEDY=0<br>APPROXIMATE=0<br>ASYNC_ITERS=0</code></sub> |

Additional result files and provenance notes are available for [CAIDA](Results/CAIDA/README.md) and the [MIT Graph Challenge](Results/MGC/README.md).

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
