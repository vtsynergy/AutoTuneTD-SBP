# AutoTuneTD-SBP

Artifact for **ML-Guided Parameter Configuration Selection for Top-Down Stochastic Block Partitioning**, accepted at IEEE HPEC 2026.

AutoTuneTD-SBP recommends a Top-down stochastic block partitioning (SBP) configuration for a graph. The tradeoff parameter `alpha` controls the objective: `0` prioritizes performance, `0.5` balances performance and accuracy, and `1` prioritizes accuracy.

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
  --configurations TopDown-SBP/configs/paper_parameter_configurations.conf \
  --alpha 0.5 \
  --output recommendations.csv
```

See [AutoTuneTD-SBP/README.md](AutoTuneTD-SBP/README.md) and [Data-Preprocessing/README.md](Data-Preprocessing/README.md) for input-column and preprocessing details.

## Results

The tables show the best observed `Param*` configuration among the checked-in candidates. For tied scores, the lowest numbered configuration is shown.

### CAIDA

| Graph name | Best configuration (`alpha = 0`) | Best configuration (`alpha = 0.5`) | Best configuration (`alpha = 1`) |
| --- | ---: | ---: | ---: |
| `20220114-000000` | Param98 | Param87 | Param86 |

Full CAIDA results and settings are in [Results/CAIDA](Results/CAIDA).

### MIT Graph Challenge (MGC)

| Graph name | Best configuration (`alpha = 0`) | Best configuration (`alpha = 0.5`) | Best configuration (`alpha = 1`) |
| --- | ---: | ---: | ---: |
| `static_highOverlap_highBlockSizeVar_1000000_nodes` | Param4 | Param103 | Param31 |
| `static_highOverlap_highBlockSizeVar_5000000_nodes` | Param4 | Param18 | Param57 |
| `static_highOverlap_highBlockSizeVar_20000000_nodes` | Param5 | Param5 | Param1 |
| `static_highOverlap_lowBlockSizeVar_1000000_nodes` | Param72 | Param74 | Param241 |
| `static_highOverlap_lowBlockSizeVar_5000000_nodes` | Param26 | Param19 | Param19 |
| `static_highOverlap_lowBlockSizeVar_20000000_nodes` | Param4 | Param4 | Param2 |
| `static_lowOverlap_highBlockSizeVar_1000000_nodes` | Param4 | Param26 | Param9 |
| `static_lowOverlap_highBlockSizeVar_5000000_nodes` | Param4 | Param4 | Param9 |
| `static_lowOverlap_highBlockSizeVar_20000000_nodes` | Param5 | Param2 | Param1 |
| `static_lowOverlap_lowBlockSizeVar_1000000_nodes` | Param4 | Param26 | Param181 |
| `static_lowOverlap_lowBlockSizeVar_5000000_nodes` | Param26 | Param26 | Param22 |
| `static_lowOverlap_lowBlockSizeVar_20000000_nodes` | Param5 | Param1 | Param1 |

Full MGC scores and settings are in [Results/MGC](Results/MGC).

## Tests

```bash
python -m unittest discover -s tests -v
```

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
