# AutoTuneTD-SBP

AutoTuneTD-SBP predicts the utility of each Top-down SBP configuration from graph properties, the encoded configuration, and `alpha`. Configurations are ranked by predicted utility.

`alpha=0` optimizes computational performance, `alpha=0.5` balances performance and clustering accuracy, and `alpha=1` optimizes clustering accuracy.

## Measurements

The measurement CSV contains three runs of each of the 27 configurations for every graph. Required fields are:

- `Graph Name` and `Configuration ID` (`Dataset` is optional);
- the configuration fields varied in the paper;
- `Inverse H_norm`, `Directed Modularity`, `Directed Conductance`, and `TEPS`;
- the graph-property columns used for training.

Historical exports remain valid: `Result Batch`, `Parameter`, `Parameter ID`, and `SUBGRAPHPARTITION` are mapped to their current names. A configuration ID can also be derived from labels such as `Param24` or `theta_24`.

For each graph and configuration, the three runs are averaged before the accuracy, performance, and utility terms are computed. With the default settings, configurations outside `theta_1`--`theta_27` are ignored, and training fails if a graph is missing a run or one of those configurations. The checks can be disabled for incomplete exploratory data with `--expected-runs 0` or `--expected-configurations 0`.

## Train

```bash
python AutoTuneTD-SBP/train.py \
  --input path/to/synthetic_measurements.csv \
  --output-dir models/xgboost
```

Graphs are split 60/10/30 into training, validation, and test partitions. The model is fitted only on training graphs. Validation results are written during training; test graphs are recorded in the split manifest but are not evaluated.

The output directory contains:

```text
model.joblib
split_manifest.json
validation_metrics.json
validation_metrics_by_alpha.csv
validation_predictions.csv
validation_selections.csv
```

The default graph-property inputs are defined in `workflow.py`. To reproduce an experiment with a different feature set, provide a text file containing one CSV column name per line:

```bash
python AutoTuneTD-SBP/train.py \
  --input path/to/synthetic_measurements.csv \
  --graph-property-manifest path/to/graph_properties.txt \
  --output-dir models/xgboost
```

The selected columns are stored in `model.joblib` and reused by evaluation and recommendation.

## Test

`test.py` is the paper evaluation command, not a unit-test suite.

```bash
python AutoTuneTD-SBP/test.py \
  --model models/xgboost/model.joblib \
  --input path/to/synthetic_measurements.csv \
  --partition test \
  --output-dir results/synthetic_test
```

Use `--partition all` for CAIDA, MIT Graph Challenge, or other external graphs:

```bash
python AutoTuneTD-SBP/test.py \
  --model models/xgboost/model.joblib \
  --input path/to/real_graph_measurements.csv \
  --partition all \
  --output-dir results/real_graphs
```

Evaluation reports Top-1, Top-5, NDCG@5, Kendall's tau, and relative utility, accuracy, and performance errors. It also evaluates the random-rank, accuracy-prior, and performance-prior baselines.

The default test result uses `alpha=0.5`. Use `--alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1` for the complete tradeoff sweep.

```text
test_metrics.json
test_method_metrics.csv
test_method_metrics_overall.csv
test_predictions.csv
test_selections.csv
```

`test_method_metrics.csv` contains one row per method and alpha. The random-rank metrics are averaged over 100 random rankings by default; change this with `--random-repeats`.

## Recommend

Recommendation requires one graph-property row per new graph and the 27 configurations in `params.conf`. It does not execute Top-down SBP.

```bash
python AutoTuneTD-SBP/recommend.py \
  --model models/xgboost/model.joblib \
  --graph-features path/to/new_graph_properties.csv \
  --configurations TopDown-SBP/configs/params.conf \
  --alpha 0,0.5,1 \
  --output recommendations.csv
```

The output includes the full configuration, predicted utility, and predicted rank for every graph and alpha.
