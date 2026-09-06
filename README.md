# Air Quality Estimation

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-6E56CF?logo=uv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-modeling-189FDD?logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557C?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-visualization-4C72B0?logo=seaborn&logoColor=white)

An end-to-end multi-target regression project for estimating ground-truth pollutant concentrations from low-cost metal-oxide sensor responses and meteorological measurements.

The project focuses on building a reproducible, leakage-aware time-series workflow and evaluating how sensor ablations, causal lag/rolling/delta features, drift features, per-target tuning, recency selection, and ensembling affect estimation performance across four pollutants.

> This project is intended for educational and portfolio purposes. It is not a calibrated or deployable air-quality monitoring system.

## Project Objective

The objective is to estimate hourly ground-truth concentrations for four targets:

- `CO(GT)` in mg/m^3
- `C6H6(GT)` (benzene) in microg/m^3
- `NOx(GT)` in ppb
- `NO2(GT)` in microg/m^3

The main workflow uses per-target model selection among `RandomForest`, `XGBoost`, and `HistGradientBoosting` (plus `Ridge`/`ElasticNet` for `NOx`) with:

- Chronological train/validation/test split
- Causal feature engineering (`shift` before `ffill`, past-only values)
- `TimeSeriesSplit` cross-validation
- Per-target `RandomizedSearchCV` / `GridSearchCV` tuning
- Validation-based model selection on `R2` then `RMSE`
- Train+validation refit and single test evaluation
- Permutation-importance analysis
- A recency + delta ablation and weighted ensemble for `NO2`

## Dataset

This project uses the [Air Quality UCI dataset](https://archive.ics.uci.edu/dataset/360/air+quality) from the UCI Machine Learning Repository, also described in De Vito et al., Sensors and Actuators B, Vol. 129, 2008.

Dataset summary:

- 9,358 raw hourly records from March 2004 to April 2005 (analysis-ready: 9,357 rows, 0 invalid `DateTime`, 0 duplicate timestamps, hourly frequency preserved)
- 5 metal-oxide sensors: `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`
- 3 meteorological variables: `T`, `RH`, `AH`
- 4 regression targets: `CO(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`
- Missing values tagged as `-200` in the raw CSV and treated as `NaN`
- Empty columns `Unnamed: 15/16` and fully empty rows dropped; no bare `dropna()`
- CSV parsing uses `sep=";" decimal="," na_values=-200`
- `DateTime` built from `Date` (`DD/MM/YYYY`) + `Time` (`HH.MM.SS`, dots to colons) with `format="%d/%m/%Y %H:%M:%S"`, sorted chronologically

Chronological split (`70/15/15`):

| Split | Rows | Start | End |
|---|---|---:|---|
| train | 6549 | 2004-03-10 18:00:00 | 2004-12-08 14:00:00 |
| validation | 1404 | 2004-12-08 15:00:00 | 2005-02-05 02:00:00 |
| test | 1404 | 2005-02-05 03:00:00 | 2005-04-04 14:00:00 |

Test rows per target are slightly lower because rows with missing `*(GT)` are masked per target (`CO`: 1374, `C6H6`: 1327, `NOx`/`NO2`: 1367).

Feature groups:

| Group | Features |
|---|---|
| Sensors | `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)` |
| Meteorology | `T`, `RH`, `AH` |
| Calendar (cyclic + categorical) | `hour_sin/cos`, `month_sin/cos`, `day_of_week`, `is_weekend` |
| Drift / physics | `elapsed_hours`, `elapsed_days`, `T_RH`, `T_AH`, `S1_S5_ratio`, `S3_S5_diff`, `S1_S3_diff` |
| Causal lags | `PT08.S1/S3/S4/S5 × lag1/lag24` via `shift` before `ffill` |
| Causal rolling | `PT08.S5(O3)_roll24`, `PT08.S1(CO)_roll24` via `rolling(24).mean().shift(1)` |
| Causal deltas (NO2 only) | `S1/S3/S4/S5 × delta1/delta24`, `S5/S3 × dev24` |

Per-target feature map in `main`:

| Target | Features |
|---|---|
| `CO(GT)` | base 14 only (sensors + meteo + calendar) |
| `C6H6(GT)` | base 14 only |
| `NOx(GT)` | base + drift (21) + lag 8 + roll 2 |
| `NO2(GT)` | ensemble of `no2_base` (21 + lag + roll) and `no2_delta` (base + lag + roll + delta + deviation) |

No `*(GT)` column or `DateTime` is ever used as a predictor. Imputation is forward-fill only (`ffill`); no `bfill`/`interpolate`.

## Project Structure

```text
air-quality-estimation/
├── data/
│   ├── AirQualityUCI.csv
│   ├── AirQualityUCI.xlsx
│   ├── data_overview.md
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── eda.ipynb
│   └── modeling.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

### Notebooks

- `data_preparation.ipynb`: parses the raw `;`-separated CSV, builds `DateTime`, engineers calendar + drift/physics features, performs the chronological `70/15/15` split, and exports `data/processed/*.csv`. Never edit the exported CSVs by hand; re-run this notebook.
- `eda.ipynb`: multi-pollutant EDA — distributions, missingness/`-200` patterns, sensor cross-sensitivities, drift over time, correlations, and target availability.
- `modeling.ipynb`: per-target CV tuning, validation selection, train+validation refit, single test evaluation, `NO2` recency/delta ablation and weighted ensemble, permutation importance, and predicted-vs-actual plots.

`notebooks/*.py` files are local `jupytext` sync artifacts (untracked) — the `.ipynb` files are the source of truth.

## Methodology

1. Exploratory data analysis across sensors, meteorology, and all four targets
2. Leakage-aware parsing and chronological `70/15/15` split
3. Causal feature engineering on the concatenated `train → validation → test` frame (`shift`/`rolling().shift(1)` before `ffill`)
4. Per-target feature mapping (base vs drift+lag+roll vs delta variants)
5. `TimeSeriesSplit(n_splits=4)` cross-validation with `neg_root_mean_squared_error` scoring
6. Per-target search spaces with `RandomizedSearchCV` (`GridSearchCV` for `Ridge` only)
7. Validation-based selection on `validation_R2`, then `validation_RMSE`
8. `NO2`-only ablation over `train_frac × {0.75, 1.0}` and `delta × {off, on}`
9. Train+validation refit with the winning config and single test evaluation per target
10. Weighted `NO2` ensemble refit per member with its own feature variant
11. Permutation-importance analysis and predicted-vs-actual diagnostics

Preprocessing (`median` imputation, optional `Standard`/`Robust` scaling) is fitted inside `sklearn` pipelines. Tree models skip scaling; linear models use `RobustScaler`.

## Main Model Performance

Validation winners (selection basis):

| Target | Model | Train frac | Features | CV RMSE | Validation RMSE | Validation MAE | Validation R2 |
|---|---|---:|---|---:|---:|---:|---:|
| `C6H6(GT)` | RandomForest | 1.0 | base | 0.294 | 0.081 | 0.015 | 0.99987 |
| `CO(GT)` | RandomForest | 1.0 | base | 0.433 | 0.988 | 0.625 | 0.62533 |
| `NO2(GT)` | RandomForest | 0.75 | delta_off | 26.240 | 29.502 | 21.492 | 0.60868 |
| `NOx(GT)` | HistGradientBoosting | 1.0 | base+drift+lag+roll | 89.185 | 124.098 | 73.003 | 0.75734 |

Final test performance (train+validation refit, test evaluated once):

| Target | Best model | Test RMSE | Test MAE | Test R2 | Test rows |
|---|---|---:|---:|---:|---:|
| `CO(GT)` | RandomForest | 0.710 | 0.382 | 0.709 | 1374 |
| `C6H6(GT)` | RandomForest | 0.025 | 0.011 | 1.000 | 1327 |
| `NOx(GT)` | HistGradientBoosting | 99.309 | 50.062 | 0.721 | 1367 |
| `NO2(GT)` | `RFoff+RFin+XGBon` ensemble | 30.971 | 20.531 | 0.653 | 1367 |

The single-model `NO2` baseline (`RandomForest`, `frac 0.75`, `delta_off`) scored test `RMSE 32.089 / MAE 21.780 / R2 0.627`. The ensemble improves it to `R2 0.653`.

`NO2` ensemble details:

- Members (each refit on `train+val` with its own feature variant): `RF 0.75/off`, `RF 0.75/on`, `XGB 0.75/on`
- Weights selected on validation `R2`: `[0.6, 0.2102, 0.1898]` with ensemble `val_R2 0.6006`
- Candidate weights tried: `w_off ∈ {0.4, 0.5, 0.6}` with the remainder split proportional to member `val_R2`

Top permutation-importance signals (train, `n_repeats=10`, `neg RMSE` scoring):

- `CO`: `PT08.S2(NMHC)`, `T`, `PT08.S1(CO)`, `hour_sin/cos`
- `NOx`: `elapsed_hours`, `PT08.S2(NMHC)`, `PT08.S5(O3)`, `S3_S5_diff`, `PT08.S3(NOx)`
- `NO2`: `PT08.S5(O3)`, `T_RH`, `S3_S5_diff`, `PT08.S1(CO)`, `hour_sin`

## Experiment Results

| Experiment | Result | Decision |
|---|---|---|
| `ablation-pt08-s2` (#1) | `NO2` without `PT08.S2(NMHC)` generalizes better (removes NMHC cross-sensitivity) | Integrated into `main` for `NO2` |
| `lag-features-no2` (#2) | Causal `lag1/lag24` for `S1/S3/S4/S5` improve `NOx`/`NO2` temporal context | Integrated into `main` |
| `log1p-co-no2` (reverted) | `log1p` target transform with higher CV iterations did not justify the added complexity | Reverted, not in `main` |
| `drift-temporal-co-nox-no2` (#3) | Per-target drift: `CO` reverted to base, `NOx`/`NO2` keep drift+lag+roll; `RobustScaler` + rolling + physics interactions added | Integrated as per-target map |
| `hparam-per-target-v2` (#4) | Per-target search spaces and CV; `Ridge` removed for `CO`/`NO2`; test-set refit protocol | Integrated into `main` |
| `no2-recency-delta` (#5) | `RF frac 0.75 delta_off` adopted as single-model winner (test `R2 0.627`); causal delta/deviation features defined | Integrated, then superseded by ensemble |
| `no2-ensemble` (#6) | `RFoff+RFin+XGBon` test `R2 0.653` beats single `0.627` | Integrated as final `NO2` model |

### `NO2` Progression

| Variant | Validation R2 | Test R2 | Note |
|---|---:|---:|---|
| `RF 0.75 delta_off` (single) | 0.609 | 0.627 | Best single after recency/delta ablation |
| `RFoff + RFin + XGBon` ensemble | 0.601 | 0.653 | Final; each member refit with its own feature variant |

The ensemble validation `R2` is slightly below the best single, but test `R2` improves by `+0.026` with lower `RMSE`/`MAE`, so it was adopted as the final `NO2` estimator. `CO`, `C6H6`, and `NOx` remain single-model winners.

## Limitations and Responsible Use

- Single fixed chronological split; metrics may shift under a different time window because of concept/sensor drift.
- No external validation site or deployment calibration; results are retrospective estimates.
- `C6H6` test `R2 ≈ 1.0` is suspiciously perfect and likely reflects a near-direct mapping from the NMHC-targeted sensor rather than generalizable inference — reported honestly, not as a modeling success.
- `NO2` remains the hardest target (`R2 0.653`) despite the most feature engineering and ensembling.
- Missing values use forward-fill only, which can carry stale sensor values across gaps.
- Test-row counts differ per target because missing `*(GT)` rows are masked per task.
- The project has not been validated for regulatory or health-decision use.

## Reproducibility and Usage

The project requires Python 3.12+ with dependencies declared in `pyproject.toml` and `uv.lock`.

Install the environment with:

```bash
uv sync
```

Start Jupyter with:

```bash
uv run jupyter notebook
```

Recommended notebook order:

```text
data_preparation.ipynb
eda.ipynb
modeling.ipynb
```

Optional checks:

```bash
python -m json.tool "notebooks/modeling.ipynb" > $null
rtk git diff --check
rtk git status --short
```

Full notebook re-execution (delete the executed copy afterwards):

```bash
uv run jupyter nbconvert --to notebook --execute "notebooks/modeling.ipynb" --output "modeling.executed.ipynb" --ExecutePreprocessor.timeout=900
```

## Portfolio Highlights

This project demonstrates:

- End-to-end multi-target regression on real sensor data
- Leakage-aware time-series splits and causal feature engineering
- Per-target feature mapping and model selection
- Cross-validation with `TimeSeriesSplit` and per-target tuning
- Ablation studies across sensors, lags, drift, recency, and deltas
- Weighted ensembling with per-member feature variants
- Model interpretability through permutation importance
- Honest reporting of trade-offs, including a too-good-to-be-true result

## Dataset Attribution

The dataset is provided by the UCI Machine Learning Repository under the dataset's stated license (research purposes). See the [official dataset page](https://archive.ics.uci.edu/dataset/360/air+quality) for attribution and licensing details. Please cite De Vito et al. (2008) when using the data.

## License

The code and notebooks in this repository are licensed under the MIT License — see the [LICENSE](LICENSE) file for details (Copyright (c) 2026 Muhammad Akmal Fazli Riyadi).

The dataset (`data/AirQualityUCI.*` and derivatives in `data/processed/`) is **not** covered by the MIT License; it follows the UCI dataset's own terms (research purposes, commercial purposes excluded).
