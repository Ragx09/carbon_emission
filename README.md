# CO₂ emissions prediction (Canadian light-duty vehicle ratings)

Predict **certified CO₂ emissions (g/km)** from vehicle attributes using reproducible **scikit-learn pipelines**. Compares **linear regression**, **k-nearest neighbours**, and **SVR** on open government fuel-consumption data.

**Repository:** [https://github.com/Ragx09/carbon_emission](https://github.com/Ragx09/carbon_emission)

## Requirements

- **Python** 3.10+ recommended  
- **Dependencies:** see `requirements.txt` (pandas, scikit-learn, matplotlib, seaborn, jupyter)

Install:

```bash
pip install -r requirements.txt
```

---

## Data

- **File:** `data/CO2 Emissions_Canada.csv`  
- **Target column:** `CO2 Emissions(g/km)`  
- **Source:** Government of Canada open data — *Fuel consumption ratings* / CO₂-related fields (cite the exact Open Government portal entry your course requires).

Paths are resolved from the repository root via `src/config.py` (`PROJECT_ROOT`, `DATA_PATH`).

---

## Machine-learning pipelines

Two levels of “pipeline” are implemented (both use `sklearn.pipeline.Pipeline`).

### 1. Preprocessing pipeline (`src/data_pipeline.py`)

`get_preprocessing_pipeline()` returns a **`ColumnTransformer`** that:

| Branch | Features | Transform |
|--------|-----------|-----------|
| Numeric | `Engine Size(L)`, `Cylinders`, `Fuel Consumption Comb (L/100 km)` | `StandardScaler` |
| Categorical | `Fuel Type`, `Transmission` | `OneHotEncoder(handle_unknown="ignore")`, **dense** output (stable with k-NN on this feature size) |

**`load_and_split_data(filepath)`** reads the CSV, drops the target from `X`, and returns an **80% / 20%** `train_test_split` with `random_state=42`.

### 2. Full model pipeline (`src/models.py`)

Each model is **`Pipeline([("preprocessor", preprocessor), ("regressor", ...)])`**:

| Key | Regressor |
|-----|-----------|
| `Linear_Regression` | `LinearRegression()` |
| `KNN_Regressor` | `KNeighborsRegressor(n_neighbors=5)` |
| `SVM_Regressor` | `SVR(kernel="rbf", C=100)` |

Preprocessing is **fit only on training data** inside the pipeline (no leakage from the test split).

**
```bash (run these to get reports)
cd carbon_emission-master
pip install -r requirements.txt
python -m src.main
```

**Outputs:**

- Console: **R²** and **MAE (g/km)** per model on the held-out test set.  
- **Figures** written to `reports/`:
  - `correlation_heatmap.png`
  - `model_comparison.png`
  - `linear_regression_actual_vs_predicted.png`
  - `knn_regressor_actual_vs_predicted.png`
  - `svm_regressor_actual_vs_predicted.png`

**Notebook (interactive EDA + same models):** open `notebooks/01_EDA_and_Baseline.ipynb`. If the kernel’s working directory is `notebooks/`, paths still resolve to `data/` and `reports/` at repo root.



## Metrics 

Results depend on the exact CSV and sklearn version; after `python -m src.main` use the printed **R²** and **MAE**. Interpretation note: including **combined fuel consumption (L/100 km)** as a feature makes high R² expected, because it is strongly tied to the certified CO₂ label.

