# Slide deck: Mid-project review — Vehicle CO₂ regression prototype

*Paste each block into one slide in PowerPoint or Google Slides, or use [Marp](https://marp.app/) by wrapping slides with `---`.*

---

## Slide 1 — Title

**Predicting certified CO₂ emissions for Canadian light-duty vehicles**  
Mid-project review — data pipeline, baselines, preliminary metrics  

Your name(s) · Course · 28 March 2026

---

## Slide 2 — Problem and motivation

- Regulatory **CO₂ label (g/km)** summarizes vehicle carbon intensity for consumers and fleets.
- We ask: can we **recover the label** from a small set of published attributes (engine, fuel, transmission, rated consumption)?
- **Why ML:** nonlinear interactions (e.g. fuel type × consumption) may improve fit; we still keep an interpretable linear baseline.

---

## Slide 3 — Data

- **Source:** Government of Canada open data — *Fuel consumption / CO₂ ratings* (national catalogue of models).
- **Mid-project file:** ~7.3k rows, many features (make, class, displacements, city/hwy/comb fuel use, etc.).
- **Target:** `CO2 Emissions(g/km)`.
- **Train/test:** 80% / 20%, `random_state=42` (frozen for reviewer reruns).

---

## Slide 4 — EDA (what we learned)

- Target distribution is **right-skewed** with a long tail of high-emitting vehicles (sports, large SUVs).
- **Combined L/100 km** is almost a sufficient statistic for CO₂ in this dataset: **strong linear correlation** with the label.
- Categoricals (**fuel type, transmission**) capture residual structure (e.g. hybrid labels, gear-count codes).

*Figures:* `reports/eda_co2_distribution.png`, `reports/correlation_heatmap.png` (or `reports/eda_correlation_numeric.png` from the notebook).

---

## Slide 5 — Data pipeline (coding deliverable)

`sklearn` **Pipeline** end-to-end:

| Step | Numeric features | Categorical features |
|------|------------------|----------------------|
| Transform | `StandardScaler` | `OneHotEncoder(handle_unknown="ignore")` |
| Model | Regressor (linear / k-NN / SVR) | |

- **Leakage protection:** preprocessing fit **only** on training fold inside the pipeline.
- Code: `src/data_pipeline.py`, `src/models.py`, entrypoint `src/main.py`.

---

## Slide 6 — Models compared

1. **Linear regression** — transparent baseline; coefficients align with physics intuition.  
2. **k-NN (k = 5)** — local smoother; needs scaling.  
3. **SVR (RBF, C = 100)** — global nonlinear baseline; needs scaling.

All three satisfy the “**compare multiple approaches**” requirement.

---

## Slide 7 — Preliminary results (quantitative)

Hold-out test set (this repo, mid-project run):

| Model | R² | MAE (g/km) |
|-------|----:|----------:|
| Linear regression | **0.991** | **3.13** |
| k-NN (k=5) | 0.985 | 3.94 |
| SVR (RBF) | **0.997** | **2.02** |

- **Takeaway:** strong predictability is expected when **rated fuel use** is an input; the interesting story is *which* algorithm trades off bias/variance and runtime for your final product choice.

*Figure:* `reports/model_comparison.png`.

---

## Slide 8 — Diagnostic plots

- **Actual vs predicted** for each model: points hug the diagonal → low systematic bias on the test split.  
- **Correlation heatmap** shows combined fuel consumption as the dominant linear correlate.

*Figures:*  
`reports/linear_regression_actual_vs_predicted.png`  
`reports/knn_regressor_actual_vs_predicted.png`  
`reports/svm_regressor_actual_vs_predicted.png`

---

## Slide 9 — Working prototype / reviewer service

**One-command regeneration (from repo root):**

```bash
pip install -r requirements.txt
python -m src.main
```

- Writes all PNGs under `reports/`.
- **Notebook path:** `notebooks/01_EDA_and_Baseline.ipynb` (interactive EDA + same metrics, extra EDA plots).

For peer review: share repo + exact commit hash + Python version.

---

## Slide 10 — Limitations and next steps

**Limitations**

- Labels = **certification cycle**, not on-road operating emissions.
- High R² partly reflects using **rated fuel consumption** as a predictor (near definitional).

**Next (toward final)**

- Nested CV / tuning for SVR and k-NN; optional tree ensembles (RandomForest/XGBoost).  
- Error slices by **vehicle class** and fuel type; residual plots.  
- Optional: drop comb L/100 km to stress-test “physics-free” prediction (harder task).

---

## Slide 11 — References (talk)

1. Alonso-Montero et al. (2022), *Applied Sciences* — fuel/CO₂ prediction for light-duty vehicles.  
2. Al-Obaidi et al. (2025), *PLOS ONE* — ML frameworks for fuel/CO₂ in ICE and hybrid fleets.  
3. Fontaras et al. (2017), *Prog. Energy Combust. Sci.* — lab vs real-world gaps.  
4. Pedregosa et al. (2011), *JMLR* — scikit-learn pipelines and best practice.

(Full citations in `docs/project_abstract.md`.)

---

## Slide 12 — Thank you / Q&A

Questions?
