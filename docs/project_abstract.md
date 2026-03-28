# Project Abstract: Machine-Learning Estimation of Light-Duty Vehicle CO₂ Emissions

## Background and motivation

Governments publish standardized fuel-economy and CO₂ emission labels so consumers and fleet managers can compare vehicles on a common scale. These ratings are tied closely to lab-measured or modeled energy use; in practice, many downstream applications—what-if comparisons, sketch design tools, or educational dashboards—benefit from fast *predictive* models that map easy-to-observe attributes (engine size, cylinders, transmission, fuel type, rated consumption) to reported CO₂ intensity. Recent work shows that classical and nonlinear machine-learning regressors can capture much of this mapping when trained on large national vehicle registries, often with strong agreement between predicted and labeled emissions, while surfacing trade-offs between interpretability, latency, and accuracy (Alonso-Montero et al., 2022; Al-Obaidi et al., 2025; Jouhari et al., 2024).

This project develops an end-to-end, reproducible workflow on the Canadian light-duty fuel-consumption ratings table (Government of Canada open data: *CO₂ Emissions / Fuel Consumption Ratings* for model years represented in the released file). The scientific question is pragmatic: **how accurately can we recover laboratory-reported CO₂ (g/km) from a compact feature set, and which off-the-shelf learners are appropriate baselines before more complex feature engineering or model search?**

## Objectives

1. **Data plumbing:** Ingest the national ratings file, document schema assumptions, and enforce a train/test split with fixed random seed for comparability across reviewers.  
2. **Preprocessing pipeline:** Encode categorical fields (fuel type, transmission), scale continuous inputs needed for distance- and margin-based methods, and package steps in a single `sklearn` `Pipeline` to prevent leakage between training and test folds.  
3. **Modeling baselines:** Compare at least three approaches—a linear model (transparent baseline), *k*-nearest neighbours (local, nonparametric), and a kernel support-vector regressor (nonlinear, global smooth)—on standard regression scores.  
4. **Communication deliverables:** Produce interpretable diagnostics (correlation structure, distribution of the target, actual-versus-predicted plots, and a compact model comparison chart) suitable for policy-literate and technical audiences.  
5. **Roadmap to final submission:** Extend evaluation with cross-validation and error analysis by segment (vehicle class, powertrain), explore feature-importance or regularized alternatives if linear models are retained for explainability, and consolidate a small command-line or notebook “reviewer service” entry point that regenerates all figures and metrics.

## Methods (mid-project status)

The working prototype selects a subset of columns aligned with course feature-engineering goals: engine displacement, cylinder count, combined fuel consumption (L/100 km), fuel-type label, and transmission label. Categorical variables are one-hot encoded; numeric predictors are standardized before *k*-NN and SVR. All steps are composed with `ColumnTransformer` and `Pipeline`, matching recommended practice for reproducible learning systems (Pedregosa et al., 2011). Hold-out evaluation uses 80% training and 20% testing (`random_state=42`).

## Preliminary results

On the current split, **all three baselines achieve high coefficient-of-determination values**, with combined fuel consumption dominating the linear correlation structure (as expected from the definition of rated CO₂). In the implementation bundled with this repository, the **RBF SVR** configuration tested at mid-project achieves the **lowest mean absolute error** on the hold-out fold, followed closely by ordinary least-squares linear regression; *k*-NN registers a modestly higher MAE. Point clouds on actual-versus-predicted plots cluster tightly around the diagonal, indicating that the selected attributes explain most variation in the label for static ratings data. These numbers characterize *interpolation* within the historical catalogue rather than real-world on-road uncertainty, which is an important limitation to state explicitly in the final report.

## Work plan through final evaluation

| Phase | Activities |
| --- | --- |
| **Analysis** | Stratify errors by vehicle class; inspect residuals versus engine technology; document outliers. |
| **Modeling** | Tune SVR and *k*-NN via cross-validation; consider tree ensembles if time permits; keep a linear report-facing model for transparency. |
| **Validation** | Report confidence intervals via bootstrapping or nested CV; stress-test `handle_unknown` categories. |
| **Product** | Freeze `requirements.txt`; provide `python -m src.main` for one-click plots; polish notebooks for teaching use. |

## Limitations

Labels reflect regulatory test cycles and certification metadata, not on-road telematics. Results should not be read as real-world emissions without further calibration (see discussion in Fontaras et al., 2017, on lab versus real-world gaps). The project also does not yet address fairness across market segments or temporal drift when new powertrains appear.

## References

1. Alonso-Montero, J., Fernández-Ahúja, A., Lara-Fanego, V., & Díaz-Plácido, J. (2022). Analysis and prediction model of fuel consumption and carbon dioxide emissions of light-duty vehicles. *Applied Sciences*, *12*(2), 803. https://doi.org/10.3390/app12020803  

2. Al-Obaidi, A., Abduljabbar, Z. N., Al-Qrimli, H. A., & Abdulkareem, A. (2025). A machine learning framework for predicting fuel consumption and CO₂ emissions in hybrid and combustion vehicles: Comparative analysis and performance evaluation. *PLOS ONE*, *20*(1), e0342418. https://doi.org/10.1371/journal.pone.0342418  

3. Fontaras, G., Zacharof, N.-G., & Ciuffo, B. (2017). Fuel consumption and CO₂ emissions from passenger cars in Europe—Lab versus real-world measurements. *Progress in Energy and Combustion Science*, *60*, 97–131. https://doi.org/10.1016/j.pecs.2016.12.004  

4. Jouhari, M., El-Mogy, F., Mudassar, M., & Arhab, S. (2024). Effective modeling of CO₂ emissions for light-duty vehicles: Linear and nonlinear models with feature selection. *Energies*, *17*(7), 1655. https://doi.org/10.3390/en17071655  

5. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html  

**Data source:** Government of Canada. (*n.d.*). Fuel consumption ratings—CO₂ emissions (open data product derived from Natural Resources Canada’s vehicle fuel consumption testing program). Retrieved via the Open Government Portal (search: “Fuel consumption ratings”).

---

*Length: ~1.5 pages in typical double-spaced 11–12pt prose; suitable for minor edits to match your course template.*
