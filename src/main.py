import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from src.config import DATA_PATH, REPORTS_DIR, TARGET_COL
from src.data_pipeline import load_and_split_data
from src.models import get_models
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_numeric_correlation(X_train, y_train, path: Path):
    eda = X_train.select_dtypes(include=[np.number]).copy()
    eda[TARGET_COL] = y_train.values
    plt.figure(figsize=(10, 7))
    sns.heatmap(eda.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix (numeric features and CO2 emissions)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_test, predictions, model_name: str, path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.35, s=12)
    lo, hi = float(y_test.min()), float(y_test.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual CO2 (g/km)")
    plt.ylabel("Predicted CO2 (g/km)")
    plt.title(f"{model_name}: actual vs predicted (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(names, r2_scores, mae_scores, path: Path):
    x = np.arange(len(names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars1 = ax1.bar(x - width / 2, r2_scores, width, label="R²", color="steelblue")
    ax1.set_ylabel("R² (higher is better)")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mae_scores, width, label="MAE (g/km)", color="coral")
    ax2.set_ylabel("MAE (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_title("Mid-project model comparison (test set)")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    _ensure_reports_dir()
    print("--- Mid-Project Progress: Carbon Emission Predictor ---")

    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH)

    plot_numeric_correlation(
        X_train, y_train, REPORTS_DIR / "correlation_heatmap.png"
    )
    print(f"Saved: {REPORTS_DIR / 'correlation_heatmap.png'}")

    models = get_models()
    names, r2_list, mae_list = [], [], []

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        names.append(name.replace("_", " "))
        r2_list.append(r2)
        mae_list.append(mae)
        print(f"{name:<20} | R2: {r2:.4f} | MAE: {mae:.2f}")

        safe = name.lower()
        plot_actual_vs_predicted(
            y_test,
            predictions,
            name.replace("_", " "),
            REPORTS_DIR / f"{safe}_actual_vs_predicted.png",
        )
        print(f"Saved: {REPORTS_DIR / f'{safe}_actual_vs_predicted.png'}")

    plot_model_comparison(names, r2_list, mae_list, REPORTS_DIR / "model_comparison.png")
    print(f"Saved: {REPORTS_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    run()
