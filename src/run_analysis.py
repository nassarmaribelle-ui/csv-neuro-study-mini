from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample" / "sample_neuro_data.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"


def ensure_dirs():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df.columns = [c.strip().lower() for c in df.columns]
    numeric = ["age", "education_years", "mmse", "hippocampal_volume_mm3"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_stats(df):
    x = df["hippocampal_volume_mm3"]
    y = df["mmse"]

    r, p = stats.pearsonr(x, y)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    with open(OUT_TABLES / "stats_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Correlation hippocampal volume vs MMSE: r={r:.3f}, p={p:.4f}\n")

    X = sm.add_constant(df[["hippocampal_volume_mm3", "age", "education_years"]])
    model = sm.OLS(y, X).fit()

    with open(OUT_TABLES / "stats_summary.txt", "a", encoding="utf-8") as f:
        f.write("\nLinear regression: MMSE ~ hippocampal_volume_mm3 + age + education_years\n\n")
        f.write(model.summary().as_text())


def make_plots(df):
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(df["hippocampal_volume_mm3"], df["mmse"])
    plt.xlabel("Hippocampal volume (mm³)")
    plt.ylabel("MMSE")
    plt.title("MMSE vs Hippocampal volume")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "scatter_mmse_vs_volume.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    df = load_data(DATA_PATH)
    df = clean_data(df)
    run_stats(df)
    make_plots(df)
    print("Analysis complete. Check outputs/ for results.")


if __name__ == "__main__":
    main()