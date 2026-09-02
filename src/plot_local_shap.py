# ============================================================
# EduPredict-XAI
# Local SHAP Visualization
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "xai"
)

INPUT_PATH = (
    RESULTS_PATH
    / "student_0_local_shap.csv"
)

# ------------------------------------------------------------
# Load SHAP result
# ------------------------------------------------------------

print("=" * 60)
print("LOCAL SHAP VISUALIZATION")
print("=" * 60)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"SHAP result not found:\n{INPUT_PATH}\n\n"
        "Run save_local_shap.py first."
    )

df = pd.read_csv(INPUT_PATH)

print("\nSHAP result loaded.")
print("Number of features:", len(df))

# ------------------------------------------------------------
# Sort by SHAP value
# ------------------------------------------------------------

plot_df = df.sort_values(
    by="SHAP_Value"
)

# ------------------------------------------------------------
# Create graph
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_df["Feature"],
    plot_df["SHAP_Value"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "SHAP Value"
)

plt.ylabel(
    "Student Feature"
)

plt.title(
    "Local SHAP Explanation — TabPFN"
)

plt.tight_layout()

# ------------------------------------------------------------
# Save graph
# ------------------------------------------------------------

output_path = (
    RESULTS_PATH
    / "student_0_local_shap.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSHAP graph saved successfully!"
)

print(output_path)

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)