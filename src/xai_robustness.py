"""Upgrade module 4: XAI quality evaluation (the core research contribution).

 1. Global importance with exact TreeSHAP for CatBoost / XGBoost / LightGBM, exact linear SHAP for
    Ridge, and permutation importance, compared with the project's stored TabPFN SHAP ranking.
 2. Agreement between methods/models (Spearman & Kendall rank correlation).
 3. Stability: SHAP rankings over bootstrap re-trainings (global rank spread, local top-3 Jaccard).
 4. Faithfulness: deletion test - mask the top-k SHAP features and compare the prediction change
    against masking k random features.
"""
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import (FEATURES, FIG_DIR, RANDOM_STATE, RESULTS_DIR, TARGET, TEST_SIZE, UPGRADE_DIR,
                        encode_for_trees, ensure_dirs, load_data)

warnings.filterwarnings("ignore")
N_BOOT = 30


def _cb(seed=RANDOM_STATE):
    return CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03, random_seed=seed,
                             verbose=0, allow_writing_files=False)


def _mean_abs(sv):
    return pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)


def run():
    ensure_dirs()
    df = load_data()
    Xtr_r, Xte_r, ytr, yte = train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE)
    Xtr, Xte = encode_for_trees(Xtr_r, df), encode_for_trees(Xte_r, df)

    # ---- 1. global importance per model ------------------------------------------------------
    models = {"CatBoost": _cb(), "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                                         subsample=0.8, colsample_bytree=0.8,
                                                         random_state=RANDOM_STATE),
              "LightGBM": LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, num_leaves=8,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        random_state=RANDOM_STATE, verbose=-1)}
    importance, shap_cache, add_err = {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(Xte)
        shap_cache[name] = sv
        add_err[name] = float(np.max(np.abs(sv.sum(axis=1) + float(np.ravel(ex.expected_value)[0]) - m.predict(Xte))))
        importance[f"{name} (TreeSHAP)"] = _mean_abs(sv)

    # Ridge on standardised ordinal-coded features -> exact linear SHAP
    mu, sd = Xtr.mean(), Xtr.std()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit((Xtr - mu) / sd, ytr)
    lin_sv = shap.LinearExplainer(ridge, shap.maskers.Independent((Xtr - mu) / sd, max_samples=len(Xtr))
                                  ).shap_values((Xte - mu) / sd)
    importance["Ridge (Linear SHAP)"] = _mean_abs(lin_sv)

    cb_full = models["CatBoost"]
    pi = permutation_importance(cb_full, Xte, yte, n_repeats=30, random_state=RANDOM_STATE,
                                scoring="neg_mean_absolute_error")
    importance["CatBoost (Permutation)"] = pd.Series(pi.importances_mean, index=FEATURES)

    stored = pd.read_csv(RESULTS_DIR / "xai" / "tabpfn_global_shap_importance.csv").set_index("Feature")
    importance["TabPFN (stored SHAP)"] = stored["Mean_Absolute_SHAP"].reindex(FEATURES)

    imp = pd.DataFrame(importance)
    imp = imp.loc[imp["TabPFN (stored SHAP)"].sort_values(ascending=False).index]
    imp.round(4).to_csv(UPGRADE_DIR / "global_importance_all_methods.csv")
    rank = imp.rank(ascending=False).astype(int)
    rank.to_csv(UPGRADE_DIR / "global_importance_ranks.csv")
    print("Global importance (mean |attribution|):\n", imp.round(3).to_string())

    # ---- 2. agreement -------------------------------------------------------------------------
    cols, rows = list(imp.columns), []
    for a, b in itertools.combinations(cols, 2):
        rows.append({"Method_A": a, "Method_B": b,
                     "Spearman": stats.spearmanr(imp[a], imp[b])[0],
                     "Kendall_tau": stats.kendalltau(imp[a], imp[b])[0]})
    agree = pd.DataFrame(rows).round(3)
    agree.to_csv(UPGRADE_DIR / "explanation_agreement.csv", index=False)
    print("\nMean pairwise Spearman:", round(agree.Spearman.mean(), 3),
          "| min:", agree.Spearman.min(), "\n")

    # figure: normalised importance, grouped bars
    norm = imp / imp.sum()
    norm.plot(kind="barh", figsize=(9, 6), width=0.85)
    plt.gca().invert_yaxis(); plt.xlabel("Share of total mean |attribution|")
    plt.title("Global feature importance across models and methods"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "global_importance_comparison.png", dpi=200); plt.close()

    plt.figure(); shap.summary_plot(shap_cache["CatBoost"], Xte, show=False)
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_beeswarm_catboost.png", dpi=200, bbox_inches="tight"); plt.close()

    # ---- 3. stability across bootstrap retrainings --------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    glob, local_top3 = [], []
    base_local = shap_cache["CatBoost"]
    base_top3 = [set(np.argsort(-np.abs(r))[:3]) for r in base_local]
    jacc = []
    for b in range(N_BOOT):
        idx = rng.choice(len(Xtr), len(Xtr))
        m = _cb(seed=b).fit(Xtr.iloc[idx], ytr.iloc[idx])
        sv = shap.TreeExplainer(m).shap_values(Xte)
        glob.append(_mean_abs(sv))
        t3 = [set(np.argsort(-np.abs(r))[:3]) for r in sv]
        jacc.append(np.mean([len(a & c) / len(a | c) for a, c in zip(base_top3, t3)]))
    G = pd.DataFrame(glob)
    rk = G.rank(axis=1, ascending=False)
    stab = pd.DataFrame({"Mean_abs_SHAP": G.mean(), "Std": G.std(), "CV_percent": 100 * G.std() / G.mean(),
                         "Mean_rank": rk.mean(), "Rank_std": rk.std(),
                         "Pct_bootstraps_in_top3": (rk <= 3).mean() * 100}).sort_values("Mean_rank")
    stab.round(3).to_csv(UPGRADE_DIR / "shap_bootstrap_stability.csv")
    pd.Series({"bootstraps": N_BOOT, "mean_local_top3_jaccard": np.mean(jacc),
               "std_local_top3_jaccard": np.std(jacc)}).round(4).to_csv(
        UPGRADE_DIR / "shap_local_top3_stability.csv", header=["value"])
    print("Bootstrap stability of global SHAP ranking:\n", stab.round(2).to_string())
    print("\nLocal top-3 Jaccard vs full-data model: %.3f +/- %.3f" % (np.mean(jacc), np.std(jacc)))

    # ---- 4. faithfulness: deletion test -------------------------------------------------------
    base_pred = cb_full.predict(Xte)
    mean_vec = Xtr.mean()
    rows = []
    sv = shap_cache["CatBoost"]
    for k in range(1, 6):
        shap_mask, rand_mask = Xte.copy(), Xte.copy()
        for i in range(len(Xte)):
            top = np.argsort(-np.abs(sv[i]))[:k]
            shap_mask.iloc[i, top] = mean_vec.values[top]
            rnd = rng.choice(len(FEATURES), k, replace=False)
            rand_mask.iloc[i, rnd] = mean_vec.values[rnd]
        rows.append({"k_features_masked": k,
                     "Mean_abs_change_top_SHAP": np.mean(np.abs(cb_full.predict(shap_mask) - base_pred)),
                     "Mean_abs_change_random": np.mean(np.abs(cb_full.predict(rand_mask) - base_pred))})
    faith = pd.DataFrame(rows)
    faith["Ratio"] = faith.Mean_abs_change_top_SHAP / faith.Mean_abs_change_random
    faith.round(3).to_csv(UPGRADE_DIR / "shap_faithfulness_deletion.csv", index=False)
    print("\nFaithfulness (deletion test):\n", faith.round(3).to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.plot(faith.k_features_masked, faith.Mean_abs_change_top_SHAP, "o-", label="Mask top-k SHAP features")
    plt.plot(faith.k_features_masked, faith.Mean_abs_change_random, "s--", label="Mask k random features")
    plt.xlabel("k features masked"); plt.ylabel("Mean |change in prediction|")
    plt.title("SHAP faithfulness (deletion test)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_faithfulness.png", dpi=200); plt.close()

    # additivity sanity check (local accuracy property of SHAP)
    pd.Series(add_err, name="max_abs_additivity_error").to_csv(UPGRADE_DIR / "shap_additivity_check.csv")
    print("\nSHAP additivity max error per model:", {k: f"{v:.1e}" for k, v in add_err.items()})


if __name__ == "__main__":
    run()
