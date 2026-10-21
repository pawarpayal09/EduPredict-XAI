"""Robust model loading for the Streamlit app.

Order of preference
  1. models/tabpfn_fitted_model.pkl      (pickle created by notebook 06; git-ignored because it is large)
  2. models/tabpfn_regressor.tabpfn_fit  (TabPFN's own save format, committed in the repository)
  3. CatBoost fallback trained on the processed data (so the app never crashes on a fresh clone)
The app shows which model is active, so results are never silently attributed to the wrong model.
"""
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import FEATURES, MODELS_DIR, RANDOM_STATE, TARGET, encode_for_trees, load_data


class EncodedCatBoost:
    """CatBoost wrapper that accepts the same raw (string-categorical) DataFrame as TabPFN."""

    def __init__(self):
        self.reference = load_data()
        X = encode_for_trees(self.reference[FEATURES], self.reference)
        self.model = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03,
                                       random_seed=RANDOM_STATE, verbose=0,
                                       allow_writing_files=False).fit(X, self.reference[TARGET])

    def predict(self, X: pd.DataFrame):
        return self.model.predict(encode_for_trees(X[FEATURES], self.reference))


def load_primary_model():
    """Return (model, display_name, conformal_key, note)."""
    pkl = MODELS_DIR / "tabpfn_fitted_model.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f), "TabPFN", "TabPFN", ""
        except Exception as e:  # noqa: BLE001
            note = f"TabPFN pickle could not be loaded ({type(e).__name__}); "
    else:
        note = "models/tabpfn_fitted_model.pkl not found; "

    native = MODELS_DIR / "tabpfn_regressor.tabpfn_fit"
    if native.exists():
        try:
            from tabpfn.model_loading import load_fitted_tabpfn_model
            return load_fitted_tabpfn_model(native, device="cpu"), "TabPFN", "TabPFN", ""
        except Exception as e:  # noqa: BLE001
            note += f"native TabPFN file could not be loaded ({type(e).__name__}); "

    return (EncodedCatBoost(), "CatBoost (fallback)", "CatBoost (fallback)",
            note + "using the CatBoost fallback model. Re-run notebook 06 to regenerate the TabPFN model.")
