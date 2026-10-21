"""Reproduce every upgraded experiment in order.  Usage:  python run_all_upgrades.py [--with-tabpfn]
Outputs go to results/upgrade/ (tables) and results/upgrade/figures/ (paper-ready figures)."""
import sys

from src import (counterfactual_whatif, cv_benchmark, fairness_audit, holdout_comparison,
                 uncertainty_and_risk, xai_robustness)

if __name__ == "__main__":
    cv_benchmark.run(with_tabpfn="--with-tabpfn" in sys.argv)
    holdout_comparison.run()
    uncertainty_and_risk.run()
    xai_robustness.run()
    fairness_audit.run()
    counterfactual_whatif.run()
    print("\nAll upgrade experiments finished. See results/upgrade/")
