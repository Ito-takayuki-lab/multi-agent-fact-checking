__author__ = "Dong Yihan"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_bootstrap_metrics(
    csv_path: str,
    n_bootstrap: int = 1000,
    random_state: int = 0,
    map_fever_labels: bool = True,
) -> pd.DataFrame:
    """
    Compute point estimates + 95% bootstrap CIs for:
      - accuracy
      - macro precision
      - macro recall
      - macro F1

    Assumes a CSV with columns:
      - 'label'   : gold labels
      - 'results' : predicted labels

    Normalises labels by stripping whitespace and uppercasing.
    If map_fever_labels=True, gold labels "SUPPORTS"/"REFUTES"
    are mapped to "TRUE"/"FALSE", and any other gold labels
    (e.g. "NOT ENOUGH INFO") are dropped.
    """

    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "results" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'results' columns.")

    # Normalise: strip spaces, uppercase
    y_true_raw = df["label"].astype(str).str.strip().str.upper().to_numpy()
    y_pred_raw = df["results"].astype(str).str.strip().str.upper().to_numpy()

    if map_fever_labels:
        # Map FEVER-style gold labels to TRUE/FALSE
        gold_map = {"SUPPORTS": "TRUE", "REFUTES": "FALSE"}

        # Keep only mapped labels (drops NOT ENOUGH INFO etc.)
        mask = np.isin(y_true_raw, list(gold_map.keys()))
        if not mask.any():
            raise ValueError(
                "No gold labels matched SUPPORTS/REFUTES in this file. "
                "Check the 'label' column or set map_fever_labels=False."
            )

        y_true = np.array([gold_map[l] for l in y_true_raw[mask]])
        y_pred = y_pred_raw[mask]
    else:
        # Use labels as they are (e.g., TRUE / PARTLY TRUE / FALSE)
        y_true = y_true_raw
        y_pred = y_pred_raw

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    # ---- helper to compute metrics on one sample ----
    def _metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", zero_division=0
        )
        return acc, p, r, f1

    # Point estimates on full test set
    acc_point, p_point, r_point, f1_point = _metrics(y_true, y_pred)

    # ---- bootstrap ----
    acc_samples, p_samples, r_samples, f1_samples = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        acc_b, p_b, r_b, f1_b = _metrics(y_true[idx], y_pred[idx])
        acc_samples.append(acc_b)
        p_samples.append(p_b)
        r_samples.append(r_b)
        f1_samples.append(f1_b)

    def ci(values, alpha=0.05):
        low = np.percentile(values, 100 * (alpha / 2))
        high = np.percentile(values, 100 * (1 - alpha / 2))
        return low, high

    acc_low, acc_high = ci(acc_samples)
    p_low, p_high = ci(p_samples)
    r_low, r_high = ci(r_samples)
    f1_low, f1_high = ci(f1_samples)

    data = {
        "metric": ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
        "point": [acc_point, p_point, r_point, f1_point],
        "ci_low": [acc_low, p_low, r_low, f1_low],
        "ci_high": [acc_high, p_high, r_high, f1_high],
    }
    results_df = pd.DataFrame(data)
    return results_df


# Example usage:
results_dir = "./experiment_results/SciFact/only_llm.csv"
df_metrics = compute_bootstrap_metrics(results_dir, map_fever_labels=True)
print(df_metrics)
