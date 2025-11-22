# models.py
import numpy as np
import pandas as pd


def build_nyfed_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build NY Fed recession probability using 10Y–3M spread (monthly).
    Model: logit(P) = β0 + β1 * spread
    Returns probability in percent.
    """

    if "DGS10" not in df.columns or "DTB3" not in df.columns:
        raise ValueError("DGS10 or DTB3 missing — cannot compute NY Fed probability.")

    spread = df["DGS10"] - df["DTB3"]

    # NY Fed coefficients (10y–3m model)
    beta0 = -0.5333
    beta1 = -1.6278

    logit = beta0 + beta1 * spread
    prob = 1 / (1 + np.exp(-logit))

    out = pd.DataFrame({"probability": prob * 100}, index=df.index)
    return out
