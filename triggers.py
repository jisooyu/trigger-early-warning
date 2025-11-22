# triggers.py
import pandas as pd
from preprocess import zscore


def evaluate_triggers(df: pd.DataFrame) -> dict:
    """
    Evaluate all triggers (Sahm, HY OAS, Term Spread, Temp YoY, Delinquencies).
    Uses only rows where required inputs are non-NaN.
    """

    required = ["UNRATE", "HY_OAS", "TEMP", "DELINQ", "term_spread"]
    df_trig = df.dropna(subset=required).copy()

    if len(df_trig) < 13:
        raise ValueError(
            "Not enough non-NaN history in trigger series (need ≥ 13 months)."
        )

    un = df_trig["UNRATE"]
    hy = df_trig["HY_OAS"]
    temp = df_trig["TEMP"]
    delinq = df_trig["DELINQ"]
    ts = df_trig["term_spread"]

    triggers = {}

    # ---- Term Spread (2Y - 3M) ----
    ts_latest = ts.iloc[-1]
    if ts_latest <= 0:
        ts_state = "Danger"
    elif ts_latest <= 0.5:
        ts_state = "Watch"
    else:
        ts_state = "Normal"

    triggers["Term Spread"] = {
        "state": ts_state,
        "detail": f"{ts_latest:.2f}%",
    }

    # ---- HY OAS ----
    hy_latest = hy.iloc[-1]
    if hy_latest >= 6.0:
        hy_state = "Danger"
    elif hy_latest >= 4.5:
        hy_state = "Watch"
    else:
        hy_state = "Normal"

    triggers["HY OAS"] = {
        "state": hy_state,
        "detail": f"{hy_latest:.2f}%",
    }

    # ---- Temp Help YoY ----
    temp_yoy = (temp.iloc[-1] / temp.iloc[-13] - 1) * 100
    if temp_yoy <= -6:
        temp_state = "Danger"
    elif temp_yoy <= -3:
        temp_state = "Watch"
    else:
        temp_state = "Normal"

    triggers["Temp Help YoY"] = {
        "state": temp_state,
        "detail": f"{temp_yoy:.1f}% YoY",
    }
    
    # ---- Unemployment (Sahm Rule) ----
    un_low = un.iloc[-12:].min()
    un_3mma = un.rolling(3).mean().iloc[-1]
    gap = un_3mma - un_low

    if gap >= 0.4:
        un_state = "Danger"
    elif gap >= 0.2:
        un_state = "Watch"
    else:
        un_state = "Normal"

    triggers["Unemployment (Sahm)"] = {
        "state": un_state,
        "detail": f"3m avg {un_3mma:.2f}% vs low {un_low:.2f}%",
    }


    # ---- Credit Card Delinquencies (z-score) ----
    d_z = zscore(delinq).iloc[-1]
    if d_z >= 1.0:
        d_state = "Danger"
    elif d_z >= 0.5:
        d_state = "Watch"
    else:
        d_state = "Normal"

    triggers["Card Delinq (z)"] = {
        "state": d_state,
        "detail": f"z={d_z:.2f}",
    }

    return triggers
