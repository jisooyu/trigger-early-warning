# trigger_dashboard_fred.py
# --------------------
# Contains: FRED loading, triggers, charts, helper functions

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import plotly.graph_objects as go

import requests
from requests.adapters import HTTPAdapter, Retry
import time

session = requests.Session()
retries = Retry(total=5, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))

# ============================================================
# FRED FETCHER
# ============================================================

def fred(series, start="1990-01-01"):
    """Robust FRED fetcher with:
    - automatic retry
    - HTML error detection
    - clean warning messages
    """
    url = f"https://fred.stlouisfed.org/series/{series}"

    for attempt in range(1, 4):
        try:
            df = web.DataReader(series, "fred", start, session=session)
            # print(f"[DEBUG] {series} head:\n{df.head()}")

            if df is None or df.empty:
                raise ValueError("Empty response")

            df.index = pd.to_datetime(df.index)
            df.columns = [series]
            return df.dropna()

        except Exception as e:
            try:
                response = session.get(url, timeout=5)
                content_type = response.headers.get("Content-Type", "")

                if "text/html" in content_type.lower():
                    print(
                        f"[FRED WARNING] {series}: HTML error page received "
                        "(likely 404 or rate-limit)."
                    )
                else:
                    print(
                        f"[FRED WARNING] {series}: fetch failed ({str(e)})"
                    )

            except Exception:
                print(f"[FRED WARNING] {series}: network error.")

            if attempt < 3:
                wait = 2 ** (attempt - 1)
                print(f"[Retrying {series} in {wait}s]...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"[FRED ERROR] {series} failed after 3 attempts."
                )

# ============================================================
# LOAD ALL INDICATORS (no LEI required)
# ============================================================
def load_all_data(start="2000-01-01"):

    series_map = {
        "DGS2":        "DGS2",
        "DTB3":        "DTB3",
        "BAMLH0A0HYM2": "HY_OAS",
        "TEMPHELPS":   "TEMP",
        "DRCCLACBS":   "DELINQ",
        "UNRATE":      "UNRATE",
        "USREC":       "USREC",
        "DGS10":       "DGS10",  
    }

    monthly_dfs = []

    for fred_series, col_name in series_map.items():

        # Fetch raw data
        s = fred(fred_series, start=start)

        # Monthly resample rules
        if col_name == "DELINQ":
            s_me = s.resample("ME").last().ffill()
        elif col_name == "USREC":
            s_me = s.resample("ME").last().ffill()
        else:
            s_me = s.resample("ME").last()

        # Correct indentation ↓↓↓
        s_me.columns = [col_name]
        monthly_dfs.append(s_me)

    # OUTER join to keep full date range
    df = pd.concat(monthly_dfs, axis=1, join="outer").sort_index()

    # Term spread
    df["term_spread"] = df["DGS2"] - df["DTB3"]

    return df

# ============================================================
# Z-score utility
# ============================================================

def zscore(series, window=60):
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()

# ============================================================
# Last Update Help Function
# ============================================================
def add_last_update_annotation(fig, df, series_name):
    """
    Adds a 'Last updated' annotation under each chart.
    series_name = column or series used for last date detection.
    """
    # choose the last non-NaN date for this series
    last_valid = df[series_name].dropna().index[-1]
    last_str = last_valid.strftime("%Y-%m-%d")

    fig.add_annotation(
        text=f"Last updated: {last_str}",
        xref="paper", yref="paper",
        x=0, y=-0.18,                # position below plot
        showarrow=False,
        font=dict(size=10, color="#cccccc"),
    )
    return fig

# ============================================================
# Add Recession Shading
# ============================================================
def add_recession_shading(fig, df):
    """Add NBER recession shading using USREC == 1 regions."""

    rec = df["USREC"]

    in_rec = False
    start_date = None

    for date, val in rec.items():

        if not in_rec and val == 1:
            in_rec = True
            start_date = date

        elif in_rec and val == 0:
            in_rec = False
            end_date = date
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor="gray",
                opacity=0.25,
                line_width=0,
                layer="below",
            )

    # Handle ongoing recession
    if in_rec:
        fig.add_vrect(
            x0=start_date,
            x1=rec.index[-1],
            fillcolor="gray",
            opacity=0.25,
            line_width=0,
            layer="below",
        )

    return fig


# ============================================================
# NY Fed logit recession probability model
# ============================================================

def build_nyfed_prob(df):
    """
    Build NY Fed recession probability using 10Y–3M spread (monthly).
    Model: logit(P) = β0 + β1 * spread
    """

    # Need both series:
    if "DGS10" not in df or "DTB3" not in df:
        raise ValueError("DGS10 or DTB3 missing — cannot compute NY Fed probability.")

    # Compute spread (percentage points)
    spread = df["DGS10"] - df["DTB3"]

    # NY Fed coefficients
    beta0 = -0.5333
    beta1 = -1.6278

    # Compute probability
    logit = beta0 + beta1 * spread
    prob = 1 / (1 + np.exp(-logit))

    out = pd.DataFrame({
        "probability": prob * 100,   # convert to percentage
    }, index=df.index)

    return out

# ============================================================
# recession probability chart
# ============================================================
def fig_nyfed_prob(df):
    """Plot NY Fed recession probability with recession shading."""
    prob_df = build_nyfed_prob(df)

    fig = go.Figure()

    # Main probability line
    fig.add_trace(go.Scatter(
        x=prob_df.index,
        y=prob_df["probability"],
        mode="lines",
        name="NY Fed Recession Probability",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Probability: %{y:.2f}%<extra></extra>"
    ))

    # Add recession shading
    fig = add_recession_shading(fig, df)

    # Last update annotation
    fig = add_last_update_annotation(fig, prob_df, "probability")

    # Threshold reference lines (optional, but highly recommended)
    fig.add_hline(y=30, line_dash="dot", line_color="orange",
                  annotation_text="Watch (30%)", annotation_position="top left")

    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Danger (50%)", annotation_position="bottom left")

    fig.update_layout(
        title="NY Fed Recession Probability (10Y–3M Yield Spread Model)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        margin=dict(b=80),
    )

    return fig


# ============================================================
# TRIGGER EVALUATION
# ============================================================

def evaluate_triggers(df):
    """
    Evaluate triggers using the latest date where the needed
    inputs exist.

    Important change:
    - Use a filtered dataframe (df_trig) that drops NaNs only
      for the columns required for trigger logic, instead of
      forcing the entire df to end at the slowest series date.
    """

    required = ["UNRATE", "HY_OAS", "TEMP", "DELINQ"]
    df_trig = df.dropna(subset=required).copy()

    if len(df_trig) < 13:
        raise ValueError(
            "Not enough non-NaN history in trigger series "
            "(need at least 13 months)."
        )

    # Work only on trigger-ready subset
    un = df_trig["UNRATE"]
    hy = df_trig["HY_OAS"]
    temp = df_trig["TEMP"]
    delinq = df_trig["DELINQ"]

    triggers = {}

    # ---- Term Spread (2Y - 3M) ----
    ts_latest = df_trig["term_spread"].iloc[-1]

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

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def fig_term_spread(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["term_spread"],
        mode="lines",
        name="Term Spread",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Spread: %{y:.2f}%<extra></extra>"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(y=0.5, line_dash="dot", line_color="orange")

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "term_spread")

    fig.update_layout(
        title="Term Spread (2Y - 3M)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
        margin=dict(b=80),
    )
    return fig


def fig_hy(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["HY_OAS"],
        mode="lines",
        name="HY OAS",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>"
    ))

    # Warning lines
    fig.add_hline(y=4.5, line_dash="dot", line_color="orange",
                  annotation_text="4.5% (Watch)", annotation_position="top left")
    fig.add_hline(y=6.0, line_dash="dash", line_color="red",
                  annotation_text="6% (Danger)", annotation_position="bottom left")
    fig = add_recession_shading(fig, df)
    fig.update_layout(
        title="High Yield Spread (HY OAS)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig



def fig_temp(df):
    yoy = df["TEMP"].pct_change(12) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=yoy.index,
        y=yoy,
        mode="lines",
        name="Temp Help YoY",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>YoY: %{y:.2f}%<extra></extra>"
    ))

    fig.add_hline(y=-3, line_dash="dot", line_color="orange",
                  annotation_text="-3% (Watch)", annotation_position="top left")
    fig.add_hline(y=-6, line_dash="dash", line_color="red",
                  annotation_text="-6% (Danger)", annotation_position="bottom left")
    fig = add_recession_shading(fig, df)
    fig.update_layout(
        title="Temp Help YoY %",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="YoY %",
    )
    return fig


def fig_unrate(df):

    un = df["UNRATE"]
    un_3mma = un.rolling(3).mean()
    un_low_12m = un.rolling(12).min()
    sahm_threshold = un_low_12m + 0.5

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=un,
        mode="lines",
        name="Unemployment Rate",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Unrate: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=un_3mma,
        mode="lines",
        name="3-Month Moving Avg",
        line=dict(width=1, dash="dot"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>3M Avg: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=sahm_threshold,
        mode="lines",
        name="Sahm Rule Threshold",
        line=dict(width=1, dash="dash"),
    ))
    fig = add_recession_shading(fig, df)
    fig.update_layout(
        title="Unemployment Rate (with Sahm Rule Signal)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig


def fig_delinquency(df):
    delinq = df["DELINQ"]
    d_z = (delinq - delinq.rolling(60).mean()) / delinq.rolling(60).std()

    fig = go.Figure()

    # main line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=delinq,
        mode="lines",
        name="Credit Card Delinquencies",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Delinq: %{y:.2f}%<extra></extra>"
    ))

    # z-score danger / watch lines (projected onto value axis)
    # compute mean+z*std for last 5y to draw boundaries
    mean = delinq.rolling(60).mean()
    std = delinq.rolling(60).std()

    watch_line = mean + 0.5 * std
    danger_line = mean + 1.0 * std

    fig.add_trace(go.Scatter(
        x=df.index,
        y=watch_line,
        mode="lines",
        name="Watch Threshold (z=0.5)",
        line=dict(dash="dot", color="orange"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Watch: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=danger_line,
        mode="lines",
        name="Danger Threshold (z=1.0)",
        line=dict(dash="dash", color="red"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Danger: %{y:.2f}%<extra></extra>"
    ))
    fig = add_recession_shading(fig, df)
    fig.update_layout(
        title="Credit Card Delinquencies (with Warning Thresholds)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig
