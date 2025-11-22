# charts.py
import pandas as pd
import plotly.graph_objects as go

from preprocess import zscore
from plot_utils import add_recession_shading, add_last_update_annotation
from models import build_nyfed_prob


# ------------------------------------------------------------
# Term Spread
# ------------------------------------------------------------
def fig_term_spread(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["term_spread"],
            mode="lines",
            name="Term Spread",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Spread: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.add_hline(y=0.5, line_dash="dot", line_color="orange")

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "term_spread")

    fig.update_layout(
        title="Term Spread (2Y - 3M)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig


# ------------------------------------------------------------
# HY OAS
# ------------------------------------------------------------
def fig_hy(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["HY_OAS"],
            mode="lines",
            name="HY OAS",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "HY OAS: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=4.5, line_dash="dot", line_color="orange")
    fig.add_hline(y=6.0, line_dash="dash", line_color="red")

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "HY_OAS")

    fig.update_layout(
        title="High Yield Spread (HY OAS)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig


# ------------------------------------------------------------
# Temp Help YoY
# ------------------------------------------------------------
def fig_temp(df: pd.DataFrame) -> go.Figure:
    yoy = df["TEMP"].pct_change(12) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=yoy.index,
            y=yoy,
            mode="lines",
            name="Temp Help YoY",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "YoY: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=-3, line_dash="dot", line_color="orange")
    fig.add_hline(y=-6, line_dash="dash", line_color="red")

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "TEMP")

    fig.update_layout(
        title="Temp Help YoY %",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="YoY %",
    )
    return fig


# ------------------------------------------------------------
# Unemployment Rate (Sahm Rule)
# ------------------------------------------------------------
def fig_unrate(df: pd.DataFrame) -> go.Figure:
    un = df["UNRATE"]
    un_3mma = un.rolling(3).mean()
    un_low_12m = un.rolling(12).min()
    sahm_threshold = un_low_12m + 0.5

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=un,
            mode="lines",
            name="Unemployment Rate",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Unrate: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=un_3mma,
            mode="lines",
            name="3-Month MA",
            line=dict(dash="dot"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "3M Avg: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sahm_threshold,
            mode="lines",
            name="Sahm Threshold",
            line=dict(dash="dash"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Threshold: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "UNRATE")

    fig.update_layout(
        title="Unemployment Rate (Sahm Rule)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig


# ------------------------------------------------------------
# Credit Card Delinquencies
# ------------------------------------------------------------
def fig_delinquency(df: pd.DataFrame) -> go.Figure:
    delinq = df["DELINQ"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=delinq,
            mode="lines",
            name="Delinquencies",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Delinq: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    # Optional z-score-based watch/danger bands on value axis
    mean = delinq.rolling(60).mean()
    std = delinq.rolling(60).std()
    watch_line = mean + 0.5 * std
    danger_line = mean + 1.0 * std

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=watch_line,
            mode="lines",
            name="Watch (z=0.5)",
            line=dict(dash="dot", color="orange"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Watch: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=danger_line,
            mode="lines",
            name="Danger (z=1.0)",
            line=dict(dash="dash", color="red"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Danger: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, df, "DELINQ")

    fig.update_layout(
        title="Credit Card Delinquency Rate",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig


# ------------------------------------------------------------
# NY Fed Recession Probability (10Y–3M model)
# ------------------------------------------------------------
def fig_nyfed_prob(df: pd.DataFrame) -> go.Figure:
    prob_df = build_nyfed_prob(df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prob_df.index,
            y=prob_df["probability"],
            mode="lines",
            name="NY Fed Probability",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Probability: %{y:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="orange",
        annotation_text="Watch (30%)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        annotation_text="Danger (50%)",
        annotation_position="bottom left",
    )

    fig = add_recession_shading(fig, df)
    fig = add_last_update_annotation(fig, prob_df, "probability")

    fig.update_layout(
        title="NY Fed Recession Probability (10Y–3M Spread Model)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
    )
    return fig
