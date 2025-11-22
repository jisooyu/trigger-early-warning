# plot_utils.py
import pandas as pd
import plotly.graph_objects as go


def add_recession_shading(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Add NBER recession shading using USREC == 1 regions.
    Assumes df['USREC'] exists and is 0/1 monthly.
    """
    if "USREC" not in df.columns:
        return fig

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

    # Ongoing recession
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


def add_last_update_annotation(
    fig: go.Figure, data_df: pd.DataFrame, series_name: str
) -> go.Figure:
    """
    Adds 'Last updated: YYYY-MM-DD' annotation under the chart,
    based on the last non-NaN value of data_df[series_name].
    """
    if series_name not in data_df.columns:
        return fig

    last_valid = data_df[series_name].dropna().index[-1]
    last_str = last_valid.strftime("%Y-%m-%d")

    fig.add_annotation(
        text=f"Last updated: {last_str}",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.18,  # under the x-axis
        showarrow=False,
        font=dict(size=10, color="#cccccc"),
    )

    # Make space for annotation at bottom
    fig.update_layout(margin=dict(b=80))

    return fig
