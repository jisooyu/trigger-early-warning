# app.py
from dash import Dash, html, dcc

from preprocess import make_dashboard_df
from triggers import evaluate_triggers
from charts import (
    fig_term_spread,
    fig_hy,
    fig_temp,
    fig_unrate,
    fig_delinquency,
    fig_nyfed_prob,
)


# ------------------------------------------------------------
# Load data and triggers
# ------------------------------------------------------------
df = make_dashboard_df()

triggers = evaluate_triggers(df)

fast_cols = ["DGS2", "DTB3", "HY_OAS"]
last_date = df[fast_cols].dropna(how="all").index[-1].date()


# ------------------------------------------------------------
# Dash app
# ------------------------------------------------------------
app = Dash(__name__)
app.title = "Recession Trigger Dashboard"


COLOR = {
    "Normal": "#2e7d32",
    "Watch": "#f9a825",
    "Danger": "#c62828",
}


def trigger_card(name, info):
    return html.Div(
        style={
            "border": f"2px solid {COLOR[info['state']]}",
            "borderRadius": "8px",
            "padding": "10px",
            "margin": "6px",
            "width": "260px",
            "backgroundColor": "#111",
        },
        children=[
            html.Div(name, style={"fontWeight": "bold"}),
            html.Div(
                info["state"],
                style={"color": COLOR[info["state"]], "fontWeight": "bold"},
            ),
            html.Div(info["detail"], style={"fontSize": "11px"}),
        ],
    )


app.layout = html.Div(
    style={"backgroundColor": "#000", "color": "#fff", "padding": "20px"},
    children=[
        html.H2("Recession Trigger Dashboard", style={"textAlign": "center"}),
        html.Div(
            f"Latest market data (rates/HY): {last_date}",
            style={"textAlign": "center"},
        ),
        html.Br(),

        # Trigger cards
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "justifyContent": "center",
            },
            children=[trigger_card(name, info) for name, info in triggers.items()],
        ),

        html.Br(),
        html.Hr(),

        # Charts
        dcc.Graph(figure=fig_term_spread(df)),
        dcc.Graph(figure=fig_hy(df)),
        dcc.Graph(figure=fig_temp(df)),
        dcc.Graph(figure=fig_unrate(df)),
        dcc.Graph(figure=fig_delinquency(df)),
        dcc.Graph(figure=fig_nyfed_prob(df)),
    ],
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
