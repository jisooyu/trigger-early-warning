# preprocess.py
import pandas as pd
from fred_fetch import fred


# ------------------------------------------------------------
# Build main dashboard dataframe from FRED
# ------------------------------------------------------------
def make_dashboard_df(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Load all indicators and resample to month-end.
    Keeps union of dates (outer join) so fast indicators
    are not truncated by slower ones.
    """

    series_map = {
        "DGS2":         "DGS2",     # 2Y
        "DTB3":         "DTB3",     # 3M
        "BAMLH0A0HYM2": "HY_OAS",   # HY OAS
        "TEMPHELPS":    "TEMP",     # Temp help employment
        "DRCCLACBS":    "DELINQ",   # Credit card delinquencies (quarterly)
        "UNRATE":       "UNRATE",   # Unemployment rate
        "USREC":        "USREC",    # NBER recession indicator (0/1)
        "DGS10":        "DGS10",    # 10Y Treasury for NY Fed model
    }

    monthly_dfs = []

    for fred_series, col_name in series_map.items():
        s = fred(fred_series, start=start)

        if col_name == "DELINQ":
            # quarterly → monthly, forward fill across months
            s_me = s.resample("ME").last().ffill()
        elif col_name == "USREC":
            # monthly 0/1 → ensure month-end & ffill
            s_me = s.resample("ME").last().ffill()
        else:
            # daily/monthly → month-end last
            s_me = s.resample("ME").last()

        s_me.columns = [col_name]
        monthly_dfs.append(s_me)

    # Outer join across all month-end series
    df = pd.concat(monthly_dfs, axis=1, join="outer").sort_index()

    # Term spread (2Y − 3M)
    df["term_spread"] = df["DGS2"] - df["DTB3"]

    return df


# ------------------------------------------------------------
# Generic z-score utility (can be reused by triggers/charts)
# ------------------------------------------------------------
def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()
