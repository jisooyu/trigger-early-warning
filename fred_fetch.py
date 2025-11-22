# fred_fetch.py
import time
import pandas as pd
from pandas_datareader import data as web
import requests
from requests.adapters import HTTPAdapter, Retry


# ------------------------------------------------------------
# Shared requests Session with retries
# ------------------------------------------------------------
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://", HTTPAdapter(max_retries=retries))


# ------------------------------------------------------------
# Robust FRED fetcher
# ------------------------------------------------------------
def fred(series: str, start: str = "1990-01-01") -> pd.DataFrame:
    """
    Robust FRED fetcher with:
    - automatic retry
    - HTML error detection
    - clean warning messages
    """
    url = f"https://fred.stlouisfed.org/series/{series}"

    for attempt in range(1, 4):
        try:
            df = web.DataReader(series, "fred", start, session=session)

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
