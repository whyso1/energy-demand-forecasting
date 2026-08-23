"""EIA hourly demand ingestion.

Unlike the legacy preprocess_dataset_energy(), this filters to a single
`respondent` before aggregating. The legacy code's
`groupby(["period", "type"]).mean()` folded together individual
balancing authorities (e.g. CISO, ERCO), EIA's own regional rollups
(e.g. CAL, CENT, MIDW), and the national US48 total into one mean per
timestamp -- a double-counting bug. Filtering to a single respondent
(config default: "US48", EIA's own Lower-48 aggregate) fixes that and
gives a series that actually corresponds to one physical quantity.
"""
import pandas as pd


def load_eia_series(csv_path: str, respondent: str, type_code: str) -> pd.Series:
    """Load one EIA (respondent, type) hourly series as a pd.Series
    indexed by 'time', named after the type code (e.g. 'D' or 'DF').
    """
    df = pd.read_csv(csv_path, usecols=["period", "respondent", "type", "value"])
    df = df[(df["respondent"] == respondent) & (df["type"] == type_code)]
    if df.empty:
        raise ValueError(f"No rows found for respondent={respondent!r}, type={type_code!r}")

    df["period"] = pd.to_datetime(df["period"])
    series = df.set_index("period")["value"].sort_index()
    series.index.name = "time"
    series.name = type_code
    return series.dropna()
