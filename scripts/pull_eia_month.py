"""One-off EIA hourly demand pull for a specific respondent/date range.

Mirrors the query shape used to build the original data/raw/eia_load_data.csv
(https://api.eia.gov/v2/electricity/rto/region-data/data/, per
~/Desktop/energy/save_data.ipynb), but scoped to just the respondent/types
this pipeline actually consumes -- no need to re-pull all 76 EIA respondent
codes to add one month.

Usage:
    EIA_API_KEY=... python scripts/pull_eia_month.py \
        --start 2025-12-01T00 --end 2025-12-31T23 \
        --respondent US48 --types D DF \
        --out data/pulled/eia_us48_202512.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
KEEP_COLS = ["period", "respondent", "respondent-name", "type", "type-name", "value", "value-units"]


def pull(start, end, respondent, types, api_key, page_size=5000, max_retries=5, retry_wait=5):
    all_rows = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "facets[type][]": types,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }

        resp_json = None
        for attempt in range(max_retries):
            try:
                r = requests.get(BASE_URL, params=params, timeout=60)
                r.raise_for_status()
                resp_json = r.json()
                break
            except Exception as e:
                print(f"attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
                time.sleep(retry_wait)
        if resp_json is None:
            raise RuntimeError("EIA API request failed after retries")

        rows = resp_json["response"]["data"]
        if not rows:
            break
        all_rows.extend(rows)

        total = int(resp_json["response"]["total"])
        offset += page_size
        print(f"  fetched {min(offset, total)}/{total} rows")
        if offset >= total:
            break
        time.sleep(0.5)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="e.g. 2025-12-01T00")
    parser.add_argument("--end", required=True, help="e.g. 2025-12-31T23")
    parser.add_argument("--respondent", default="US48")
    parser.add_argument("--types", nargs="+", default=["D", "DF"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    api_key = os.environ["EIA_API_KEY"]
    df = pull(args.start, args.end, args.respondent, args.types, api_key)
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
