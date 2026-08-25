"""One-off ERA5T monthly pull via the Copernicus CDS API.

Mirrors the exact request shape used to build the existing
data/raw/eta_2025_*.nc files (see ~/Desktop/energy/save_data.ipynb, cell 4)
so a newly pulled month has the same variables/provenance as the rest of
the year. Requires ~/.cdsapirc to be present (CDS URL + API key).

Usage:
    python scripts/pull_era5t_month.py --year 2025 --month 12 --out-dir era5t_pull
"""
import argparse
import calendar
import shutil
import time
import zipfile
from pathlib import Path

import cdsapi

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "cloud_cover",
]
AREA_US = [50, -130, 24, -66]  # N, W, S, E -- matches src/ingestion/era5.py's CONUS clip


def days_in_month(year: int, month: int) -> list[str]:
    return [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    year, month = args.year, f"{int(args.month):02d}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"era5t_us_{year}_{month}.nc"

    client = cdsapi.Client()
    for attempt in range(args.max_retries):
        try:
            print(f"Requesting ERA5T {year}-{month} (attempt {attempt + 1}/{args.max_retries})...")
            client.retrieve(
                "reanalysis-era5-single-levels",  # ERA5T switches automatically for recent dates
                {
                    "product_type": "reanalysis",
                    "variable": VARIABLES,
                    "year": year,
                    "month": month,
                    "day": days_in_month(int(year), int(month)),
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": AREA_US,
                    "format": "netcdf",
                },
                str(zip_path),
            )
            print(f"Saved: {zip_path}")
            break
        except Exception as e:
            print(f"Error: {e}")
            if attempt < args.max_retries - 1:
                wait = 20 * (attempt + 1)
                print(f"Waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    instant_file = out_dir / "data_stream-oper_stepType-instant.nc"
    final_file = out_dir / f"eta_{year}_{month}.nc"
    shutil.move(str(instant_file), str(final_file))
    print(f"Final file: {final_file}")


if __name__ == "__main__":
    main()
