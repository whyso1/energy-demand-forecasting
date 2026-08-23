"""ERA5 ingestion: CONUS clip, unit conversion, wind speed, spatial mean.

Ported from the legacy notebook's preprocess_dataset_era5(), with the
unused cartopy import dropped and the output standardized to a 'time'
index so downstream code never has to detect the time dimension name.
"""
import numpy as np
import pandas as pd
import xarray as xr


def load_era5_conus_mean(
    path_pattern: str,
    lat_slice: tuple[float, float] = (50, 24),
    lon_slice: tuple[float, float] = (-125, -66),
) -> pd.DataFrame:
    """Load ERA5 files, clip to CONUS, convert K->C, add wind speed,
    and return the CONUS spatial mean as an hourly DataFrame indexed
    by 'time'.
    """
    ds = xr.open_mfdataset(path_pattern, engine="netcdf4", combine="by_coords")
    ds_conus = ds.sel(latitude=slice(*lat_slice), longitude=slice(*lon_slice))

    da_vars = {}
    for name, da in ds_conus.data_vars.items():
        long_name = da.attrs.get("long_name", "").lower()
        units = da.attrs.get("units", "").upper()
        if "temperature" in long_name and units == "K":
            da = da - 273.15
            da.attrs["units"] = "C"
            da.attrs["long_name"] = da.attrs.get("long_name", name)
        da_vars[name] = da

    if "u10" in da_vars and "v10" in da_vars:
        da_vars["wind"] = np.sqrt(da_vars["u10"] ** 2 + da_vars["v10"] ** 2)
        da_vars["wind"].attrs.update({"units": "m/s", "long_name": "10m wind speed"})

    merged = xr.Dataset(da_vars)
    spatial_mean = merged.mean(dim=["latitude", "longitude"])

    df = spatial_mean.to_dataframe()
    time_col = [c for c in df.index.names if pd.api.types.is_datetime64_any_dtype(df.index.get_level_values(c))]
    if not time_col:
        raise ValueError("No datetime coordinate found in ERA5 dataset.")
    df.index = df.index.get_level_values(time_col[0])
    df.index.name = "time"

    df = df.dropna(how="any").sort_index()
    return df
