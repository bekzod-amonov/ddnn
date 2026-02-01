"""merge_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities that combine the ENTSOE market data (price, load, generation) with
fuel‑/FX‑related variables from *datastream_data* into a single modelling table
and optionally persist the result to ``.csv``.
"""

from pathlib import Path
from typing import List, Optional
import pyarrow
import pandas as pd
import polars as pl

def merge_datasets(
    price: pl.DataFrame,
    load: pl.DataFrame,
    generation: pl.DataFrame,
    fuels: pl.DataFrame,
    join_type: str = "left",) -> pl.DataFrame:
    merged = (price
        .join(load, on=["MapCode", "time_utc"], how=join_type)
        .join(generation, on=["MapCode", "time_utc"], how=join_type)
        .join(fuels, on=["time_utc"], how=join_type) )

    return merged

def _ffill_weekend_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df[columns] = df.groupby("MapCode")[columns].transform("ffill")
    return df

_RENAME_MAP = {
    "Wind_Onshore_DA": "WindOn_DA",
    "coal_fM_01_EUR": "Coal",
    "gas_fM_01": "NGas",
    "oil_fM_01_EUR": "Oil",
    "EUA_fM_01": "EUA",
    "Solar_unspecified_DA": "Solar_DA"}

_REGRESSOR_COLUMNS = [
    "time_utc",
    "Price",
    "Load_A",
    "Load_DA",          
    "WindOn_DA",        # renewables generation
    "Solar_DA",         # renewables generation
    "Coal",             # fossil fuel generation
    "NGas",             # fossil fuel generation
    "Oil",              # fossil fuel generation
    "EUA",              # fossil fuel generation
    #"Temp",             # weather
    #"Solar",            # weather
    #"WindS",            # weather
    #"WindDir",          # weather
    #"Press",            # weather
    #"Humid"             # weather
]

def build_training_dataset(
    merged: pl.DataFrame,
    mapcode: Optional[str] = None,
    save_csv: Optional[str | Path] = None,
    fill_weekends: bool = True,) -> pd.DataFrame:

    df = merged.to_pandas()
    if fill_weekends:
        _fill_cols = [
            "EUA_fM_01",
            "USD_EUR",
            "gas_fM_01",
            "oil_fM_01",
            "oil_fM_01_EUR",
            "coal_fM_01",
            "coal_fM_01_EUR",
            #"Wind_Offshore_DA",
            #"Wind_Onshore_DA",
        ]
        df = _ffill_weekend_values(df, _fill_cols)
    df = df.rename(columns=_RENAME_MAP)

    if mapcode is not None:
        df = df[df["MapCode"] == mapcode]
        df = df.drop(columns="MapCode")

    df = df[_REGRESSOR_COLUMNS]
    if save_csv is not None:
        Path(save_csv).expanduser().with_suffix(".csv")
        df.to_csv(save_csv, index=False)

    return df