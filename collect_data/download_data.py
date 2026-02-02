import locale
import sys
import os
import time
import math
import random
import calendar
import urllib.parse
from datetime import date

import pandas as pd
import numpy as np
import polars as pl
import pyarrow
import matplotlib.pyplot as plt
import optuna
import requests
import tensorflow as tf

from calendar import day_abbr
from pathlib import Path
from datetime import datetime
from typing import Tuple, Union, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

DATASTREAM_DB: Dict[str, str | int] = {
    "user": "student",
    "password": "#q6a21I&OA5k",
    "host": "132.252.60.112",
    "port": 3306,
    "dbname": "DATASTREAM",}
ENTSOE_DB = {
    "user": "student",
    "password": "#q6a21I&OA5k",
    "host": "132.252.60.112",
    "port": 3306,
    "dbname": "ENTSOE"}
DEFAULT_START_DATE = date.fromisoformat("2020-01-01") 
DEFAULT_END_DATE   = date.fromisoformat("2025-12-31" ) 
DEFAULT_START_YEAR = DEFAULT_START_DATE.year
DEFAULT_END_YEAR   = DEFAULT_END_DATE.year
DEFAULT_MAPCODES   = ["ES"]
DEFAULT_VARS = [
    "coal_fM_01",  # front‑month ICE Rotterdam coal (USD)
    "gas_fM_01",   # front‑month TTF gas (EUR)
    "oil_fM_01",   # ICE Brent (USD)
    "EUA_fM_01",   # EUA allowance front‑month (EUR)
    "USD_EUR",]    # FX rate USD -> EUR (quote: 1 USD = x EUR)    


def setup_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_device() -> str:
    return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

def make_mysql_engine(cfg: dict) -> Engine:
    url = (
        "mysql+pymysql://"
        f"{urllib.parse.quote_plus(str(cfg['user']))}:"
        f"{urllib.parse.quote_plus(str(cfg['password']))}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}")
    return create_engine(url)

def get_tables(engine: Engine) -> List[str]:
    return inspect(engine).get_table_names()

def read_table(engine: Engine, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table}", engine)

def build_backbone(
    mapcodes: List[str],
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> pl.DataFrame:
    start_ts = datetime(start_year, 1, 1)
    end_ts_excl = datetime(end_year + 1, 1, 1) 
    hours_df = pl.select(pl.datetime_range(start_ts, end_ts_excl, interval="1h", closed="left").alias("time_utc"))
    backbone = (pl.DataFrame({"MapCode": mapcodes}).join(hours_df, how="cross").sort(["MapCode", "time_utc"]))
    return backbone

def prepare_datastream(
    engine: Engine,
    variables: List[str] | None = None,
    start: date = DEFAULT_START_DATE,
    end: date = DEFAULT_END_DATE,
) -> pl.DataFrame:
    
    vars_to_use = variables or DEFAULT_VARS
    raw = pd.read_sql_query("SELECT * FROM datastream", engine)
    raw = raw.loc[raw["name"].isin(vars_to_use)].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    mask = (raw["Date"] >= pd.Timestamp(start)) & (raw["Date"] <= pd.Timestamp(end))
    raw = raw.loc[mask].sort_values("Date")
    if "RIC" in raw.columns: raw = raw.drop(columns="RIC")
    wide = raw.pivot(index="Date", columns="name", values="Value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Date": "time_utc"})
    if {"oil_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["oil_fM_01_EUR"] = wide["oil_fM_01"] / wide["USD_EUR"]
    if {"coal_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["coal_fM_01_EUR"] = wide["coal_fM_01"] / wide["USD_EUR"]
    n_days = len(wide)
    wide = wide.loc[wide.index.repeat(24)].copy()
    hours = np.tile(np.arange(24).astype("timedelta64[h]"), n_days)
    wide["time_utc"] = pd.to_datetime(wide["time_utc"]).to_numpy() + hours

    return pl.from_pandas(wide)

def download_data(
    targets: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> pd.DataFrame:
    ids_list = targets["TimeSeriesID"].dropna().astype(int).tolist()
    if not ids_list: return pd.DataFrame()

    ids = ", ".join(map(str, ids_list))
    start_dt = f"{start_year}-01-01"
    end_dt_excl = f"{end_year + 1}-01-01"
    values_query = f"""
    SELECT *
    FROM vals
    WHERE TimeSeriesID IN ({ids})
      AND `DateTime` >= '{start_dt}'
      AND `DateTime` <  '{end_dt_excl}'
    """
    values = pd.read_sql_query(values_query, engine)
    return pd.merge(values, targets, on="TimeSeriesID", how="inner")

def group_hourly(
    df: pd.DataFrame,
    key_cols: List[str],
    value_cols: List[str],
    agg_func: str = "sum",
) -> pl.DataFrame:
    df = df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"]).dt.floor("h")
    pl_df = pl.from_pandas(df)
    agg = [(pl.col(c).sum() if agg_func == "sum" else pl.col(c).mean()).alias(c) for c in value_cols]
    return pl_df.group_by(key_cols).agg(agg).sort(key_cols)

def _entsoe_targets(spec: pd.DataFrame, name: str, mapcodes: List[str], types: List[str]) -> pd.DataFrame:
    return spec[
        (spec["Name"] == name)
        & (spec["Type"].isin(types))
        & (spec["OutMapCode"].isin(mapcodes))
        & (spec["OutMapTypeCode"].str.contains(r"\bBZN\b", na=False))]

def prepare_load(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    targets = _entsoe_targets(spec, name="Load", mapcodes=mapcodes, types=["DayAhead", "Actual"])
    df = download_data(targets, engine, start_year, end_year)
    df = df[["DateTime", "Type", "Value", "OutMapCode"]].rename(columns={"OutMapCode": "MapCode"})
    pivot = df.pivot_table(index=["DateTime", "MapCode"], columns=["Type"], values="Value").reset_index()
    pivot = pivot.rename(columns={"DateTime": "time_utc", "Actual": "Load_A", "DayAhead": "Load_DA"})
    return group_hourly(pivot, ["MapCode", "time_utc"], ["Load_A", "Load_DA"], agg_func="sum")

def prepare_price(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
  ) -> pl.DataFrame:
    targets = _entsoe_targets(spec, name="Prices", mapcodes=mapcodes, types=["DayAhead"])
    df = download_data(targets, engine, start_year, end_year)
    df = df[["DateTime", "Value", "OutMapCode"]].rename(columns={"OutMapCode": "MapCode"})
    df["time_utc"] = pd.to_datetime(df["DateTime"]).dt.floor("h")
    pl_df = pl.from_pandas(df[["MapCode", "time_utc", "Value"]]).rename({"Value": "Price"})    
    return (pl_df.group_by(["MapCode", "time_utc"]).agg(pl.col("Price").mean()).sort(["MapCode", "time_utc"]))
    
def prepare_generation(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    targets = _entsoe_targets(spec, name="Generation", mapcodes=mapcodes, types=["DayAhead", "Actual"])
    df = download_data(targets, engine, start_year, end_year)
    df = df[["DateTime", "Type", "ProductionType", "Value", "OutMapCode"]].rename(columns={"OutMapCode": "MapCode"})
    pivot = df.pivot_table(index=["DateTime", "MapCode"], columns=["Type", "ProductionType"], values="Value").reset_index()
    suffix = {"Actual": "A", "DayAhead": "DA"}
    pivot.columns = [
        "time_utc" if t == "DateTime" else
        "MapCode"  if t == "MapCode" else
        f"{prod.replace(' ', '_')}_{suffix.get(t, t)}"
        for (t, prod) in pivot.columns]
    value_cols = [c for c in pivot.columns if c not in ["MapCode", "time_utc"]]
    return group_hourly(pivot, ["MapCode", "time_utc"], value_cols, agg_func="sum")

def _cast_time_us(df: pl.DataFrame, col: str = "time_utc") -> pl.DataFrame:
    return df.with_columns(pl.col(col).cast(pl.Datetime("us")))

def merge_on_backbone(
    backbone: pl.DataFrame,
    price: pl.DataFrame,
    load: pl.DataFrame,
    generation: pl.DataFrame,
    fuels: pl.DataFrame,
) -> pl.DataFrame:
    
    backbone   = _cast_time_us(backbone)
    price      = _cast_time_us(price)
    load       = _cast_time_us(load)
    generation = _cast_time_us(generation)
    fuels      = _cast_time_us(fuels)

    merged = (
        backbone
        .join(price, on=["MapCode", "time_utc"], how="left")
        .join(load, on=["MapCode", "time_utc"], how="left")
        .join(generation, on=["MapCode", "time_utc"], how="left")
        .join(fuels, on=["time_utc"], how="left")
        .sort(["MapCode", "time_utc"])
    )
    return merged

def _ffill_by_mapcode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame: 
    df = df.sort_values("time_utc")
    df[columns] = df[columns].ffill()
    return df

_RENAME_MAP = {"Wind_Onshore_DA": "WindOn_DA","coal_fM_01_EUR": "Coal","gas_fM_01": "NGas","oil_fM_01_EUR": "Oil","EUA_fM_01": "EUA","Solar_unspecified_DA": "Solar_DA"}
_REGRESSOR_COLUMNS = ["time_utc","Price","Load_A","Load_DA", "WindOn_DA","Solar_DA","Coal","NGas","Oil","EUA",]

def build_training_dataset(
    merged: pl.DataFrame,
    mapcode: Optional[str] = None,
    save_csv: Optional[str | Path] = None,
    fill_weekends: bool = True,
) -> pd.DataFrame:
    df = merged.to_pandas()
    if fill_weekends:
        ffill_cols = [c for c in ["EUA_fM_01","gas_fM_01","oil_fM_01","oil_fM_01_EUR","coal_fM_01","coal_fM_01_EUR"] if c in df.columns]
        if ffill_cols: df = _ffill_by_mapcode(df, ffill_cols)
    df = df.rename(columns=_RENAME_MAP)
    if mapcode is not None:
        df = df[df["MapCode"] == mapcode].drop(columns="MapCode")
    df = df[_REGRESSOR_COLUMNS]
    if save_csv is not None:
        save_path = Path(save_csv).expanduser()
        df.to_csv(save_path, index=False)

    return df

# execute everything
setup_seed(42)
device    = get_device()
engine    = make_mysql_engine(ENTSOE_DB)
spec      = read_table(engine, "spec")
gen_pl    = prepare_generation(spec, engine)
load_pl   = prepare_load(spec, engine)
price_pl  = prepare_price(spec, engine)
ds_engine = make_mysql_engine(DATASTREAM_DB)
fuel_pl   = prepare_datastream(ds_engine)
backbone  = build_backbone(DEFAULT_MAPCODES, DEFAULT_START_YEAR, DEFAULT_END_YEAR)
merged_pl = merge_on_backbone(backbone, price_pl, load_pl, gen_pl, fuel_pl)
repo_root = Path.cwd()
train_df  = build_training_dataset(merged=merged_pl,mapcode=DEFAULT_MAPCODES[0],save_csv=repo_root / "Datasets" / f"{DEFAULT_MAPCODES[0]}_raw.csv",)

# featur engineering RES and orders
train_df = train_df.drop(columns=["Load_A"])
train_df["Price"] = train_df["Price"].fillna(0)
train_df["Renewables_DA_Forecast"] = train_df["WindOn_DA"] + train_df["Solar_DA"]
train_df = train_df.drop(columns=["WindOn_DA", "Solar_DA"])
order = ["time_utc", "Price", "Load_DA", "Renewables_DA_Forecast", "EUA", "Coal", "NGas", "Oil"]
train_df = train_df[order]
train_df.to_csv(repo_root / "Datasets" / f"{DEFAULT_MAPCODES[0]}_all.csv", index=False)

