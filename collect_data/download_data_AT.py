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


def setup_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return "/GPU:0" if gpus else "/CPU:0"

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

# Default parameters for the prep helper
DEFAULT_START_DATE = date.fromisoformat("2010-01-01") 
DEFAULT_END_DATE   = date.fromisoformat("2025-12-31" ) 
DEFAULT_START_YEAR = DEFAULT_START_DATE.year
DEFAULT_END_YEAR   = DEFAULT_END_DATE.year
DEFAULT_MAPCODES   = ["AT"]
DEFAULT_VARS = [
    "coal_fM_01",  # front‑month ICE Rotterdam coal (USD)
    "gas_fM_01",   # front‑month TTF gas (EUR) – already EUR‑denominated
    "oil_fM_01",   # ICE Brent (USD)
    "EUA_fM_01",   # EUA allowance front‑month (EUR)
    "USD_EUR",]     # FX rate USD → EUR (quote: 1 USD = x EUR)    

# Engine / Meta helpers
def _make_url(cfg: Dict[str, str | int]) -> str:
    """Return a SQLAlchemy URL for *pymysql*."""
    return (
        "mysql+pymysql://"
        f"{urllib.parse.quote_plus(str(cfg['user']))}:"
        f"{urllib.parse.quote_plus(str(cfg['password']))}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}")

def create_datastream_engine(cfg: Optional[Dict[str, str | int]] = None) -> Engine:
    """Instantiate a SQLAlchemy engine for the DATASTREAM schema."""
    return create_engine(_make_url(cfg or DATASTREAM_DB))

def get_tables(engine: Engine) -> List[str]:
    """List tables available in the connected database."""
    return inspect(engine).get_table_names()

# ETL helper
def prepare_datastream(
    engine: Engine,
    variables: List[str] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,) -> pl.DataFrame:
    vars_to_use = variables or DEFAULT_VARS
    raw = pd.read_sql_query("SELECT * FROM datastream", engine)
    raw = raw[raw["name"].isin(vars_to_use)].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (raw["Date"] >= start_ts) & (raw["Date"] <= end_ts)
    raw = raw.loc[mask]
    raw = raw.sort_values("Date")
    if "RIC" in raw.columns:
        raw = raw.drop(columns="RIC")
    wide = raw.pivot(index="Date", columns="name", values="Value").reset_index()

    # Currency conversions
    if {"oil_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["oil_fM_01_EUR"] = wide["oil_fM_01"] / wide["USD_EUR"]
    if {"coal_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["coal_fM_01_EUR"] = wide["coal_fM_01"] / wide["USD_EUR"]

    wide.columns.name = None
    wide = wide.rename(columns={"Date": "time_utc"})
    n = len(wide)
    wide = wide.loc[wide.index.repeat(24)].copy()
    hours = np.tile(np.arange(24, dtype="timedelta64[h]"), n)
    wide["time_utc"] = pd.to_datetime(wide["time_utc"]) + hours

    return pl.from_pandas(wide)

def _build_payload(lat: float, lon: float, start: str, end: str) -> Dict[str, str]:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(
            [
                "temperature_2m",
                "shortwave_radiation",
                "windspeed_10m",
                "winddirection_10m",
                "pressure_msl",
                "relative_humidity_2m",
            ]
        ),
        "timezone": "Europe/Oslo",  # UTC+1 / UTC+2 DST – consistent with ENTSOE MapCodes
    }

def create_entsoe_engine(config: dict = None):
    """
    Create a SQLAlchemy engine for the ENTSOE MySQL database.
    Parameters:
    - config: Optional dict with keys 'user', 'password', 'host', 'port', 'dbname'.
              If None, uses the default ENTSOE_DB.
    Returns:
    - SQLAlchemy Engine
    """
    cfg = config or ENTSOE_DB
    url = (
        f"mysql+pymysql://"
        f"{urllib.parse.quote_plus(cfg['user'])}:"
        f"{urllib.parse.quote_plus(cfg['password'])}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )
    return create_engine(url)

def get_tables(engine):
    """
    List all table names in the connected database.
    Parameters:
    - engine: SQLAlchemy Engine
    Returns:
    - List of table names (List[str])
    """
    inspector = inspect(engine)
    return inspector.get_table_names()

def get_spec(engine) -> pd.DataFrame:
    """
    Load the full 'spec' table into a pandas DataFrame.
    Parameters:
    - engine: SQLAlchemy Engine
    Returns:
    - DataFrame with all rows from 'spec'
    """
    query = "SELECT * FROM spec"
    return pd.read_sql_query(query, engine)

def get_market_divisions(spec_df: pd.DataFrame):
    """
    Return unique market division codes (MapTypeCode) in spec.
    """
    return spec_df.MapTypeCode.unique()

def get_map_codes(spec_df: pd.DataFrame):
    return spec_df.MapCode.unique()

def get_map_codes_starting_with(spec_df: pd.DataFrame, prefix: str):
    """
    Return MapCode values starting with the given prefix.
    """
    mask = spec_df.MapCode.fillna('').str.startswith(prefix)
    return spec_df.MapCode[mask].unique()

def get_resolution_codes(spec_df: pd.DataFrame):
    """
    Return unique ResolutionCode values in spec (e.g., 'PT60M').
    """
    return spec_df.ResolutionCode.unique()

def download_data(targets: pd.DataFrame, 
                  engine, 
                  start_year: int = DEFAULT_START_YEAR,
                  end_year: int = DEFAULT_END_YEAR,) -> pd.DataFrame:
    ids_list = targets["TimeSeriesID"].dropna().tolist()
    ids = ", ".join(map(str, ids_list))
    start_dt = f"{start_year}-01-01"
    end_dt_exclusive = f"{end_year + 1}-01-01"   
    values_query = f"""  
    SELECT *
    FROM vals
    WHERE TimeSeriesID IN ({ids})
      AND `DateTime` >= '{start_dt}'
      AND `DateTime` <= '{end_dt_exclusive}'
    """
    values = pd.read_sql_query(values_query, engine)
    merged = pd.merge(values, targets, on='TimeSeriesID')
    return merged

def group_hourly(
    df: pd.DataFrame,
    key_cols: List[str],
    value_cols: List[str],
    agg_func: str = 'sum',
) -> pl.DataFrame:
    df = df.copy()
    df['time_utc'] = pd.to_datetime(df['time_utc']).dt.floor('h')
    pl_df = pl.from_pandas(df)
    if agg_func == 'sum':
        agg = [pl.col(c).sum().alias(c) for c in value_cols]
    else:
        agg = [pl.col(c).mean().alias(c) for c in value_cols]
    return (pl_df.group_by(key_cols).agg(agg).sort(key_cols))

def prepare_generation(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,) -> pl.DataFrame:
    prod_types = ['Wind Onshore','Wind Offshore','Solar','DC Link','AC Link']
    targets = spec[
        (spec['Name']=='Generation') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        #(spec['ProductionType'].isin(prod_types)) &
        (spec['OutMapCode'].isin(mapcodes)) &
        #(spec['OutMapTypeCode']=='BZN')
        spec['OutMapTypeCode'].str.contains(r'\bBZN\b', na=False)]
    print(targets.groupby(["Type", "ProductionType"]))
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','ProductionType','Value','OutMapCode']].rename(columns={'OutMapCode':'MapCode'})
    df = df.sort_values('DateTime')
    print(df.columns.to_list())
    pivot = df.pivot_table(index=['DateTime','MapCode'],columns=['Type','ProductionType'],values='Value').reset_index()
    # flatten column names
    suffix = {'Actual':'A','DayAhead':'DA'}
    pivot.columns = [
        'time_utc' if t=='DateTime' else
        'MapCode' if t=='MapCode' else
        f"{prod.replace(' ','_')}_{suffix.get(t,t)}"
        for (t,prod) in pivot.columns]
    return group_hourly(
        pivot, key_cols=['MapCode','time_utc'],
        value_cols=[c for c in pivot.columns if c not in ['MapCode','time_utc']],
        agg_func='sum')
#
def prepare_load(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare load data.
    """
    targets = spec[
        (spec['Name']=='Load') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        (spec['OutMapCode'].isin(mapcodes)) &
        (spec['OutMapTypeCode'].str.contains(r'\bBZN\b', na=False))]
    
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','Value','OutMapCode']].rename(columns={"OutMapCode":"MapCode"})
    pivot = df.pivot_table(
        index=['DateTime','MapCode'],
        columns=['Type'],
        values='Value'
    ).reset_index()
    pivot.columns = ['time_utc','MapCode','Load_A','Load_DA']
    return group_hourly(
        pivot, ['MapCode','time_utc'], ['Load_A','Load_DA'], agg_func='sum'
    )

def prepare_price(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    targets = spec[
        (spec['Name']=='Prices') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        (spec['OutMapCode'].isin(mapcodes)) &
        (spec['OutMapTypeCode'].str.contains(r'\bBZN\b', na=False))
    ]
    
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','Value','OutMapCode']].rename(columns={"OutMapCode":"MapCode"})
    pivot = df.pivot_table(
        index=['DateTime','MapCode'],
        columns=['Type'],
        values='Value'
    ).reset_index()
    pivot.columns = ['time_utc','MapCode','Price']
    return group_hourly(
        pivot, ['MapCode','time_utc'], ['Price'], agg_func='mean')

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

# setup server and seed
setup_seed(42)
device = get_device()

# entsoe_data - connect to engine
engine = create_entsoe_engine()
tables = get_tables(engine)
spec = get_spec(engine)

# entsoe_data - collect data
gen_pl = prepare_generation(spec, engine)
load_pl = prepare_load(spec, engine)
price_pl = prepare_price(spec, engine)

# datastream data
ds_engine = create_datastream_engine()
ds_tables = get_tables(ds_engine)
fuel_pl = prepare_datastream(ds_engine)

# merge everything
merged_pl = merge_datasets(price=price_pl,load=load_pl,generation=gen_pl,fuels=fuel_pl,)
repo_root = Path.cwd()
train_df = build_training_dataset(merged=merged_pl,mapcode=DEFAULT_MAPCODES[0],save_csv=repo_root / "Datasets" / f"{DEFAULT_MAPCODES[0]}_raw.csv")
train_df = train_df.drop(columns=["Load_A"])
train_df["Renewables_DA_Forecast"] = train_df["WindOn_DA"] + train_df["Solar_DA"]
train_df = train_df.drop(columns=["WindOn_DA","Solar_DA"])
order = ["time_utc","Price","Load_DA","Renewables_DA_Forecast","EUA","Coal","NGas","Oil"]
train_df = train_df[order]
train_df.to_csv(repo_root/"Datasets"/f"{DEFAULT_MAPCODES[0]}.csv",index=False)


