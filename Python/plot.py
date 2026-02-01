#!/usr/bin/env python3
"""
Plot ground-truth ("real") vs model "forecast" from a time-indexed table.

Supports:
- CSV input with a datetime index column + columns: real, forecast
- Or running with the embedded example data below

Examples:
  python plot_forecast_vs_real.py --demo
  python plot_forecast_vs_real.py --csv my_table.csv --time-col timestamp
  python plot_forecast_vs_real.py --csv my_table.csv  # assumes first column is datetime index
"""

from __future__ import annotations

import argparse
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt


DEMO_TEXT = """timestamp,real,forecast
2018-12-27 00:00:00,47.41,37.641506
2018-12-27 01:00:00,47.10,35.65876
2018-12-27 02:00:00,46.74,34.85302
2018-12-27 03:00:00,45.03,34.12413
2018-12-27 04:00:00,46.02,34.649971
2018-12-27 05:00:00,49.18,36.974339
2018-12-27 06:00:00,51.03,44.830193
2018-12-27 07:00:00,58.89,53.304893
2018-12-27 08:00:00,64.96,55.267914
2018-12-27 09:00:00,64.61,53.216255
2018-12-27 10:00:00,59.01,51.077
2018-12-27 11:00:00,58.45,49.677734
2018-12-27 12:00:00,66.14,46.453129
2018-12-27 13:00:00,65.00,44.863121
2018-12-27 14:00:00,65.51,44.628971
2018-12-27 15:00:00,68.11,45.692413
2018-12-27 16:00:00,69.22,48.723618
2018-12-27 17:00:00,70.74,53.447651
2018-12-27 18:00:00,71.97,57.786469
2018-12-27 19:00:00,71.25,58.040207
2018-12-27 20:00:00,70.00,53.303856
2018-12-27 21:00:00,62.56,48.583508
2018-12-27 22:00:00,64.62,45.965427
2018-12-27 23:00:00,56.81,40.52586
"""


def load_df(csv_path: str | None, time_col: str | None, demo: bool) -> pd.DataFrame:
    if demo:
        df = pd.read_csv(StringIO(DEMO_TEXT), parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        return df

    if not csv_path:
        raise SystemExit("Provide --csv PATH or use --demo.")

    df = pd.read_csv(csv_path)

    # If user specifies which column is time, use it.
    if time_col:
        if time_col not in df.columns:
            raise SystemExit(f"--time-col '{time_col}' not found in CSV columns: {list(df.columns)}")
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        # Otherwise assume the first column is a datetime index.
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first])
        df = df.set_index(first)

    return df


def metrics(df: pd.DataFrame) -> dict[str, float]:
    # drop rows with missing values just in case
    d = df[["real", "forecast"]].dropna()
    err = d["forecast"] - d["real"]
    mae = err.abs().mean()
    rmse = (err.pow(2).mean()) ** 0.5
    mape = (err.abs() / d["real"].abs().replace(0, pd.NA)).mean() * 100
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": float(mape)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV with datetime index + columns real,forecast")
    ap.add_argument("--time-col", type=str, default=None, help="Name of datetime column (if not first column)")
    ap.add_argument("--out", type=str, default=None, help="Save plot to file (png/pdf/etc). If omitted, shows window.")
    ap.add_argument("--title", type=str, default="Forecast vs Real")
    ap.add_argument("--demo", action="store_true", help="Run using embedded example data")
    args = ap.parse_args()

    df = load_df(args.csv, args.time_col, args.demo)

    required = {"real", "forecast"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns {missing}. Found columns: {list(df.columns)}")

    df = df.sort_index()

    m = metrics(df)

    # --- Plot: time series ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.index, df["real"], label="real")
    ax.plot(df.index, df["forecast"], label="forecast")
    ax.set_title(
        f"{args.title}  |  MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, MAPE={m['MAPE_%']:.2f}%"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {args.out}")
    else:
        plt.show()

    # --- Plot: scatter (forecast vs real) ---
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(df["real"], df["forecast"])
    lo = min(df["real"].min(), df["forecast"].min())
    hi = max(df["real"].max(), df["forecast"].max())
    ax2.plot([lo, hi], [lo, hi])  # y=x reference line
    ax2.set_title("Forecast vs Real (scatter)")
    ax2.set_xlabel("real")
    ax2.set_ylabel("forecast")
    fig2.tight_layout()

    if args.out:
        # save second plot with suffix
        if "." in args.out:
            base, ext = args.out.rsplit(".", 1)
            out2 = f"{base}_scatter.{ext}"
        else:
            out2 = f"{args.out}_scatter.png"
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        print(f"Saved scatter plot to: {out2}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
    
