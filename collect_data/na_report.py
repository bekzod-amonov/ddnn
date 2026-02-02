def missing_hours_report(
    df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    column: str = "Price",          # choose which column to analyze
    start: str | None = None,       # optional; if None, inferred from df[time_col].min()
    end: str | None = None,         # optional; if None, inferred from df[time_col].max()
    freq: str = "H",
    print_top: int = 40,
) -> dict:
    """
    Returns a dict with:
      - expected_hours, observed_hours, missing_hours_count, extra_hours_count
      - missing_timestamps_df: DataFrame of missing timestamps (and day/hour breakdown)
      - missing_by_day: Series (#missing hours per day)
      - missing_by_hour: Series (#missing by hour-of-day)
      - na_summary: (na_count, na_pct) for the chosen column
    """

    out = {}
    d = df.copy()

    # --- parse / sanitize time ---
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    d = d.sort_values(time_col)

    # If you have duplicates in time_utc, keep them but completeness check uses unique timestamps
    observed_ts = pd.DatetimeIndex(d[time_col].unique()).sort_values()

    # --- define expected range ---
    if start is None:
        start_ts = observed_ts.min()
    else:
        start_ts = pd.to_datetime(start)

    if end is None:
        end_ts = observed_ts.max()
    else:
        end_ts = pd.to_datetime(end)

    # Expected hourly grid (inclusive endpoints)
    expected_ts = pd.date_range(start=start_ts, end=end_ts, freq=freq)

    # --- completeness checks ---
    missing_ts = expected_ts.difference(observed_ts)
    extra_ts   = observed_ts.difference(expected_ts)  # timestamps outside expected range

    out["expected_hours"] = len(expected_ts)
    out["observed_hours"] = len(observed_ts)
    out["missing_hours_count"] = len(missing_ts)
    out["extra_hours_count"] = len(extra_ts)

    print("=== COMPLETENESS CHECK (timestamps) ===")
    print(f"expected hours: {out['expected_hours']:,}")
    print(f"observed unique hours: {out['observed_hours']:,}")
    print(f"missing hours vs expected: {out['missing_hours_count']:,}")
    print(f"extra hours outside expected: {out['extra_hours_count']:,}")
    print(f"range: {expected_ts.min()}  ->  {expected_ts.max()}")

    # --- NA summary for chosen column ---
    if column not in d.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(d.columns)}")

    na_count = d[column].isna().sum()
    na_pct = na_count / len(d) * 100
    out["na_summary"] = {"na_count": int(na_count), "na_pct": float(na_pct)}

    print(f"\n=== NA SUMMARY for '{column}' (rows) ===")
    print(f"rows: {len(d):,} | {column} NA rows: {na_count:,} ({na_pct:.3f}%)")

    # --- missingness as timestamps for chosen column (within observed timestamps) ---
    # This is different from "missing timestamps": these timestamps exist, but the value is NA.
    na_ts = pd.DatetimeIndex(d.loc[d[column].isna(), time_col].unique()).sort_values()

    print(f"\n=== '{column}' NA as timestamps (value missing, timestamp present) ===")
    print(f"unique hours where {column} is NA: {len(na_ts):,}")

    # Build a nice table for the NA timestamps (value missing)
    na_ts_df = pd.DataFrame({time_col: na_ts})
    na_ts_df["date"] = na_ts_df[time_col].dt.date
    na_ts_df["hour"] = na_ts_df[time_col].dt.hour
    out["na_timestamps_df"] = na_ts_df

    missing_ts_df = pd.DataFrame({time_col: missing_ts})
    missing_ts_df["date"] = missing_ts_df[time_col].dt.date
    missing_ts_df["hour"] = missing_ts_df[time_col].dt.hour
    out["missing_timestamps_df"] = missing_ts_df

    # Breakdown: missing timestamps per day / hour (timestamp missing from dataset)
    out["missing_by_day"] = missing_ts_df.groupby("date")[time_col].size().sort_values(ascending=False)
    out["missing_by_hour"] = missing_ts_df.groupby("hour")[time_col].size().sort_values(ascending=False)

    print("\n=== MISSING TIMESTAMPS (rows absent) breakdown ===")
    print("missing hours per day (top 20):")
    print(out["missing_by_day"].head(20).to_string())
    print("\nmissing by hour-of-day (0-23):")
    print(out["missing_by_hour"].sort_index().to_string())

    # Show some examples
    if len(missing_ts_df) > 0:
        print(f"\nFirst {min(print_top, len(missing_ts_df))} missing timestamps:")
        print(missing_ts_df.head(print_top).to_string(index=False))

    # Breakdown: NA-values per day/hour (timestamp present but value missing)
    out["na_by_day"] = na_ts_df.groupby("date")[time_col].size().sort_values(ascending=False)
    out["na_by_hour"] = na_ts_df.groupby("hour")[time_col].size().sort_values(ascending=False)

    print("\n=== NA VALUE (timestamp present) breakdown ===")
    print("NA-value hours per day (top 20):")
    print(out["na_by_day"].head(20).to_string())
    print("\nNA-values by hour-of-day (0-23):")
    print(out["na_by_hour"].sort_index().to_string())

    if len(na_ts_df) > 0:
        print(f"\nFirst {min(print_top, len(na_ts_df))} NA-value timestamps:")
        print(na_ts_df.head(print_top).to_string(index=False))

    return out
