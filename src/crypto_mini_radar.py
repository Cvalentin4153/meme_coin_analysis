#!/usr/bin/env python3
"""
End-to-end: fetch 15-day history for a set of CoinGecko coin IDs,
normalize to a tidy table, compute KPIs, and rank by a composite score.

Inputs:
  - data/processed/candidates.csv   (must contain: id, symbol, name, market_cap, total_volume)

Outputs (created/overwritten on each run):
  - data/raw/market_chart/<YYYY-MM-DD>/market_chart_<coin_id>.json   (raw API snapshots)
  - data/processed/daily_prices.parquet
  - data/processed/kpis.parquet
  - data/processed/ranked.parquet
"""

# ---------- imports ----------
from __future__ import annotations  # enable forward type hints in older Python versions
import os                           # filesystem utilities (environment, paths)
import json                         # write raw API responses to disk
import time                         # sleep for backoff on errors
from datetime import date           # build today's folder name like '2025-08-17'
from pathlib import Path            # easy, OS-safe file paths
from typing import Dict, List, Tuple

import requests                     # HTTP client for the CoinGecko API
import numpy as np                  # math (z-scores, epsilons)
import pandas as pd                 # dataframes for cleaning and KPIs

# ---------- constants & paths ----------
BASE = "https://api.coingecko.com/api/v3"  # CoinGecko base URL
TODAY = date.today().isoformat()           # 'YYYY-MM-DD' string for today
RAW_DIR = Path("data/raw/market_chart") / TODAY   # dated folder for raw JSONs
PROCESSED_DIR = Path("data/processed")            # folder for tidy outputs
CANDIDATES_CSV = PROCESSED_DIR / "candidates.csv" # your 50 tokens (already filtered < $10M)
DAILY_PRICES_PARQUET = PROCESSED_DIR / "daily_prices.parquet"  # tidy per-day rows
KPIS_PARQUET = PROCESSED_DIR / "kpis.parquet"                  # one row per token (KPIs)
RANKED_PARQUET = PROCESSED_DIR / "ranked.parquet"              # ranked tokens
HEADERS = {"User-Agent": "crypto-mini-radar/0.1"}              # polite UA header

# Ensure folders exist before writing anything
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------- tiny helpers ----------
def read_candidates(path: Path) -> pd.DataFrame:
    """Load your 50-token list and do minimal validation."""
    df = pd.read_csv(path)                                     # read CSV into a DataFrame
    required = {"id", "symbol", "name"}                        # minimal set needed going forward
    missing = required - set(df.columns)                       # which columns are missing?
    if missing:                                                # if any are missing,
        raise ValueError(f"Candidates CSV missing columns: {missing}")  # fail early with a clear message
    # Drop duplicates by 'id' so we don't fetch the same coin twice
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def get_json(url: str, params: Dict, max_retries: int = 5, backoff_secs: float = 2.0) -> Dict:
    """
    GET a JSON endpoint with exponential backoff and rate limit handling.
    - Retries on HTTP 429 / 5xx (rate limit or transient errors).
    - Special handling for 429 with longer waits.
    - Raises on other errors so you see the problem.
    """
    for attempt in range(1, max_retries + 1):                  # 1..max_retries
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)  # send request
        if resp.status_code < 400:                             # success path (2xx/3xx)
            return resp.json()                                 # parse and return JSON body
        
        if resp.status_code == 429:                            # rate limit hit
            # For rate limits, wait longer (CoinGecko has 30 calls/minute = 2s between calls minimum)
            wait_time = max(backoff_secs, 60)                  # wait at least 60 seconds for rate limit
            print(f"[warn] Rate limit hit (attempt {attempt}/{max_retries}), waiting {wait_time}s...")
            time.sleep(wait_time)
            backoff_secs *= 2                                  # increase for next retry
            continue
        elif resp.status_code in (500, 502, 503, 504):         # server errors
            print(f"[warn] Server error {resp.status_code} (attempt {attempt}/{max_retries}), waiting {backoff_secs}s...")
            time.sleep(backoff_secs)                           # wait a bit before retrying
            backoff_secs *= 2                                  # exponential backoff
            continue                                           # try again
        
        # Non-retryable error (e.g., 404 invalid coin); raise with snippet of body
        raise RuntimeError(f"GET {url} failed [{resp.status_code}]: {resp.text[:200]}")
    
    # If we exhausted retries, raise with the last response status we saw
    raise RuntimeError(f"GET {url} failed after {max_retries} retries")


def fetch_market_chart_15d(coin_id: str) -> Dict:
    """
    Fetch last-15-day daily 'prices' and 'total_volumes' for a given coin ID.
    Also writes the raw JSON to disk for auditability.
    """
    url = f"{BASE}/coins/{coin_id}/market_chart"                # endpoint for market chart
    params = {"vs_currency": "usd", "days": "15", "interval": "daily"}  # request 15 daily points
    payload = get_json(url, params)                             # perform the HTTP GET with retries

    # Defensive: ensure the structure we expect is present and non-empty
    if not isinstance(payload, dict) or "prices" not in payload or "total_volumes" not in payload:
        raise ValueError(f"Unexpected payload shape for {coin_id}")
    if not payload["prices"] or not payload["total_volumes"]:
        raise ValueError(f"No data in market chart for {coin_id}")

    # Write a raw snapshot for transparency/reproducibility
    raw_path = RAW_DIR / f"market_chart_{coin_id}.json"         # e.g., data/raw/market_chart/2025-08-17/market_chart_pepe.json
    with raw_path.open("w", encoding="utf-8") as f:             # open the file for writing text
        json.dump(payload, f, ensure_ascii=False)               # dump JSON exactly as received (utf-8)

    return payload                                              # return the dict for immediate normalization


def normalize_market_chart(coin_id: str, payload: Dict) -> pd.DataFrame:
    """
    Convert one coin's raw payload into a tidy daily table:
      date, coin_id, close_price_usd, volume_usd
    """
    # ---- prices → daily close ----
    prices = pd.DataFrame(payload["prices"], columns=["ts_ms", "price"])      # make DF with 2 named columns
    prices["date"] = pd.to_datetime(prices["ts_ms"], unit="ms", utc=True).dt.date  # convert ms epoch to UTC date (no time)
    # Keep the *last* price per date (some payloads may have more granularity)
    prices = prices.drop(columns=["ts_ms"]).drop_duplicates(subset=["date"], keep="last")
    prices = prices.rename(columns={"price": "close_price_usd"})               # rename for clarity

    # ---- volumes → daily volume ----
    vols = pd.DataFrame(payload["total_volumes"], columns=["ts_ms", "volume"]) # similar steps for volume
    vols["date"] = pd.to_datetime(vols["ts_ms"], unit="ms", utc=True).dt.date  # align to UTC date
    vols = vols.drop(columns=["ts_ms"]).drop_duplicates(subset=["date"], keep="last")
    vols = vols.rename(columns={"volume": "volume_usd"})                       # rename for clarity

    # ---- join prices & volumes on date ----
    df = pd.merge(prices, vols, on="date", how="inner")                        # inner join ensures we only keep days present in both
    df.insert(1, "coin_id", coin_id)                                          # add a coin_id column at position 1 for grouping
    df = df[["date", "coin_id", "close_price_usd", "volume_usd"]]             # enforce exact column order

    # ---- basic quality filters ----
    df = df[(df["close_price_usd"] > 0) & (df["volume_usd"] >= 0)]            # remove invalid rows
    # Expect ~15 dates; allow a couple of gaps. If too few days, return empty to signal exclusion.
    if df["date"].nunique() < 12:                                             # fewer than 12 daily rows out of 15 is suspicious
        return pd.DataFrame(columns=["date", "coin_id", "close_price_usd", "volume_usd"])
    return df


def build_daily_prices(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    For each candidate coin_id, fetch + normalize and stack into one tidy DataFrame.
    Returns (daily_prices_df, failures_list).
    
    Includes rate limiting to respect CoinGecko's 30 calls/minute limit.
    """
    frames: List[pd.DataFrame] = []                                           # collect per-coin DFs here
    failures: List[Tuple[str, str]] = []                                      # collect (coin_id, error) here
    coin_ids = candidates["id"].tolist()                                      # get list of coin IDs
    total_coins = len(coin_ids)
    
    print(f"[info] Fetching data for {total_coins} coins with rate limiting (30 calls/minute max)...")
    
    for i, coin_id in enumerate(coin_ids, 1):                                 # iterate IDs from your CSV with progress
        try:
            print(f"[info] Fetching {coin_id} ({i}/{total_coins})...")
            payload = fetch_market_chart_15d(coin_id)                          # raw JSON
            df_one = normalize_market_chart(coin_id, payload)                  # tidy DF for this coin
            if df_one.empty:                                                   # too few days or invalid data
                print(f"[warn] {coin_id}: insufficient daily data")
                failures.append((coin_id, "insufficient_daily_rows"))
                continue
            frames.append(df_one)                                              # keep this coin's data
            print(f"[success] {coin_id}: got {len(df_one)} daily records")
            
            # Rate limiting: wait between successful calls to respect 30 calls/minute
            # 30 calls/minute = 2 seconds between calls minimum
            if i < total_coins:                                                # don't wait after the last call
                time.sleep(2.1)                                                # slightly more than 2s to be safe
                
        except Exception as e:                                                 # any error: log and continue
            print(f"[error] {coin_id}: {str(e)}")
            failures.append((coin_id, str(e)))
            # Still wait on errors to maintain rate limiting
            if i < total_coins:
                time.sleep(2.1)
            continue
            
    if not frames:                                                             # if *every* coin failed,
        return pd.DataFrame(columns=["date", "coin_id", "close_price_usd", "volume_usd"]), failures
    # stack all coins vertically into one DataFrame
    daily = pd.concat(frames, ignore_index=True)
    # sort for readability and downstream determinism
    daily = daily.sort_values(["coin_id", "date"]).reset_index(drop=True)
    return daily, failures


# ---------- KPI & ranking ----------
def compute_kpis(daily: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    From tidy daily prices, compute:
      - ret_1d (daily returns) → used to derive:
      - mom_7d (7-day momentum)
      - vol_7d (std dev of last 7 returns)
      - median_vol_15d (median daily volume over the 15-day window)
      - pct_positive_days (share of positive daily returns)
    Returns one row per coin_id with symbol/name merged in.
    """
    # Ensure types
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])                              # convert back to datetime for window ops

    # Compute daily returns per coin_id (vectorized)
    daily["ret_1d"] = daily.groupby("coin_id")["close_price_usd"].pct_change()  # (p_t / p_{t-1}) - 1

    # 7-day momentum: (close_T / close_{T-7}) - 1
    # We'll use a 7-day shift within each group to align prices
    daily["close_shift_7"] = daily.groupby("coin_id")["close_price_usd"].shift(7)
    
    # For each coin_id, we want the *last available* values to summarize the 15-day window
    last_rows = daily.groupby("coin_id", as_index=False).tail(1)               # last row per coin (kept for last_date if needed)

    # Grab the last row per coin and compute mom_7d from that alignment
    mom = last_rows[["coin_id", "close_price_usd", "close_shift_7"]].copy()
    mom["mom_7d"] = (mom["close_price_usd"] / mom["close_shift_7"]) - 1        # may be NaN if fewer than 8 prices
    mom = mom[["coin_id", "mom_7d"]]                                           # keep only needed columns

    # 7-day volatility: stddev of the last 7 daily returns
    # Use rolling(window=7).std() within each coin_id and then take the last value
    daily["vol_7d_series"] = daily.groupby("coin_id")["ret_1d"].rolling(window=7).std(ddof=0).reset_index(level=0, drop=True)
    vol = daily.groupby("coin_id", as_index=False).tail(1)[["coin_id", "vol_7d_series"]].rename(columns={"vol_7d_series": "vol_7d"})

    # Median 15-day volume per coin
    vol_median = daily.groupby("coin_id", as_index=False)["volume_usd"].median().rename(columns={"volume_usd": "median_vol_15d"})

    # % positive days across the 15-day window
    pos_share = (
        daily.assign(pos=(daily["ret_1d"] > 0))                                # boolean per row
             .groupby("coin_id", as_index=False)["pos"].mean()                 # mean of booleans == share True
             .rename(columns={"pos": "pct_positive_days"})
    )

    # Merge all KPI pieces together
    kpis = mom.merge(vol, on="coin_id", how="outer").merge(vol_median, on="coin_id", how="outer").merge(pos_share, on="coin_id", how="outer")

    # Add latest date per coin for reference
    kpis = kpis.merge(last_rows[["coin_id", "date"]].rename(columns={"date": "last_date"}), on="coin_id", how="left")

    # Merge human-friendly columns (symbol, name) from candidates meta
    meta_small = meta[["id", "symbol", "name"]].rename(columns={"id": "coin_id"})
    kpis = kpis.merge(meta_small, on="coin_id", how="left")

    # Order columns for readability
    kpis = kpis[[
        "coin_id", "symbol", "name", "last_date",
        "mom_7d", "vol_7d", "median_vol_15d", "pct_positive_days"
    ]]

    return kpis


def zscore(series: pd.Series) -> pd.Series:
    """Standard z-score with tiny epsilon to avoid division-by-zero."""
    s = series.astype(float)                                                  # ensure numeric
    mu = s.mean(skipna=True)                                                  # mean of available values
    sigma = s.std(skipna=True, ddof=0)                                        # population std dev
    if sigma == 0 or np.isnan(sigma):                                         # guard against constant or all-NaN
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)        # z=0 for all
    return (s - mu) / (sigma + 1e-9)                                          # z = (x - μ) / (σ + ε)


def rank_tokens(kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Build a composite 'potential_score' and rank descending.
    Score weights:
      0.40 * z(mom_7d) + 0.30 * z(median_vol_15d) + 0.20 * z(pct_positive_days) + 0.10 * (-z(vol_7d))
    """
    df = kpis.copy()

    # Exclude coins missing core features (documenting this rule makes your pipeline robust & explainable)
    df = df[~df["mom_7d"].isna() & ~df["median_vol_15d"].isna()]              # must have momentum & liquidity proxy

    # Compute z-scores per feature
    z_mom  = zscore(df["mom_7d"])
    z_liq  = zscore(df["median_vol_15d"])
    z_cons = zscore(df["pct_positive_days"])
    z_vol  = zscore(df["vol_7d"])

    # Composite score (lower volatility is better → negative weight)
    df["potential_score"] = 0.40 * z_mom + 0.30 * z_liq + 0.20 * z_cons + 0.10 * (-z_vol)

    # Rank descending (1 = best)
    df = df.sort_values("potential_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)                                        # human-friendly rank starting at 1

    # Final column order
    df = df[[
        "rank", "coin_id", "symbol", "name",
        "mom_7d", "vol_7d", "median_vol_15d", "pct_positive_days",
        "potential_score"
    ]]
    return df


# ---------- main entry ----------
def main() -> None:
    """Orchestrate the full run: read candidates → fetch+normalize → KPIs → rank."""
    # 1) Load candidate coins (your 50 small-caps)
    candidates = read_candidates(CANDIDATES_CSV)
    print(f"[info] candidates loaded: {len(candidates)} rows")

    # 2) Fetch & normalize 15-day history per coin
    daily, failures = build_daily_prices(candidates)
    print(f"[info] daily rows: {len(daily)} | coins kept: {daily['coin_id'].nunique()} | failures: {len(failures)}")

    # 3) Persist daily prices (Parquet = fast & typed)
    daily.to_parquet(DAILY_PRICES_PARQUET, index=False)
    print(f"[info] wrote {DAILY_PRICES_PARQUET}")

    # 4) Save failures, if any, for your run log
    if failures:
        fail_df = pd.DataFrame(failures, columns=["coin_id", "error"])
        fail_path = PROCESSED_DIR / "fetch_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"[warn] wrote failures to {fail_path}")

    # 5) Compute KPIs
    kpis = compute_kpis(daily, candidates)
    kpis.to_parquet(KPIS_PARQUET, index=False)
    print(f"[info] wrote {KPIS_PARQUET}")

    # 6) Rank by potential score
    ranked = rank_tokens(kpis)
    ranked.to_parquet(RANKED_PARQUET, index=False)
    print(f"[info] wrote {RANKED_PARQUET}")

    # 7) Tiny console summary you can paste into your report
    top3 = ranked.head(3)[["rank", "symbol", "name", "potential_score"]]
    print("\n[summary] Top 3 by potential_score:")
    print(top3.to_string(index=False))


if __name__ == "__main__":   # only runs when you execute this file directly
    main()                   # kick off the pipeline
