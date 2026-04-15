"""Feature engineering for LightGBM ML strategy — BLUEPRINT sections 4, 5, 6.

ZERO lookahead bias: all features use shift(k>=1). Target uses shift(-1) (future,
only for training labels — never used as a feature).

Target semantics: 1 if the NEXT candle closes at or above its own open
(close[i+1] >= open[i+1]), matching Polymarket's settlement logic
(resolver.py: winner = "Up" if close_price >= open_price else "Down").

32 features total: candle shape (7), volume (2), 15m context (3), 1h context (3),
funding (2), OHLCV pressure (5), time-of-day cyclical (4), volatility regime (2),
momentum (4: rsi14, candle_streak, price_in_range, ema_cross_5m).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column order — MUST match exactly (32 features)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "body_ratio_n1", "body_ratio_n2", "body_ratio_n3",
    "upper_wick_n1", "upper_wick_n2",
    "lower_wick_n1", "lower_wick_n2",
    "volume_ratio_n1", "volume_ratio_n2",
    "body_ratio_15m", "dir_15m", "volume_ratio_15m",
    "body_ratio_1h", "dir_1h", "ema9_slope_1h",
    "funding_rate", "funding_zscore",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "vol_zscore", "vol_trend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",  # cyclical time (replaces hour_utc, dow)
    "atr_percentile_24h", "vol_regime",
    "rsi14", "candle_streak", "price_in_range", "ema_cross_5m",  # momentum features
]


def compute_atr14(df: pd.DataFrame) -> pd.Series:
    """ATR14 using EWM (BLUEPRINT spec)."""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()


def _asof_backward(left_ts: pd.Series, right: pd.DataFrame, right_cols: list[str]) -> pd.DataFrame:
    """
    Backward-fill lookup: for each timestamp in left_ts, find the last row in
    right where right['timestamp'] <= left_ts.  Uses pd.merge_asof (vectorized,
    C-level) instead of a Python row loop — identical semantics, ~100x faster.

    left_ts  : Series of tz-aware timestamps (may contain NaT), any name.
    right    : sorted DataFrame with a 'timestamp' column + right_cols.
    Returns  : DataFrame indexed 0..len(left_ts)-1 with right_cols,
               NaN where no prior right row exists or left_ts is NaT.
    """
    n = len(left_ts)

    # Build a left frame with a positional index column so we can reindex after
    # the merge.  Give the key a unique name to avoid collisions with `right`.
    left_df = pd.DataFrame({"_left_ts": left_ts.values, "_pos": np.arange(n)})

    # Ensure both key columns share the exact same dtype before merge_asof.
    # Localize if tz-naive, then cast to datetime64[ms, UTC].
    col = left_df["_left_ts"]
    if col.dt.tz is None:
        col = col.dt.tz_localize("UTC")
    left_df["_left_ts"] = col.astype("datetime64[ms, UTC]")
    right = right.copy()
    ts_col = right["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    right["timestamp"] = ts_col.astype("datetime64[ms, UTC]")

    # pd.merge_asof refuses NaT in the left key (raises ValueError).
    # ts_n1 = df5["timestamp"].shift(1) always produces NaT at row 0.
    # Solution: filter those rows out, merge the valid subset, then reindex
    # back to the full 0..n-1 range — NaT positions stay NaN in output.
    valid_mask = left_df["_left_ts"].notna()
    left_valid = left_df[valid_mask].reset_index(drop=True)

    if left_valid.empty:
        # All rows were NaT — return all-NaN frame of correct shape.
        return pd.DataFrame(np.nan, index=np.arange(n), columns=right_cols)

    merged = pd.merge_asof(
        left_valid,
        right[["timestamp"] + right_cols],
        left_on="_left_ts",
        right_on="timestamp",
        direction="backward",
    )

    # Restore original positions: set _pos as index, reindex to 0..n-1.
    # Rows that were NaT (excluded above) will have NaN filled automatically.
    result = (
        merged[["_pos"] + right_cols]
        .set_index("_pos")
        .reindex(np.arange(n))
    )
    return result.reset_index(drop=True)


def build_features(
    df5: pd.DataFrame,
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    funding: pd.DataFrame,
) -> pd.DataFrame:
    """Build 32 features per BLUEPRINT sections 4-6. Returns df with FEATURE_COLS + 'target'."""

    # Work on copies with clean RangeIndex
    df5 = df5.copy().reset_index(drop=True)
    df15 = df15.copy().reset_index(drop=True)
    df1h = df1h.copy().reset_index(drop=True)
    funding = funding.copy().reset_index(drop=True)

    # Sort ascending (should already be sorted, but be safe)
    df5 = df5.sort_values("timestamp").reset_index(drop=True)
    df15 = df15.sort_values("timestamp").reset_index(drop=True)
    df1h = df1h.sort_values("timestamp").reset_index(drop=True)
    funding = funding.sort_values("timestamp").reset_index(drop=True)

    # Normalize all timestamps to ms UTC for consistent merging
    for df in [df5, df15, df1h, funding]:
        df["timestamp"] = df["timestamp"].astype("datetime64[ms, UTC]")

    # -----------------------------------------------------------------------
    # 5m features — all use shift(k>=1), NEVER shift(0)
    # -----------------------------------------------------------------------
    atr5 = compute_atr14(df5)

    df5["body_ratio_n1"] = (df5["close"].shift(1) - df5["open"].shift(1)) / atr5.shift(1)
    df5["body_ratio_n2"] = (df5["close"].shift(2) - df5["open"].shift(2)) / atr5.shift(2)
    df5["body_ratio_n3"] = (df5["close"].shift(3) - df5["open"].shift(3)) / atr5.shift(3)

    df5["upper_wick_n1"] = (
        df5["high"].shift(1) - df5[["open", "close"]].shift(1).max(axis=1)
    ) / atr5.shift(1)
    df5["upper_wick_n2"] = (
        df5["high"].shift(2) - df5[["open", "close"]].shift(2).max(axis=1)
    ) / atr5.shift(2)

    df5["lower_wick_n1"] = (
        df5[["open", "close"]].shift(1).min(axis=1) - df5["low"].shift(1)
    ) / atr5.shift(1)
    df5["lower_wick_n2"] = (
        df5[["open", "close"]].shift(2).min(axis=1) - df5["low"].shift(2)
    ) / atr5.shift(2)

    # volume_ratio_n1: N-1 volume divided by rolling mean of the 20 candles
    # ending at N-2 (i.e. vol[i-2]..vol[i-21]).
    # shift(2).rolling(20) at row i = mean of vol[i-2]..vol[i-21] — N-1 candle
    # is deliberately excluded from its own mean, matching the live formula
    # vol_series[-22:-2] and the blueprint Section 5 English spec.
    vol_mean_n1 = df5["volume"].shift(2).rolling(20).mean()
    df5["volume_ratio_n1"] = df5["volume"].shift(1) / vol_mean_n1
    # volume_ratio_n2: N-2 volume divided by rolling mean of vol[i-3]..vol[i-22]
    vol_mean_n2 = df5["volume"].shift(3).rolling(20).mean()
    df5["volume_ratio_n2"] = df5["volume"].shift(2) / vol_mean_n2

    # ts_n1 = N-1 timestamp (shift by 1 for all multi-tf merges)
    ts_n1 = df5["timestamp"].shift(1)

    # -----------------------------------------------------------------------
    # 15m features — merge_asof backward on ts_n1
    # -----------------------------------------------------------------------
    atr15 = compute_atr14(df15)
    df15["body_ratio_15m"] = (df15["close"] - df15["open"]) / atr15
    df15["dir_15m"] = np.sign(df15["close"] - df15["open"])
    df15["volume_ratio_15m"] = df15["volume"] / df15["volume"].rolling(20, min_periods=2).mean()

    r15 = _asof_backward(ts_n1, df15, ["body_ratio_15m", "dir_15m", "volume_ratio_15m"])
    df5["body_ratio_15m"] = r15["body_ratio_15m"].values
    df5["dir_15m"] = r15["dir_15m"].values
    df5["volume_ratio_15m"] = r15["volume_ratio_15m"].values

    # -----------------------------------------------------------------------
    # 1h features — same merge_asof pattern
    # -----------------------------------------------------------------------
    atr1h = compute_atr14(df1h)
    df1h["body_ratio_1h"] = (df1h["close"] - df1h["open"]) / atr1h
    df1h["dir_1h"] = np.sign(df1h["close"] - df1h["open"])
    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
    df1h["ema9_slope_1h"] = (ema9 - ema9.shift(1)) / atr1h

    r1h = _asof_backward(ts_n1, df1h, ["body_ratio_1h", "dir_1h", "ema9_slope_1h"])
    df5["body_ratio_1h"] = r1h["body_ratio_1h"].values
    df5["dir_1h"] = r1h["dir_1h"].values
    df5["ema9_slope_1h"] = r1h["ema9_slope_1h"].values

    # -----------------------------------------------------------------------
    # Funding features
    # -----------------------------------------------------------------------
    funding["funding_zscore"] = (
        funding["funding_rate"] - funding["funding_rate"].rolling(24, min_periods=2).mean()
    ) / funding["funding_rate"].rolling(24, min_periods=2).std()
    funding.loc[funding["funding_rate"].rolling(24, min_periods=2).std() == 0, "funding_zscore"] = np.nan

    rf = _asof_backward(ts_n1, funding, ["funding_rate", "funding_zscore"])
    df5["funding_rate"] = rf["funding_rate"].values
    df5["funding_zscore"] = rf["funding_zscore"].values

    # -----------------------------------------------------------------------
    # OHLCV-native pressure features — computed purely from df5, zero parity gap
    # -----------------------------------------------------------------------
    hl_range = (df5["high"] - df5["low"]).clip(lower=1e-9)
    body      = df5["close"] - df5["open"]

    # body_ratio: candle body direction and strength, [-1, 1]
    df5["body_ratio"] = (body / hl_range).clip(-1.0, 1.0).shift(1)

    # upper_wick_ratio: selling rejection at highs, [0, 1]
    upper_wick = df5["high"] - df5[["open", "close"]].max(axis=1)
    df5["upper_wick_ratio"] = (upper_wick / hl_range).clip(0.0, 1.0).shift(1)

    # lower_wick_ratio: buying rejection at lows, [0, 1]
    lower_wick = df5[["open", "close"]].min(axis=1) - df5["low"]
    df5["lower_wick_ratio"] = (lower_wick / hl_range).clip(0.0, 1.0).shift(1)

    # vol_zscore: volume surge detection vs 20-bar rolling mean/std
    vol_mean20 = df5["volume"].rolling(20).mean()
    vol_std20  = df5["volume"].rolling(20).std(ddof=1).clip(lower=1e-8)
    df5["vol_zscore"] = ((df5["volume"] - vol_mean20) / vol_std20).shift(1)

    # vol_trend: short vs long volume momentum (5-bar / 20-bar rolling mean)
    vol_ma5  = df5["volume"].rolling(5).mean()
    vol_ma20 = df5["volume"].rolling(20).mean().clip(lower=1e-8)
    df5["vol_trend"] = (vol_ma5 / vol_ma20).shift(1)

    # -----------------------------------------------------------------------
    # Time-of-day cyclical features — derived from N-1 candle timestamp
    # Replaces raw hour_utc and dow with sine/cosine encoding so the model
    # can learn periodic patterns without discontinuities at midnight / week-end.
    # -----------------------------------------------------------------------
    ts_n1_series = df5["timestamp"].shift(1)
    hour_raw = ts_n1_series.dt.hour
    dow_raw = ts_n1_series.dt.dayofweek
    df5["hour_sin"] = np.sin(2 * np.pi * hour_raw / 24)
    df5["hour_cos"] = np.cos(2 * np.pi * hour_raw / 24)
    df5["dow_sin"]  = np.sin(2 * np.pi * dow_raw / 7)
    df5["dow_cos"]  = np.cos(2 * np.pi * dow_raw / 7)

    # Volatility regime features — derived from ATR of the N-1 candle
    # atr_percentile_24h: percentile rank (0.0–1.0) of atr5[i-1] within a 288-candle rolling window
    # vol_regime: zscore of atr5[i-1] within same 288-candle rolling window (std-normalized)
    # 288 = 24 hours * 12 five-minute candles per hour
    _ATR_WINDOW = 288
    atr_shifted = atr5.shift(1)
    def _rolling_percentile(s: pd.Series, w: int) -> pd.Series:
        # min_periods=14 allows partial windows during ATR warmup rows;
        # NaNs within the window are stripped before ranking so warmup NaNs
        # don't silently corrupt the percentile (NaN < value == False in numpy).
        def _pct(x: np.ndarray) -> float:
            x = x[~np.isnan(x)]
            if len(x) < 2:
                return np.nan
            return float(np.sum(x[:-1] < x[-1])) / max(len(x) - 1, 1)
        return s.rolling(w, min_periods=14).apply(_pct, raw=True)
    df5["atr_percentile_24h"] = _rolling_percentile(atr_shifted, _ATR_WINDOW)
    roll = atr_shifted.rolling(_ATR_WINDOW, min_periods=14)
    atr_roll_mean = roll.mean()
    atr_roll_std  = roll.std()
    df5["vol_regime"] = (atr_shifted - atr_roll_mean) / atr_roll_std.clip(lower=1e-10)

    # -----------------------------------------------------------------------
    # Momentum features (new) — all use shift(k>=1) for zero lookahead
    # -----------------------------------------------------------------------

    # rsi14: Wilder's RSI(14) on 5m closes, N-1 value
    _delta = df5["close"].diff()
    _gain = _delta.clip(lower=0)
    _loss = (-_delta).clip(lower=0)
    _avg_gain = _gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _avg_loss = _loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _rs = _avg_gain / _avg_loss.clip(lower=1e-10)
    _rsi = 100.0 - (100.0 / (1.0 + _rs))
    df5["rsi14"] = _rsi.shift(1)  # N-1 value, zero lookahead

    # candle_streak: consecutive same-direction candles ending at N-1
    # Vectorized approach: group consecutive same-direction runs
    _direction = np.sign(df5["close"] - df5["open"])
    _same_as_prev = (_direction == _direction.shift(1)) & (_direction != 0)
    _streak = _same_as_prev.groupby((~_same_as_prev).cumsum()).cumsum()
    _streak = _streak.where(_direction != 0, 0).astype(float)
    df5["candle_streak"] = _streak.shift(1)  # N-1 value

    # price_in_range: where N-1 close sits within 20-candle range ending at N-1
    # rolling(20).max/min on shift(1) gives range of [i-20..i-1] — zero lookahead
    _rolling_high = df5["high"].shift(1).rolling(20, min_periods=5).max()
    _rolling_low  = df5["low"].shift(1).rolling(20, min_periods=5).min()
    _rng = (_rolling_high - _rolling_low).clip(lower=1e-10)
    df5["price_in_range"] = (df5["close"].shift(1) - _rolling_low) / _rng

    # ema_cross_5m: sign of EMA9 vs EMA21 at N-1 candle (-1, 0, +1)
    _ema9_5m  = df5["close"].ewm(span=9,  adjust=False).mean()
    _ema21_5m = df5["close"].ewm(span=21, adjust=False).mean()
    df5["ema_cross_5m"] = np.sign(_ema9_5m - _ema21_5m).shift(1)  # N-1 cross state

    # -----------------------------------------------------------------------
    # Target: 1 if next candle closes >= its own open (future label, NOT a feature)
    # Matches Polymarket settlement: close >= open within candle i+1
    # (resolver.py: winner = "Up" if close_price >= open_price else "Down")
    # -----------------------------------------------------------------------
    df5["target"] = (df5["close"].shift(-1) >= df5["open"].shift(-1)).astype(int)

    # -----------------------------------------------------------------------
    # Drop rows with any NaN in features or target, return feature cols + target
    # -----------------------------------------------------------------------
    all_cols = FEATURE_COLS + ["target"]
    df_out = df5[all_cols].dropna()
    log.info("build_features: %d rows after dropna (started with %d)", len(df_out), len(df5))
    return df_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live feature computation
# ---------------------------------------------------------------------------

def build_live_features(
    df5_live: pd.DataFrame,
    df15_live: pd.DataFrame,
    df1h_live: pd.DataFrame,
    funding_rate_float: float | None,
    funding_buffer: deque,
) -> "tuple[np.ndarray, list[str]] | tuple[None, list[str]]":
    """
    Build a single feature row (shape 1×32) for live inference.

    Returns a 2-tuple (feature_row, nan_features):
      - feature_row : np.ndarray shape (1, 32), or None on hard failure.
      - nan_features: list of feature names that were NaN (empty on success).
                      Populated even when feature_row is None so callers can
                      log exactly which features caused the skip.

    Returns (None, []) if ATR warmup is not satisfied (fewer than 14 candles).
    Returns (None, [<name>, ...]) when one or more features are NaN.
    Returns (row, []) on full success.
    """
    # Validate ATR warmup
    if len(df5_live) < 14:
        return None, []

    df5 = df5_live.copy().reset_index(drop=True)
    df15 = df15_live.copy().reset_index(drop=True)
    df1h = df1h_live.copy().reset_index(drop=True)

    # Normalize timestamps
    for df in [df5, df15, df1h]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ms, UTC]")

    atr5 = compute_atr14(df5)
    if atr5.iloc[-1] is None or pd.isna(atr5.iloc[-1]):
        return None, []

    # 5m features using last row (index -1 = current candle N)
    # We use shift(1) = N-1
    def safe(series, k=1):
        idx = len(series) - 1 - k
        return series.iloc[idx] if idx >= 0 else np.nan

    atr5_val = safe(atr5, 1)

    body_ratio_n1 = (safe(df5["close"], 1) - safe(df5["open"], 1)) / atr5_val
    body_ratio_n2 = (safe(df5["close"], 2) - safe(df5["open"], 2)) / safe(atr5, 2)
    body_ratio_n3 = (safe(df5["close"], 3) - safe(df5["open"], 3)) / safe(atr5, 3)

    upper_wick_n1 = (safe(df5["high"], 1) - max(safe(df5["open"], 1), safe(df5["close"], 1))) / atr5_val
    upper_wick_n2 = (safe(df5["high"], 2) - max(safe(df5["open"], 2), safe(df5["close"], 2))) / safe(atr5, 2)
    lower_wick_n1 = (min(safe(df5["open"], 1), safe(df5["close"], 1)) - safe(df5["low"], 1)) / atr5_val
    lower_wick_n2 = (min(safe(df5["open"], 2), safe(df5["close"], 2)) - safe(df5["low"], 2)) / safe(atr5, 2)

    vol_series = df5["volume"].values
    # volume_ratio_n1: N-1 volume divided by rolling mean of 20 candles ending at N-2
    # Matches training: vol_mean = df['volume'].shift(2).rolling(20).mean()
    # which at row i gives mean of vol[i-2]..vol[i-21] — N-1 candle excluded from its own mean.
    # In the live array (last index = N, second-to-last = N-1):
    #   N-1 candle value : vol_series[-2]
    #   Mean window for N-1: vol_series[-22:-2]  (up to and excluding N-1)
    #   N-2 candle value : vol_series[-3]
    #   Mean window for N-2: vol_series[-23:-3]  (up to and excluding N-2)
    if len(vol_series) >= 22:
        vol_ratio_n1 = vol_series[-2] / np.mean(vol_series[-22:-2])
    elif len(vol_series) >= 4:
        # Fewer than 20 prior candles available — use what we have (graceful degradation)
        vol_ratio_n1 = vol_series[-2] / np.mean(vol_series[:-2]) if len(vol_series) > 2 else np.nan
    else:
        vol_ratio_n1 = np.nan

    if len(vol_series) >= 23:
        vol_ratio_n2 = vol_series[-3] / np.mean(vol_series[-23:-3])
    elif len(vol_series) >= 5:
        vol_ratio_n2 = vol_series[-3] / np.mean(vol_series[:-3]) if len(vol_series) > 3 else np.nan
    else:
        vol_ratio_n2 = np.nan

    # 15m features
    if len(df15) >= 14:
        atr15 = compute_atr14(df15)
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            # Find the last 15m candle at or before ts_n1
            mask15 = df15["timestamp"] <= ts_n1
            if mask15.any():
                idx15 = df15[mask15].index[-1]
                atr15_val = atr15.iloc[idx15]
                if pd.notna(atr15_val) and atr15_val > 0:
                    body_ratio_15m = (df15["close"].iloc[idx15] - df15["open"].iloc[idx15]) / atr15_val
                    dir_15m = np.sign(df15["close"].iloc[idx15] - df15["open"].iloc[idx15])
                    # volume_ratio_15m: matches training — rolling(20, min_periods=2).mean()
                    vol15_rolling_mean = df15["volume"].rolling(20, min_periods=2).mean()
                    vol15_mean_val = vol15_rolling_mean.iloc[idx15]
                    vol_ratio_15m = df15["volume"].iloc[idx15] / vol15_mean_val if pd.notna(vol15_mean_val) and vol15_mean_val > 0 else np.nan
                else:
                    body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
            else:
                body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
        else:
            body_ratio_15m = dir_15m = vol_ratio_15m = np.nan
    else:
        body_ratio_15m = dir_15m = vol_ratio_15m = np.nan

    # 1h features
    if len(df1h) >= 14:
        atr1h = compute_atr14(df1h)
        ts_n1 = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
        if ts_n1 is not None and not pd.isna(ts_n1):
            mask1h = df1h["timestamp"] <= ts_n1
            if mask1h.any():
                idx1h = df1h[mask1h].index[-1]
                # Use ATR at idx1h only — do NOT scan forward into future candles.
                # Scanning forward would pull OHLC/EMA data from candles that haven't
                # closed yet at the time of the N-1 5m bar, introducing lookahead bias.
                # If ATR is NaN here (warmup), fall back to NaN for all 1h features.
                atr1h_val = atr1h.iloc[idx1h]
                if pd.notna(atr1h_val) and atr1h_val > 0:
                    body_ratio_1h = (df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h]) / atr1h_val
                    dir_1h = np.sign(df1h["close"].iloc[idx1h] - df1h["open"].iloc[idx1h])
                    ema9 = df1h["close"].ewm(span=9, adjust=False).mean()
                    ema9_slope_1h = (ema9.iloc[idx1h] - ema9.iloc[idx1h - 1]) / atr1h_val if idx1h > 0 else np.nan
                else:
                    body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
            else:
                body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
        else:
            body_ratio_1h = dir_1h = ema9_slope_1h = np.nan
    else:
        body_ratio_1h = dir_1h = ema9_slope_1h = np.nan

    # Funding features
    if funding_rate_float is not None and len(funding_buffer) > 0:
        buf = list(funding_buffer)
        fr = funding_rate_float
        if len(buf) >= 2:
            mean24 = np.mean(buf)
            std24 = np.std(buf, ddof=1) if len(buf) >= 2 else 0.0
            funding_zscore = (fr - mean24) / std24 if std24 > 0 else np.nan
        else:
            funding_zscore = np.nan
    else:
        fr = np.nan
        funding_zscore = np.nan

    # OHLCV-native pressure features (live) — identical formulas to build_features
    # All use N-1 candle (index -2 after the still-forming candle is trimmed by caller)
    hl_range_live = (df5["high"] - df5["low"]).clip(lower=1e-9)
    body_live      = df5["close"] - df5["open"]

    # body_ratio: [-1, 1]
    body_ratio = float(np.clip(body_live.iloc[-2] / hl_range_live.iloc[-2], -1.0, 1.0))

    # upper_wick_ratio: [0, 1]
    upper_wick_live = df5["high"] - df5[["open", "close"]].max(axis=1)
    upper_wick_ratio = float(np.clip(upper_wick_live.iloc[-2] / hl_range_live.iloc[-2], 0.0, 1.0))

    # lower_wick_ratio: [0, 1]
    lower_wick_live = df5[["open", "close"]].min(axis=1) - df5["low"]
    lower_wick_ratio = float(np.clip(lower_wick_live.iloc[-2] / hl_range_live.iloc[-2], 0.0, 1.0))

    # vol_zscore: (vol_n1 - mean20) / std20, window ends at N-1 (index -2)
    if len(df5) >= 21:
        vol_window = df5["volume"].iloc[-21:-1]  # 20 bars ending at N-1 inclusive
        v_mean = float(vol_window.mean())
        v_std  = float(vol_window.std(ddof=1))
        vol_zscore = (float(df5["volume"].iloc[-2]) - v_mean) / max(v_std, 1e-8)
    else:
        vol_zscore = np.nan

    # vol_trend: ma5 / ma20 at N-1
    if len(df5) >= 21:
        vol_ma5_live  = float(df5["volume"].iloc[-6:-1].mean())   # 5 bars ending at N-1
        vol_ma20_live = float(df5["volume"].iloc[-21:-1].mean())  # 20 bars ending at N-1
        vol_trend = vol_ma5_live / max(vol_ma20_live, 1e-8)
    else:
        vol_trend = np.nan

    # -----------------------------------------------------------------------
    # Time-of-day cyclical features — use N-1 candle timestamp (index -2)
    # Replaces raw hour_utc and dow with sin/cos encoding.
    # -----------------------------------------------------------------------
    ts_n1_live = df5["timestamp"].iloc[-2] if len(df5) >= 2 else None
    if ts_n1_live is not None and not pd.isna(ts_n1_live):
        ts = pd.Timestamp(ts_n1_live)
        hour_raw_live = float(ts.hour)
        dow_raw_live  = float(ts.dayofweek)
        hour_sin = float(np.sin(2 * np.pi * hour_raw_live / 24))
        hour_cos = float(np.cos(2 * np.pi * hour_raw_live / 24))
        dow_sin  = float(np.sin(2 * np.pi * dow_raw_live / 7))
        dow_cos  = float(np.cos(2 * np.pi * dow_raw_live / 7))
    else:
        hour_sin = hour_cos = dow_sin = dow_cos = np.nan

    # Volatility regime features — rolling window on atr5 series
    _ATR_WINDOW = 288
    if len(atr5) >= 14:
        atr5_arr = atr5.values  # full series up to and including current candle
        atr_n1 = atr5_arr[-2] if len(atr5_arr) >= 2 else np.nan  # N-1 value
        if pd.notna(atr_n1):
            # Window matches training: rolling(288) at the current row includes atr_n1
            # as the last element — atr5_arr[L-289 .. L-2] in 0-based terms.
            # Slice atr5_arr[max(0, len-289):-1] gives exactly those 288 values
            # (or fewer near the start), with atr_n1 = atr5_arr[-2] as the last entry.
            window_vals = atr5_arr[max(0, len(atr5_arr)-_ATR_WINDOW-1):-1]
            window_vals = window_vals[~np.isnan(window_vals)]  # strip ATR warmup NaNs
            if len(window_vals) >= 14:
                # Rank atr_n1 (last element) against all prior values in the window,
                # matching the training _rolling_percentile logic: x[-1] vs x[:-1].
                atr_percentile_24h = float(np.sum(window_vals[:-1] < atr_n1)) / max(len(window_vals) - 1, 1)
            else:
                atr_percentile_24h = np.nan
            if len(window_vals) >= 2:
                w_mean = float(np.mean(window_vals))
                w_std  = float(np.std(window_vals, ddof=1))
                vol_regime = (atr_n1 - w_mean) / max(w_std, 1e-10)
            else:
                vol_regime = np.nan
        else:
            atr_percentile_24h = np.nan
            vol_regime = np.nan
    else:
        atr_percentile_24h = np.nan
        vol_regime = np.nan

    # -----------------------------------------------------------------------
    # Momentum features (live) — zero lookahead, all use N-1 values
    # -----------------------------------------------------------------------

    # rsi14 (live): Wilder's RSI(14) on 5m closes at N-1
    if len(df5) >= 15:
        delta_live = df5["close"].diff()
        gain_live  = delta_live.clip(lower=0)
        loss_live  = (-delta_live).clip(lower=0)
        avg_gain_live = gain_live.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss_live = loss_live.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs_live  = avg_gain_live / avg_loss_live.clip(lower=1e-10)
        rsi_live = 100.0 - (100.0 / (1.0 + rs_live))
        rsi14 = float(rsi_live.iloc[-2])  # N-1 value
        if np.isnan(rsi14):
            rsi14 = np.nan
    else:
        rsi14 = np.nan

    # candle_streak (live): consecutive same-direction candles BEFORE N-1
    # Training uses _streak.shift(1) at row N-1, which gives the streak count
    # accumulated by candles N-2 and earlier — N-1 itself does NOT contribute.
    # So we use N-1's direction as the reference but only walk back from N-2 onward.
    if len(df5) >= 2:
        dir_live = np.sign(df5["close"].values - df5["open"].values)
        ref_dir = dir_live[-2]  # N-1 direction (reference, not counted)
        streak_val = 0.0
        if ref_dir != 0:
            # Walk backwards from N-2 (index -3) counting same-direction candles
            for k in range(3, len(dir_live) + 1):
                if dir_live[-k] == ref_dir:
                    streak_val += 1.0
                else:
                    break
        candle_streak = streak_val
    else:
        candle_streak = np.nan

    # price_in_range (live): where N-1 close sits within 20-candle range ending at N-1
    if len(df5) >= 6:
        high_arr = df5["high"].values
        low_arr  = df5["low"].values
        close_arr = df5["close"].values
        # N-1 close: index -2
        # 20-candle range ending at N-1 (inclusive): high/low of [-21:-1] or available
        window_hi = high_arr[max(0, len(high_arr)-21):-1]
        window_lo = low_arr[max(0, len(low_arr)-21):-1]
        if len(window_hi) >= 5:
            rng_hi = float(np.max(window_hi))
            rng_lo = float(np.min(window_lo))
            rng = max(rng_hi - rng_lo, 1e-10)
            price_in_range = (close_arr[-2] - rng_lo) / rng
        else:
            price_in_range = np.nan
    else:
        price_in_range = np.nan

    # ema_cross_5m (live): sign of EMA9 vs EMA21 at N-1
    if len(df5) >= 22:
        ema9_live  = df5["close"].ewm(span=9,  adjust=False).mean()
        ema21_live = df5["close"].ewm(span=21, adjust=False).mean()
        ema_cross_5m = float(np.sign(ema9_live.iloc[-2] - ema21_live.iloc[-2]))  # N-1
    else:
        ema_cross_5m = np.nan

    # -----------------------------------------------------------------------
    # Assemble final row — order MUST match FEATURE_COLS exactly (32 features)
    # -----------------------------------------------------------------------
    row = np.array([[
        body_ratio_n1, body_ratio_n2, body_ratio_n3,
        upper_wick_n1, upper_wick_n2,
        lower_wick_n1, lower_wick_n2,
        vol_ratio_n1, vol_ratio_n2,
        body_ratio_15m, dir_15m, vol_ratio_15m,
        body_ratio_1h, dir_1h, ema9_slope_1h,
        fr, funding_zscore,
        body_ratio, upper_wick_ratio, lower_wick_ratio, vol_zscore, vol_trend,
        hour_sin, hour_cos, dow_sin, dow_cos,
        atr_percentile_24h, vol_regime,
        rsi14, candle_streak, price_in_range, ema_cross_5m,
    ]], dtype=np.float64)

    nan_features = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if np.isnan(row[0][i])]
    if nan_features:
        log.warning("build_live_features: NaN in features, skipping inference. NaN features: %s", nan_features)
        return None, nan_features

    return row, []
