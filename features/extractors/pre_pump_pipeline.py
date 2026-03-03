from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Dict, Tuple, Optional

# Optional: scikit-learn for PCA + logistic regression
try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except ImportError:
    PCA = None
    LogisticRegression = None


# ============================================================
# 0. BASIC ASSUMPTIONS
# ============================================================
# We assume a trades DataFrame with columns:
#   ts:     pandas.Timestamp (UTC or consistent timezone)
#   symbol: str
#   price:  float
#   volume: float
#   side:   'buy' or 'sell'
#
# Windows are fixed-length (e.g. 60s).
# All feature functions assume the data for each symbol is sorted by ts.


# ============================================================
# 1. WINDOW ASSIGNMENT
# ============================================================

def assign_windows(trades: pd.DataFrame,
                   window: str = "60s") -> pd.DataFrame:
    """
    Assign each trade to a time window.

    Parameters
    ----------
    trades : DataFrame
        columns: ts (datetime64), symbol, price, volume, side
    window : str
        pandas offset alias, e.g. '60s', '1min', '5min'

    Returns
    -------
    DataFrame with added:
        - window_start (Timestamp)
        - window_id (symbol|window_start string)
    """
    df = trades.copy()
    df = df.sort_values(["symbol", "ts"])
    df["window_start"] = df["ts"].dt.floor(window)
    df["window_id"] = (
        df["symbol"].astype(str) + "|" + df["window_start"].astype(str)
    )
    return df


# ============================================================
# 2. WITHIN-WINDOW ADVANCED FEATURES
# ============================================================

def _compute_price_convexity(prices: np.ndarray) -> Dict[str, float]:
    """
    Fit a quadratic p(k) ≈ a k^2 + b k + c over the tick index k and return:
        a  = curvature (convexity)
        r2 = fit quality

    Interpretation:
        - a >> 0 with decent r2 ~ 1: accelerating move
        - a << 0: decelerating / topping behavior
    """
    n = len(prices)
    if n < 3:
        return {"price_convexity_a": 0.0,
                "price_convexity_r2": 0.0}

    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, prices, 2)  # [a, b, c]
    a, b, c = coeffs

    y_hat = a * x**2 + b * x + c
    ss_res = np.sum((prices - y_hat) ** 2)
    ss_tot = np.sum((prices - prices.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"price_convexity_a": float(a), "price_convexity_r2": float(r2)}


def _compute_buy_run_features(bin_signs: np.ndarray) -> Dict[str, float]:
    """
    bin_signs: array of {0,1}, 1 = bin is buy-dominant (buy_vol > sell_vol).

    Metrics:
        max_buy_run_len    = longest streak of consecutive buy-dominant bins
        fraction_buy_bins  = share of bins that are buy-dominant
    """
    if len(bin_signs) == 0:
        return {"max_buy_run_len": 0.0, "fraction_buy_bins": 0.0}

    max_run = 0
    current = 0
    for v in bin_signs:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    fraction = float(bin_signs.sum()) / len(bin_signs)
    return {"max_buy_run_len": float(max_run),
            "fraction_buy_bins": float(fraction)}


def _bin_buy_signs(group: pd.DataFrame,
                   bin_size: str = "5s") -> np.ndarray:
    """
    Downsample trades in a window into sub-bins and mark each sub-bin
    as buy-dominant (1) if buy_vol > sell_vol, else 0.
    """
    if group.empty:
        return np.array([], dtype=int)

    g = group.set_index("ts")
    buy_vol = g.loc[g["side"] == "buy", "volume"].resample(bin_size).sum()
    sell_vol = g.loc[g["side"] == "sell", "volume"].resample(bin_size).sum()

    buy_vol = buy_vol.fillna(0.0)
    sell_vol = sell_vol.fillna(0.0)

    sign = (buy_vol > sell_vol).astype(int).values
    return sign


def compute_window_features(trades: pd.DataFrame,
                            window: str = "60s",
                            buy_run_bin: str = "5s",
                            short_horizon: str = "30s",
                            long_horizon: str = "10min"
                            ) -> pd.DataFrame:
    """
    Core per-window feature computation from raw trades.

    Features per (symbol, window_start):
        - OHLC: p_open, p_close, p_high, p_low
        - total_volume, price_range, price_mean, price_std
        - realized_vol (std of log returns inside window)
        - buy_volume, sell_volume, signed_volume
        - imbalance = signed_vol / total_volume
        - compression_index = total_volume / price_range
        - pressure_ratio = signed_vol / realized_vol
        - return_ratio_short_long (30s vs 10min by default)
        - max_buy_run_len, fraction_buy_bins
        - price_convexity_a, price_convexity_r2
    """
    df = assign_windows(trades, window=window)

    features = []

    for (symbol, w_start), g in df.groupby(["symbol", "window_start"]):
        g = g.sort_values("ts")
        prices = g["price"].values
        vols = g["volume"].values
        sides = g["side"].values

        if len(prices) == 0:
            continue

        p_open = float(prices[0])
        p_close = float(prices[-1])
        p_high = float(prices.max())
        p_low = float(prices.min())

        total_volume = float(vols.sum())
        price_range = float(p_high - p_low)
        price_mean = float(prices.mean())
        price_std = float(prices.std(ddof=1)) if len(prices) > 1 else 0.0

        # realized volatility from log returns
        log_p = np.log(prices)
        if len(log_p) > 1:
            rets = np.diff(log_p)
            realized_vol = float(rets.std(ddof=1))
        else:
            realized_vol = 0.0

        # buy vs sell volumes
        buy_mask = (sides == "buy")
        sell_mask = (sides == "sell")
        buy_vol = float(vols[buy_mask].sum())
        sell_vol = float(vols[sell_mask].sum())
        signed_vol = buy_vol - sell_vol

        imbalance = signed_vol / (total_volume + 1e-12)

        # compression: lots of volume in tight price range
        compression_index = total_volume / (price_range + 1e-12)

        # pressure: signed flow vs realized volatility
        pressure_ratio = signed_vol / (realized_vol + 1e-12) \
            if not np.isnan(realized_vol) else np.nan

        # run-length buy dominance
        bin_signs = _bin_buy_signs(g, bin_size=buy_run_bin)
        buy_run_feats = _compute_buy_run_features(bin_signs)

        # convexity (acceleration/deceleration)
        convex_feats = _compute_price_convexity(prices)

        # short vs long horizon return ratio
        ts_last = g["ts"].max()

        # Ensure numeric types for p_short and p_long
        p_short = float(p_open)
        p_long = float(p_open)

        t_short = ts_last - pd.Timedelta(short_horizon)
        t_long = ts_last - pd.Timedelta(long_horizon)

        p_last = p_close

        # approximate past prices at those horizons
        p_short = p_open
        p_long = p_open

        short_candidates = g[g["ts"] >= t_short]["price"]
        if not short_candidates.empty:
            p_short = float(short_candidates.iloc[0])

        long_candidates = g[g["ts"] >= t_long]["price"]
        if not long_candidates.empty:
            p_long = float(long_candidates.iloc[0])

        r_short = np.log(p_last) - np.log(p_short) if p_short > 0 else 0.0
        r_long = np.log(p_last) - np.log(p_long) if p_long > 0 else 0.0
        
        # Handle cases where r_long might be zero or very close to zero
        if abs(r_long) < 1e-12: # Use a small epsilon to avoid division by zero
            return_ratio = 0.0 # Or np.nan, depending on desired behavior for undefined ratio
        else:
            return_ratio = float(r_short / r_long) # No need for abs() here if we want signed ratio

        row = {
            "symbol": symbol,
            "window_start": w_start,
            "p_open": p_open,
            "p_close": p_close,
            "p_high": p_high,
            "p_low": p_low,
            "total_volume": total_volume,
            "price_range": price_range,
            "price_mean": price_mean,
            "price_std": price_std,
            "realized_vol": realized_vol,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "signed_volume": signed_vol,
            "imbalance": imbalance,
            "compression_index": compression_index,
            "pressure_ratio": pressure_ratio,
            "return_ratio_short_long": return_ratio,
        }
        row.update(buy_run_feats)
        row.update(convex_feats)

        features.append(row)

    feat_df = pd.DataFrame(features)
    if not feat_df.empty:
        feat_df = feat_df.sort_values(["symbol", "window_start"]).reset_index(drop=True)
    return feat_df


# ============================================================
# 3. BASELINE (TIME-SERIES) Z-SCORES PER SYMBOL
# ============================================================

def add_baseline_zscores(
    window_features: pd.DataFrame,
    feature_cols: Sequence[str]
) -> pd.DataFrame:
    """
    For each symbol, compute expanding mean/std of each feature and
    add z-score columns: z_<feature> = (x - mean) / std.

    This gives a per-coin notion of "how extreme is this window vs its own past".
    """
    df = window_features.sort_values(["symbol", "window_start"]).copy()

    for col in feature_cols:
        mean_name = f"{col}_mean_hist"
        std_name = f"{col}_std_hist"
        z_name = f"z_{col}"

        df[mean_name] = (
            df.groupby("symbol")[col]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )
        df[std_name] = (
            df.groupby("symbol")[col]
              .expanding()
              .std(ddof=1)
              .reset_index(level=0, drop=True)
        )

        df[z_name] = (df[col] - df[mean_name]) / (df[std_name] + 1e-12)

    return df


# ============================================================
# 4. CROSS-SECTIONAL STATS (RELATIVE TO OTHER COINS SAME WINDOW)
# ============================================================

def add_cross_sectional_stats(
    window_features: pd.DataFrame,
    xsec_cols: Sequence[str]
) -> pd.DataFrame:
    """
    For each window_start, compute cross-sectional z-score and percentile rank
    for each feature in xsec_cols.

    Adds:
        xsec_z_<col>
        xsec_rank_<col>  (0..1, percentile)
    """
    df = window_features.copy()

    for col in xsec_cols:
        z_name = f"xsec_z_{col}"
        r_name = f"xsec_rank_{col}"

        grp = df.groupby("window_start")[col]
        mu = grp.transform("mean")
        sigma = grp.transform("std")

        df[z_name] = (df[col] - mu) / (sigma + 1e-12)
        df[r_name] = grp.rank(pct=True, method="average")

    return df


# ============================================================
# 5. PCA FACTORS (OPTIONAL)
# ============================================================

def fit_pca(
    window_features: pd.DataFrame,
    feature_cols: Sequence[str],
    n_components: int = 3
):
    """
    Fit PCA on standardized features. Use this offline on a big sample.

    Returns
    -------
    pca: PCA object
    feature_cols: list[str]
    col_means: np.ndarray
    col_stds:  np.ndarray
    """
    if PCA is None:
        raise ImportError("sklearn is required for PCA")

    X = window_features[feature_cols].values.astype(float)
    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0) + 1e-12
    X_norm = (X - col_means) / col_stds

    pca = PCA(n_components=n_components)
    pca.fit(X_norm)

    return pca, list(feature_cols), col_means, col_stds


def add_pca_factors(
    window_features: pd.DataFrame,
    pca,
    feature_cols: Sequence[str],
    col_means: np.ndarray,
    col_stds: np.ndarray
) -> pd.DataFrame:
    """
    Project current features onto PCA factors and append:
        factor_1, factor_2, ...
    """
    df = window_features.copy()
    X = df[feature_cols].values.astype(float)
    X_norm = (X - col_means) / col_stds
    F = pca.transform(X_norm)

    for i in range(F.shape[1]):
        df[f"factor_{i+1}"] = F[:, i]

    return df


# ============================================================
# 6. LABEL GENERATION: PRE-PUMP WINDOWS
# ============================================================
# Idea:
#   - Define a "pump" as: within the next lookahead_n windows, price rises
#     by at least pump_return_thresh (e.g. 10%) from current p_close.
#   - Define a "peak" window as where the local max is reached.
#   - Mark windows in [peak - pre_pump_start_offset, peak - pre_pump_end_offset]
#     as label = 1 (pre-pump region).
#   - Everything else: label = 0.


def label_pre_pump_windows(
    window_features: pd.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "window_start",
    price_col: str = "p_close",
    lookahead_n: int = 60,           # windows to look ahead for pump (e.g. 60 * 1min = 1h)
    pump_return_thresh: float = 0.10,  # 10% move
    pre_pump_start_offset: int = 30, # windows before peak where pre-pump starts
    pre_pump_end_offset: int = 5     # windows before peak where pre-pump ends
) -> pd.DataFrame:
    """
    Generate a binary label 'is_pre_pump' for each window based on future price path.

    Returns
    -------
    DataFrame: original columns + 'is_pre_pump' (0/1)
    """
    df = window_features.sort_values([symbol_col, time_col]).copy()
    df["is_pre_pump"] = 0  # default

    # We'll build labels per symbol
    for sym, g in df.groupby(symbol_col, sort=False):
        idx = g.index.to_numpy()
        prices = g[price_col].values

        n = len(g)
        # label array for this symbol
        labels = np.zeros(n, dtype=int)

        # for each window i, we check if a pump occurs in lookahead_n windows
        for i in range(n):
            p0 = prices[i]
            # lookahead slice (i+1 .. i+lookahead_n)
            j_start = i + 1
            j_end = min(n, i + 1 + lookahead_n)
            if j_start >= j_end:
                continue

            future_prices = prices[j_start:j_end]
            max_idx_rel = np.argmax(future_prices)
            peak_j = j_start + max_idx_rel
            peak_price = future_prices[max_idx_rel]

            # is this a "pump" relative to p0?
            if peak_price >= p0 * (1.0 + pump_return_thresh):
                # define pre-pump label region before this peak
                start = max(0, peak_j - pre_pump_start_offset)
                end = max(0, peak_j - pre_pump_end_offset)
                if end > start:
                    labels[start:end] = 1

        # write back labels
        df.loc[idx, "is_pre_pump"] = labels

    return df


# ============================================================
# 7. HAZARD MODEL: LOGISTIC REGRESSION
# ============================================================

def fit_hazard_model(
    window_features: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str]
):
    """
    Fit logistic regression for P(is_pre_pump=1 | features).

    Parameters
    ----------
    window_features : DataFrame
        Must contain label_col (0/1) and feature_cols (floats).
    label_col : str
        Name of label column, e.g. 'is_pre_pump'.
    feature_cols : Sequence[str]
        Features to use in the model.

    Returns
    -------
    clf: trained LogisticRegression model
    feature_cols: list[str]
    """
    if LogisticRegression is None:
        raise ImportError("sklearn is required for LogisticRegression")

    df = window_features.dropna(subset=list(feature_cols) + [label_col]).copy()
    X = df[list(feature_cols)].values.astype(float)
    y = df[label_col].values.astype(int)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"  # usually very imbalanced
    )
    clf.fit(X, y)
    return clf, list(feature_cols)


def add_hazard_score(
    window_features: pd.DataFrame,
    clf,
    feature_cols: Sequence[str],
    score_col: str = "pre_pump_score"
) -> pd.DataFrame:
    """
    Add column score_col = P(is_pre_pump=1 | features).

    Use this both offline and in live scoring.
    """
    df = window_features.copy()
    X = df[list(feature_cols)].values.astype(float)
    prob = clf.predict_proba(X)[:, 1]
    df[score_col] = prob
    return df


# ============================================================
# 8. HIGH-LEVEL WRAPPERS
# ============================================================

def build_feature_matrix(
    trades: pd.DataFrame,
    window: str = "60s"
) -> pd.DataFrame:
    """
    Full feature pipeline (no labels, no model):
        1) compute_window_features
        2) add_baseline_zscores
        3) add_cross_sectional_stats

    This is what you run every minute in backtest or batch mode.
    """
    feat_df = compute_window_features(trades, window=window)

    if feat_df.empty:
        return feat_df

    # Baseline z-scores per symbol
    baseline_cols = [
        "total_volume",
        "imbalance",
        "price_range",
        "realized_vol",
        "compression_index",
        "pressure_ratio",
        "max_buy_run_len",
        "price_convexity_a",
    ]
    feat_df = add_baseline_zscores(feat_df, baseline_cols)

    # Cross-sectional stats per window_start
    xsec_cols = ["total_volume", "compression_index", "price_convexity_a"]
    feat_df = add_cross_sectional_stats(feat_df, xsec_cols)

    return feat_df


def build_training_set(
    trades: pd.DataFrame,
    window: str = "60s",
    lookahead_n: int = 60,
    pump_return_thresh: float = 0.10,
    pre_pump_start_offset: int = 30,
    pre_pump_end_offset: int = 5
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Convenience function:
        trades -> features -> labels -> training set

    Returns
    -------
    labeled_df : DataFrame with 'is_pre_pump'
    feature_cols_for_model : list[str] of recommended feature columns
    """
    feat_df = build_feature_matrix(trades, window=window)
    if feat_df.empty:
        return feat_df, []

    labeled_df = label_pre_pump_windows(
        feat_df,
        symbol_col="symbol",
        time_col="window_start",
        price_col="p_close",
        lookahead_n=lookahead_n,
        pump_return_thresh=pump_return_thresh,
        pre_pump_start_offset=pre_pump_start_offset,
        pre_pump_end_offset=pre_pump_end_offset
    )

    # Recommended core feature set for hazard model
    feature_cols = [
        "z_total_volume",
        "z_compression_index",
        "z_imbalance",
        "z_pressure_ratio",
        "z_max_buy_run_len",
        "price_convexity_a",
        "price_convexity_r2",
        "xsec_z_total_volume",
        "xsec_z_compression_index",
        "xsec_z_price_convexity_a",
        "xsec_rank_total_volume",
        "xsec_rank_compression_index",
        "xsec_rank_price_convexity_a",
        "return_ratio_short_long",
    ]

    # Filter only those that exist (depending on what you turned on)
    feature_cols = [c for c in feature_cols if c in labeled_df.columns]

    return labeled_df, feature_cols


# ============================================================
# 9. QUICK USAGE EXAMPLE (OFFLINE / BACKTEST)
# ============================================================

if __name__ == "__main__":
    # Example synthetic usage.
    # Replace this with your actual trades DataFrame.
    #
    # trades DataFrame required columns:
    #   ts (datetime64[ns]), symbol (str), price (float), volume (float), side ('buy'/'sell')
    #
    # Example:
    #
    # trades = pd.read_csv("trades.csv", parse_dates=["ts"])
    #
    # For demo, we'll build a minimal toy dataset.

    import datetime as dt

    # Build dummy trades for 2 symbols for ~2 hours with random walk prices
    np.random.seed(42)

    def make_dummy_trades(symbol: str, start: dt.datetime, minutes: int = 120):
        rows = []
        price = 1.0
        for m in range(minutes):
            t0 = start + dt.timedelta(minutes=m)
            # a few trades per minute
            for k in range(5):
                ts = t0 + dt.timedelta(seconds=10 * k)
                # random walk with occasional bumps
                price *= np.exp(np.random.normal(0, 0.002))
                volume = float(np.random.exponential(50.0))
                side = "buy" if np.random.rand() < 0.5 else "sell"
                rows.append({
                    "ts": ts,
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "side": side,
                })
        return rows

    start_time = dt.datetime(2025, 1, 1, 0, 0, 0)
    rows = []
    rows += make_dummy_trades("COIN_A", start_time, minutes=180)
    rows += make_dummy_trades("COIN_B", start_time, minutes=180)
    trades = pd.DataFrame(rows)
    trades["ts"] = pd.to_datetime(trades["ts"])

    # 1) Build training set (features + labels)
    labeled_df, feature_cols = build_training_set(
        trades,
        window="1min",
        lookahead_n=60,            # look 60 min ahead
        pump_return_thresh=0.10,   # 10% pump
        pre_pump_start_offset=30,  # 30 windows (30min) before peak
        pre_pump_end_offset=5      # up to 5min before peak
    )

    print("Training set shape:", labeled_df.shape)
    print("Pre-pump positive label rate:",
          labeled_df["is_pre_pump"].mean() if "is_pre_pump" in labeled_df else None)
    print("Using features:", feature_cols)

    # 2) Fit hazard model (if sklearn available)
    if LogisticRegression is not None and feature_cols:
        clf, used_cols = fit_hazard_model(
            labeled_df,
            label_col="is_pre_pump",
            feature_cols=feature_cols
        )

        # 3) Add score on the same dataset (for evaluation / debugging)
        scored_df = add_hazard_score(labeled_df, clf, used_cols)

        print(scored_df[["symbol", "window_start", "is_pre_pump", "pre_pump_score"]].head())
    else:
        print("sklearn not available or no features – skipping model fit.")