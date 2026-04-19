"""
etf_return_probability.py
─────────────────────────
Calculate the empirical probability that the monthly total return of a given
ETF exceeds a specified threshold, over an optional date window.

Usage (CLI):
    python etf_return_probability.py [--ticker TICKER] [--threshold THRESHOLD]
                                     [--start YYYY-MM] [--end YYYY-MM]
                                     [--data PATH]

Examples:
    # Probability SPY monthly return > 1% over all available data
    python etf_return_probability.py --ticker SPY --threshold 0.01

    # Probability QQQ monthly return > 2% from Jan 2015 to Dec 2019
    python etf_return_probability.py --ticker QQQ --threshold 0.02 \
        --start 2015-01 --end 2019-12

    # Use a custom data file path
    python etf_return_probability.py --ticker AGG --threshold 0.005 \
        --data /path/to/etf_monthly_ohlc_dividends.csv
"""

import argparse
import os
import sys

import pandas as pd


# ── Default path: same directory as this script, or one level up in data/ ────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "..", "data", "etf_monthly_ohlc_dividends.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load and lightly validate the ETF CSV."""
    if not os.path.exists(path):
        sys.exit(f"Error: data file not found at '{path}'")

    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "return"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Error: CSV is missing columns: {missing}")

    return df


def filter_data(
    df: pd.DataFrame,
    ticker: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Subset to the requested ticker and optional date window."""
    ticker = ticker.upper()
    available = df["ticker"].unique().tolist()
    if ticker not in available:
        sys.exit(f"Error: ticker '{ticker}' not found. Available: {available}")

    sub = df[df["ticker"] == ticker].copy()

    if start:
        try:
            start_dt = pd.to_datetime(start)
        except ValueError:
            sys.exit(f"Error: cannot parse --start '{start}'. Use YYYY-MM format.")
        sub = sub[sub["date"] >= start_dt]

    if end:
        try:
            # Treat end as inclusive month-end
            end_dt = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        except ValueError:
            sys.exit(f"Error: cannot parse --end '{end}'. Use YYYY-MM format.")
        sub = sub[sub["date"] <= end_dt]

    # Drop the first observation of the original series if it has NaN return
    # (no previous month available), then drop any remaining NaNs
    sub = sub.dropna(subset=["return"])

    if sub.empty:
        sys.exit("Error: no data remaining after applying filters.")

    return sub.sort_values("date").reset_index(drop=True)


def compute_probability(
    df: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Compute the empirical probability that monthly return > threshold.

    Returns a dict with summary statistics.
    """
    returns = df["return"]
    n_total = len(returns)
    n_above = (returns > threshold).sum()
    probability = n_above / n_total

    return {
        "n_months": n_total,
        "n_above_threshold": int(n_above),
        "probability": probability,
        "threshold": threshold,
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "std_return": returns.std(ddof=1),
        "min_return": returns.min(),
        "max_return": returns.max(),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
    }


def print_report(ticker: str, stats: dict) -> None:
    """Pretty-print the analysis results."""
    pct = lambda x: f"{x * 100:.4f}%"  # noqa: E731

    print()
    print("=" * 55)
    print(f"  ETF Monthly Return Probability Analysis")
    print("=" * 55)
    print(f"  Ticker     : {ticker.upper()}")
    print(f"  Period     : {stats['date_start'].strftime('%b %Y')} → "
          f"{stats['date_end'].strftime('%b %Y')}")
    print(f"  Months     : {stats['n_months']}")
    print("-" * 55)
    print(f"  Threshold  : {pct(stats['threshold'])}")
    print(f"  P(return > threshold) : "
          f"{stats['probability']:.4f}  ({pct(stats['probability'])})")
    print(f"  Months above threshold: {stats['n_above_threshold']} "
          f"/ {stats['n_months']}")
    print("-" * 55)
    print(f"  Mean return   : {pct(stats['mean_return'])}")
    print(f"  Median return : {pct(stats['median_return'])}")
    print(f"  Std dev       : {pct(stats['std_return'])}")
    print(f"  Min return    : {pct(stats['min_return'])}")
    print(f"  Max return    : {pct(stats['max_return'])}")
    print("=" * 55)
    print()


# ── Programmatic API ──────────────────────────────────────────────────────────

def return_probability(
    ticker: str,
    threshold: float = 0.01,
    start: str | None = None,
    end: str | None = None,
    data_path: str = _DEFAULT_DATA,
    verbose: bool = True,
) -> dict:
    """
    Calculate the probability that a given ETF's monthly return exceeds
    `threshold` over the specified date range.

    Parameters
    ----------
    ticker : str
        ETF ticker symbol (e.g. 'SPY', 'QQQ', 'AGG').
    threshold : float
        Return threshold as a decimal (default 0.01 = 1%).
    start : str, optional
        Start month in 'YYYY-MM' format (inclusive). Default: all available.
    end : str, optional
        End month in 'YYYY-MM' format (inclusive). Default: all available.
    data_path : str
        Path to the CSV file. Defaults to ../data/etf_monthly_ohlc_dividends.csv
        relative to this script.
    verbose : bool
        If True, print a formatted report to stdout.

    Returns
    -------
    dict with keys: n_months, n_above_threshold, probability, threshold,
                    mean_return, median_return, std_return, min_return,
                    max_return, date_start, date_end
    """
    df_raw = load_data(data_path)
    df_filtered = filter_data(df_raw, ticker, start, end)
    stats = compute_probability(df_filtered, threshold)

    if verbose:
        print_report(ticker, stats)

    return stats


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probability that an ETF's monthly return exceeds a threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", "-t",
        default="SPY",
        help="ETF ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--threshold", "-p",
        type=float,
        default=0.01,
        help="Return threshold as a decimal (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        metavar="YYYY-MM",
        help="Start month, inclusive (default: earliest available)",
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        metavar="YYYY-MM",
        help="End month, inclusive (default: latest available)",
    )
    parser.add_argument(
        "--data", "-d",
        default=_DEFAULT_DATA,
        metavar="PATH",
        help="Path to the ETF CSV file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    return_probability(
        ticker=args.ticker,
        threshold=args.threshold,
        start=args.start,
        end=args.end,
        data_path=args.data,
        verbose=True,
    )
