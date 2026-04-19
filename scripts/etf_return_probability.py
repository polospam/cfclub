"""
etf_return_probability.py
─────────────────────────
Calculate the empirical probability that the monthly total return of a given
ETF exceeds a specified threshold, over an optional date window.

Also produces:
  • A heatmap of P(return > threshold) broken down by year × ticker.
  • A rolling-window chart: P(cumulative N-month return > threshold) by
    start-of-window calendar month (Jan–Dec), per ticker.

Usage (CLI):
    python etf_return_probability.py [--ticker TICKER] [--threshold THRESHOLD]
                                     [--start YYYY-MM] [--end YYYY-MM]
                                     [--data PATH] [--heatmap] [--rolling]
                                     [--window N] [--out PATH]

Examples:
    # Probability SPY monthly return > 1% over all available data
    python etf_return_probability.py --ticker SPY --threshold 0.01

    # Probability QQQ monthly return > 2% from Jan 2015 to Dec 2019
    python etf_return_probability.py --ticker QQQ --threshold 0.02 \\
        --start 2015-01 --end 2019-12

    # Year × ticker heatmap (monthly returns, threshold 1%)
    python etf_return_probability.py --heatmap --threshold 0.01

    # Rolling-window chart: P(12-month return > 9%) by start month, all tickers
    python etf_return_probability.py --rolling --window 12 --threshold 0.09

    # Same but filtered to 2010-2020 and saved to a custom path
    python etf_return_probability.py --rolling --window 12 --threshold 0.09 \\
        --start 2010-01 --end 2020-12 --out my_rolling.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Default paths ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "..", "data", "etf_monthly_ohlc_dividends.csv")
_DEFAULT_OUT  = os.path.join(_SCRIPT_DIR, "..", "data", "etf_return_heatmap.png")


# ── Data loading & filtering ──────────────────────────────────────────────────

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
            end_dt = pd.to_datetime(end) + pd.offsets.MonthEnd(0)
        except ValueError:
            sys.exit(f"Error: cannot parse --end '{end}'. Use YYYY-MM format.")
        sub = sub[sub["date"] <= end_dt]

    sub = sub.dropna(subset=["return"])

    if sub.empty:
        sys.exit("Error: no data remaining after applying filters.")

    return sub.sort_values("date").reset_index(drop=True)


# ── Single-series analysis ────────────────────────────────────────────────────

def compute_probability(df: pd.DataFrame, threshold: float) -> dict:
    """Compute the empirical P(return > threshold) and summary stats."""
    returns = df["return"]
    n_total = len(returns)
    n_above = (returns > threshold).sum()

    return {
        "n_months":           n_total,
        "n_above_threshold":  int(n_above),
        "probability":        n_above / n_total,
        "threshold":          threshold,
        "mean_return":        returns.mean(),
        "median_return":      returns.median(),
        "std_return":         returns.std(ddof=1),
        "min_return":         returns.min(),
        "max_return":         returns.max(),
        "date_start":         df["date"].min(),
        "date_end":           df["date"].max(),
    }


def print_report(ticker: str, stats: dict) -> None:
    """Pretty-print the analysis results."""
    pct = lambda x: f"{x * 100:.4f}%"  # noqa: E731

    print()
    print("=" * 55)
    print("  ETF Monthly Return Probability Analysis")
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


# ── Heatmap ───────────────────────────────────────────────────────────────────

def build_heatmap_data(
    df: pd.DataFrame,
    threshold: float,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a (year × ticker) DataFrame of P(return > threshold).

    Cells with fewer than 3 observations are set to NaN to avoid
    misleading probabilities from very small samples.
    """
    df = df.dropna(subset=["return"]).copy()
    df["year"] = df["date"].dt.year

    if tickers:
        df = df[df["ticker"].isin([t.upper() for t in tickers])]

    MIN_OBS = 3  # require at least this many months per cell

    def prob(series):
        if len(series) < MIN_OBS:
            return np.nan
        return (series > threshold).sum() / len(series)

    pivot = (
        df.groupby(["year", "ticker"])["return"]
        .apply(prob)
        .unstack("ticker")
    )

    # Order columns as they appear in the data (chronological first ticker)
    col_order = df.groupby("ticker")["date"].min().sort_values().index.tolist()
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    return pivot


def plot_heatmap(
    pivot: pd.DataFrame,
    threshold: float,
    out_path: str,
) -> None:
    """Render and save the year × ticker heatmap."""
    n_years, n_tickers = pivot.shape

    fig_w = max(5, 2.2 * n_tickers + 1.5)
    fig_h = max(5, 0.42 * n_years + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colour map: white at 50%, green above, red below
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(
        pivot.values,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    # Axes labels
    ax.set_xticks(range(n_tickers))
    ax.set_xticklabels(pivot.columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_years))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Cell annotations
    for row_i, year in enumerate(pivot.index):
        for col_j, tkr in enumerate(pivot.columns):
            val = pivot.loc[year, tkr]
            if np.isnan(val):
                label = "—"
                txt_color = "#999999"
            else:
                label = f"{val * 100:.0f}%"
                # White text on dark cells (extreme red/green), dark text on mid-range
                txt_color = "white" if (val < 0.30 or val > 0.72) else "#111111"
            ax.text(
                col_j, row_i, label,
                ha="center", va="center",
                fontsize=9, color=txt_color, fontweight="bold",
            )

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("P(return > threshold)", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Title & footnote
    thr_str = f"{threshold * 100:g}%"
    ax.set_title(
        f"Monthly Return Probability by Year  ·  threshold = {thr_str}",
        fontsize=13, fontweight="bold", pad=18,
    )
    fig.text(
        0.5, 0.01,
        f"P(monthly total return > {thr_str}) per calendar year. "
        "Cells with < 3 observations shown as —.",
        ha="center", fontsize=7.5, color="#555555",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved → {out_path}")


def generate_heatmap(
    threshold: float = 0.01,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    data_path: str = _DEFAULT_DATA,
    out_path: str = _DEFAULT_OUT,
) -> str:
    """
    Build and save a year × ticker heatmap of P(return > threshold).

    Parameters
    ----------
    threshold : float   Return threshold as a decimal (default 0.01 = 1%).
    tickers   : list    Tickers to include (default: all in the file).
    start     : str     Start month 'YYYY-MM' (default: earliest available).
    end       : str     End month 'YYYY-MM' (default: latest available).
    data_path : str     Path to the ETF CSV.
    out_path  : str     Output PNG path.

    Returns
    -------
    str  Absolute path to the saved PNG.
    """
    df = load_data(data_path)
    df = df.dropna(subset=["return"])

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end) + pd.offsets.MonthEnd(0)]

    pivot = build_heatmap_data(df, threshold, tickers)
    plot_heatmap(pivot, threshold, out_path)
    return os.path.abspath(out_path)


# ── Rolling-window analysis ──────────────────────────────────────────────────

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def compute_rolling_returns(
    df: pd.DataFrame,
    ticker: str,
    window: int,
) -> pd.DataFrame:
    """
    Compute cumulative total returns for every rolling N-month window
    for a single ticker.

    The cumulative return of a window starting at month t is:
        (1 + r_t) * (1 + r_{t+1}) * ... * (1 + r_{t+N-1}) - 1

    Returns a DataFrame with columns: date_start, start_month, cum_return
    where date_start is the first month of each window.
    """
    sub = df[df["ticker"] == ticker].dropna(subset=["return"]).sort_values("date").reset_index(drop=True)

    gross = (1 + sub["return"]).values
    dates = sub["date"].values

    records = []
    for i in range(len(gross) - window + 1):
        cum = gross[i : i + window].prod() - 1
        start_date = pd.Timestamp(dates[i])
        records.append({
            "date_start":   start_date,
            "start_month":  start_date.month,   # 1–12
            "cum_return":   cum,
        })

    return pd.DataFrame(records)


def build_rolling_prob_table(
    df: pd.DataFrame,
    window: int,
    threshold: float,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a (start_month × ticker) table of P(N-month cum. return > threshold).

    Rows = calendar months 1–12 (Jan–Dec)
    Cols = tickers
    Values = empirical probability (NaN if fewer than 3 observations)
    """
    MIN_OBS = 3
    all_tickers = tickers or sorted(df["ticker"].dropna().unique().tolist())

    table = {}
    for tkr in all_tickers:
        roll = compute_rolling_returns(df, tkr, window)
        probs = {}
        for m in range(1, 13):
            subset = roll[roll["start_month"] == m]["cum_return"]
            probs[m] = (subset > threshold).sum() / len(subset) if len(subset) >= MIN_OBS else float("nan")
        table[tkr] = probs

    result = pd.DataFrame(table, index=range(1, 13))
    result.index.name = "start_month"
    return result


def plot_rolling_chart(
    prob_table: pd.DataFrame,
    window: int,
    threshold: float,
    out_path: str,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """
    Bar-chart (grouped by ticker) of P(N-month return > threshold)
    for each calendar start month.
    """
    tickers  = prob_table.columns.tolist()
    months   = list(range(1, 13))
    n_tkr    = len(tickers)
    x        = np.arange(12)
    bar_w    = 0.72 / n_tkr

    # Palette — distinct, colourblind-friendly
    palette = ["#2166ac", "#d6604d", "#4dac26", "#8073ac"]
    colors   = palette[:n_tkr]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    for i, tkr in enumerate(tickers):
        vals   = [prob_table.loc[m, tkr] for m in months]
        offset = (i - (n_tkr - 1) / 2) * bar_w
        bars   = ax.bar(
            x + offset, vals,
            width=bar_w * 0.92,
            label=tkr,
            color=colors[i],
            alpha=0.88,
            zorder=3,
        )
        # Annotate each bar
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val * 100:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color=colors[i], fontweight="bold",
                )

    # 50% reference line
    ax.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", zorder=2, label="50% line")

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, fontsize=10)
    ax.set_xlabel("Start Month of Window", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, min(1.15, ax.get_ylim()[1] + 0.08))
    ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    thr_str = f"{threshold * 100:g}%"
    period_str = ""
    if start or end:
        period_str = f"  ·  {start or 'start'} → {end or 'end'}"

    ax.set_title(
        f"P({window}-Month Cumulative Return > {thr_str})  by Start Month{period_str}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.legend(title="Ticker", fontsize=9, title_fontsize=9, framealpha=0.7)

    fig.text(
        0.5, 0.01,
        f"Empirical probability that the cumulative total return of a "
        f"{window}-month window exceeds {thr_str}, grouped by the window's "
        f"starting calendar month. Cells with < 3 observations excluded.",
        ha="center", fontsize=7.5, color="#444444",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Rolling-window chart saved → {out_path}")


def generate_rolling_chart(
    window: int = 12,
    threshold: float = 0.0,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    data_path: str = _DEFAULT_DATA,
    out_path: str | None = None,
) -> str:
    """
    Build and save P(N-month return > threshold) by start month chart.

    Parameters
    ----------
    window    : int    Rolling window length in months (default 12).
    threshold : float  Cumulative return threshold as decimal (default 0.0 = 0%).
    tickers   : list   Tickers to include (default: all in the file).
    start     : str    Earliest data month 'YYYY-MM' to include.
    end       : str    Latest data month 'YYYY-MM' to include.
    data_path : str    Path to the ETF CSV.
    out_path  : str    Output PNG path (auto-generated if None).

    Returns
    -------
    str  Absolute path to the saved PNG.
    """
    if out_path is None:
        stem = f"etf_rolling{window}m_{int(threshold*100)}pct_heatmap.png"
        out_path = os.path.join(os.path.dirname(_DEFAULT_OUT), stem)

    df = load_data(data_path)
    df = df.dropna(subset=["return"])

    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end) + pd.offsets.MonthEnd(0)]

    prob_table = build_rolling_prob_table(df, window, threshold, tickers)
    plot_rolling_chart(prob_table, window, threshold, out_path, start, end)
    return os.path.abspath(out_path)


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
    ticker    : str    ETF ticker symbol (e.g. 'SPY', 'QQQ', 'AGG').
    threshold : float  Return threshold as a decimal (default 0.01 = 1%).
    start     : str    Start month 'YYYY-MM' (inclusive). Default: all data.
    end       : str    End month 'YYYY-MM' (inclusive). Default: all data.
    data_path : str    Path to the CSV file.
    verbose   : bool   If True, print a formatted report.

    Returns
    -------
    dict with keys: n_months, n_above_threshold, probability, threshold,
                    mean_return, median_return, std_return, min_return,
                    max_return, date_start, date_end
    """
    df_raw      = load_data(data_path)
    df_filtered = filter_data(df_raw, ticker, start, end)
    stats       = compute_probability(df_filtered, threshold)

    if verbose:
        print_report(ticker, stats)

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ETF return probability toolkit. "
            "Single-ticker stats (default), year×ticker heatmap (--heatmap), "
            "or rolling N-month window by start month (--rolling)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",    "-t", default="SPY",
                        help="ETF ticker (default: SPY). Ignored with --heatmap / --rolling.")
    parser.add_argument("--threshold", "-p", type=float, default=0.01,
                        help="Return threshold as decimal (default: 0.01 = 1%%)")
    parser.add_argument("--start",     "-s", default=None, metavar="YYYY-MM",
                        help="Start month, inclusive (default: earliest available)")
    parser.add_argument("--end",       "-e", default=None, metavar="YYYY-MM",
                        help="End month, inclusive (default: latest available)")
    parser.add_argument("--data",      "-d", default=_DEFAULT_DATA, metavar="PATH",
                        help="Path to the ETF CSV file")
    # Mode flags
    parser.add_argument("--heatmap",   action="store_true",
                        help="Generate a year × ticker monthly-return heatmap")
    parser.add_argument("--rolling",   action="store_true",
                        help="Generate a rolling N-month return probability chart by start month")
    parser.add_argument("--window",    "-w", type=int, default=12, metavar="N",
                        help="Rolling window length in months for --rolling (default: 12)")
    parser.add_argument("--out",       "-o", default=None, metavar="PATH",
                        help="Output PNG path (auto-named if omitted)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.heatmap:
        generate_heatmap(
            threshold=args.threshold,
            start=args.start,
            end=args.end,
            data_path=args.data,
            out_path=args.out or _DEFAULT_OUT,
        )
    elif args.rolling:
        generate_rolling_chart(
            window=args.window,
            threshold=args.threshold,
            start=args.start,
            end=args.end,
            data_path=args.data,
            out_path=args.out,
        )
    else:
        return_probability(
            ticker=args.ticker,
            threshold=args.threshold,
            start=args.start,
            end=args.end,
            data_path=args.data,
            verbose=True,
        )
