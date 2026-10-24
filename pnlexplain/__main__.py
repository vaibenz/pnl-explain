"""Daily P&L explain report.

    python -m pnlexplain                       # whole history
    python -m pnlexplain --day 2026-09-21      # drill into one day
    python -m pnlexplain --first-order         # without vanna / volga
"""
import argparse
from pathlib import Path

import pandas as pd

from .attribution import FIRST, SECOND, explain_day, explain_range, new_trade_pnl, step_reval
from .book import load_marks, load_trades


def drill(trades, marks, day, second_order):
    dates = marks.index.get_level_values("date").unique().sort_values()
    i = dates.get_loc(day)
    d0, d1 = dates[i - 1], dates[i]
    e = explain_day(trades, marks, d0, d1, second_order)
    print(f"\n--- {d1.date()} (vs {d0.date()}) ---")
    print("\nby underlying\n", e.groupby("underlying")[FIRST + SECOND + ["actual", "unexplained"]].sum())
    print("\nlargest unexplained lines\n",
          e.reindex(e["unexplained"].abs().sort_values(ascending=False).index).head(5)
          [["underlying", "kind", "strike", "expiry", "qty", "actual", "unexplained"]])
    print("\nstep revaluation\n", step_reval(trades, marks, d0, d1).to_frame("usd"))
    nt = new_trade_pnl(trades, marks, d1)
    if len(nt):
        print(f"\nnew trades: {len(nt)}, pnl {nt['pnl'].sum():,.0f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--marks", default="data/marks.csv")
    p.add_argument("--trades", default="data/trades.csv")
    p.add_argument("--day")
    p.add_argument("--first-order", action="store_true")
    p.add_argument("--plot", action="store_true", help="save out/pnl_explain.png")
    a = p.parse_args(argv)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.0f}".format)
    trades, marks = load_trades(a.trades), load_marks(a.marks)
    second = not a.first_order

    if a.day:
        drill(trades, marks, pd.Timestamp(a.day), second)
        return

    df = explain_range(trades, marks, second_order=second)
    cols = FIRST + (SECOND if second else []) + ["unexplained", "actual", "new_trades", "total"]
    print("\ndaily explain (USD), last 10 days\n", df[cols].tail(10))
    print("\nperiod totals\n", df[cols].sum().to_frame("usd").T)

    abs_unexpl = df["unexplained"].abs().sum()
    abs_actual = df["actual"].abs().sum()
    print(f"\nexplain ratio: {1 - abs_unexpl / abs_actual:.1%} of absolute daily P&L explained by greeks")
    print(f"new trade P&L (edge): {df['new_trades'].sum():,.0f} over {int(df['n_trades'].sum())} fills")

    flagged = df[df["flag"]]
    print(f"\n{len(flagged)} day(s) flagged for review")
    if len(flagged):
        print(flagged[["actual", "unexplained"]])
        for d in flagged.index:
            drill(trades, marks, d, second)

    if a.plot:
        from .plots import explain_chart
        Path("out").mkdir(exist_ok=True)
        explain_chart(df, "out/pnl_explain.png")
        print("\nsaved out/pnl_explain.png")


if __name__ == "__main__":
    main()
