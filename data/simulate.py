"""Builds a fake but realistic-ish trading history to explain.

Writes data/marks.csv (end of day spot, ATM vol, skew per underlying) and
data/trades.csv (fills). The first day's trades are the opening book.
Each day the "desk" picks up a handful of fills at theo +/- a small edge,
so new trade P&L should come out positive on average.

Day 40 has a sell off with a vol spike, which is where a first order
explain should struggle.

    python data/simulate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnlexplain.bs import price  # noqa: E402

rng = np.random.default_rng(11)
DAYS = 65
dates = pd.bdate_range(end="2026-10-23", periods=DAYS)

UND = {
    #        spot    vol   skew   r      q      mult  usd       tick
    "NKY": (38500, 0.21, -0.25, 0.002, 0.018, 1000, 0.0066, 250),
    "HSI": (17800, 0.26, -0.20, 0.040, 0.035, 50, 0.1286, 200),
    "TWSE": (22400, 0.19, -0.22, 0.015, 0.028, 200, 0.0312, 100),
}
EXPIRIES = pd.to_datetime(["2026-09-11", "2026-10-09", "2026-11-13", "2026-12-11", "2027-03-12"])

marks = []
spot = {u: v[0] for u, v in UND.items()}
vol = {u: v[1] for u, v in UND.items()}
for i, d in enumerate(dates):
    common = rng.standard_normal()
    for u, (s0, v0, skew, r, q, mult, fx, tick) in UND.items():
        if i > 0:
            ret = vol[u] / np.sqrt(252) * (0.7 * common + 0.71 * rng.standard_normal())
            if i == 40:
                ret -= 0.045
            dv = -0.4 * ret + 0.003 * rng.standard_normal() + 0.05 * (v0 - vol[u])
            if i == 40:
                dv += 0.05
            spot[u] *= 1 + ret
            vol[u] = max(vol[u] + dv, 0.08)
        marks.append([d.date(), u, round(spot[u], 2), round(vol[u], 4), skew, r, q, mult, fx])

marks = pd.DataFrame(marks, columns=["date", "underlying", "spot", "atm_vol", "skew", "rate", "div_yield",
                                     "multiplier", "usd_per_unit"])
marks.to_csv("data/marks.csv", index=False)

m = marks.set_index(["date", "underlying"])
trades = []
tid = 1
for i, d in enumerate(dates):
    n = 30 if i == 0 else rng.integers(3, 9)
    for _ in range(n):
        u = rng.choice(list(UND))
        s0, v0, skew, r, q, mult, fx, tick = UND[u]
        live = EXPIRIES[EXPIRIES > d + pd.Timedelta(days=7)]
        exp = pd.Timestamp(rng.choice(live[:3]))
        row = m.loc[(d.date(), u)]
        S = row["spot"]
        # intraday spot somewhere between yesterday's close and today's
        S_prev = m.loc[(dates[i - 1].date(), u), "spot"] if i else S
        S_fill = S_prev + rng.uniform() * (S - S_prev)
        K = round(S * rng.choice([0.92, 0.96, 1.0, 1.04, 1.08]) / tick) * tick
        call = K >= S if rng.uniform() < 0.8 else K < S      # mostly OTM
        T = (exp - d).days / 365
        v = row["atm_vol"] + skew * np.log(K / S_fill)
        theo = float(price(S_fill, K, T, r, q, v, call))
        qty = int(rng.choice([-1, 1]) * rng.integers(5, 40))
        edge = max(0.015 * theo, 0.5 * tick / 100)
        px = theo - edge if qty > 0 else theo + edge         # buy below theo, sell above
        trades.append([d.date(), f"T{tid:04d}", u, "call" if call else "put", K, exp.date(),
                       qty, round(max(px, 0.01), 2), mult])
        tid += 1
    # small futures hedge most days to keep delta in check
    if i > 0 and rng.uniform() < 0.6:
        u = rng.choice(list(UND))
        trades.append([d.date(), f"T{tid:04d}", u, "future", np.nan, pd.NaT, int(rng.integers(-15, 15)),
                       m.loc[(d.date(), u), "spot"], UND[u][5]])
        tid += 1

pd.DataFrame(trades, columns=["date", "trade_id", "underlying", "kind", "strike", "expiry", "qty", "price",
                              "multiplier"]).to_csv("data/trades.csv", index=False)
print(f"{len(marks)} marks, {len(trades)} trades")
