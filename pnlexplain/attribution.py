"""Daily P&L attribution.

For the book held overnight (positions at the start of day d1), compare

    actual  = V(d1 market) - V(d0 market)            full revaluation

against a greeks based explain using d0 greeks:

    delta   = delta * dS
    gamma   = 1/2 gamma * dS^2
    vega    = vega * dvol            (per line vol change, includes skew roll)
    theta   = theta * calendar days
    vanna   = vanna * dS * dvol
    volga   = 1/2 volga * dvol^2
    unexplained = actual - sum of the above

New trades done during d1 are reported separately as new trade P&L
(trade price vs end of day theo), which is where a market maker's edge
shows up.

There is also a step revaluation explain (spot, then vol, then time, each
fully revalued) which has no Taylor error, the cross terms just get
assigned to whichever factor moves later.
"""
import numpy as np
import pandas as pd

from . import bs
from .book import book_value, line_inputs, positions_at, unit_value

FIRST = ["delta", "gamma", "vega", "theta"]
SECOND = ["vanna", "volga"]


def explain_day(trades, marks, d0, d1, second_order=True):
    pos = positions_at(trades, d1)  # what we held going into d1
    x0 = line_inputs(pos, marks, d0)
    x1 = line_inputs(pos, marks, d1)

    v0 = unit_value(x0)
    v1 = unit_value(x1)
    actual = (v1 - v0) * x0["scale"]

    g = bs.greeks(x0["S"], x0["K"], x0["T"], x0["r"], x0["q"], x0["vol"], x0["call"])
    fut = x0["fut"]
    for k in g:
        g[k] = np.where(fut, 1.0 if k == "delta" else 0.0, g[k])

    dS = x1["S"] - x0["S"]
    dvol = (x1["vol"] - x0["vol"]) * 100        # in vol points
    days = (d1 - d0).days

    parts = {
        "delta": g["delta"] * dS,
        "gamma": 0.5 * g["gamma"] * dS**2,
        "vega": g["vega"] * dvol,
        "theta": g["theta"] * days,
        "vanna": g["vanna"] * dS * dvol,
        "volga": 0.5 * g["volga"] * dvol**2,
    }
    out = pos[["underlying", "kind", "strike", "expiry", "qty"]].copy()
    for k, v in parts.items():
        out[k] = v * x0["scale"]
    if not second_order:
        out[SECOND] = 0.0
    out["actual"] = actual
    out["unexplained"] = out["actual"] - out[FIRST + SECOND].sum(axis=1)
    return out


def step_reval(trades, marks, d0, d1):
    """Sequential full revaluation: move spot, then vol, then roll time."""
    pos = positions_at(trades, d1)
    x0 = line_inputs(pos, marks, d0)
    x1 = line_inputs(pos, marks, d1)

    base = book_value(pos, marks, d0)
    s = book_value(pos, marks, d0, spot_override=x1["S"])
    # new spot and new atm, skew still applied at the new spot
    sv = book_value(pos, marks, d0, spot_override=x1["S"], vol_override=x1["atm"])
    full = book_value(pos, marks, d1)

    return pd.Series({
        "spot": (s - base).sum(),          # includes the skew roll from the spot move
        "vol": (sv - s).sum(),
        "time_carry": (full - sv).sum(),   # theta plus any rate / div change
        "actual": (full - base).sum(),
    })


def new_trade_pnl(trades, marks, d1):
    """Today's fills marked against today's close theo."""
    t = trades[trades["date"] == d1].copy()
    if t.empty:
        return t.assign(theo=[], pnl=[])
    t["expiry"] = t["expiry"].fillna(pd.Timestamp("2100-01-01"))
    x = line_inputs(t, marks, d1)
    t["theo"] = unit_value(x)
    fx = x["scale"] / (t["qty"] * t["multiplier"]).to_numpy(float)
    t["pnl"] = (t["theo"] - t["price"]) * t["qty"] * t["multiplier"] * fx
    return t


def explain_range(trades, marks, threshold_pct=0.10, threshold_abs=5000.0, second_order=True):
    """Run the explain for every day in the marks and summarise per day.

    A day is flagged when the unexplained is bigger than both threshold_pct
    of the actual P&L and threshold_abs USD. Those are the days someone
    should look at (missing greek, bad mark, booking error...).
    """
    dates = marks.index.get_level_values("date").unique().sort_values()
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        e = explain_day(trades, marks, d0, d1, second_order)
        nt = new_trade_pnl(trades, marks, d1)
        row = {"date": d1, **e[FIRST + SECOND + ["actual", "unexplained"]].sum().to_dict()}
        row["new_trades"] = nt["pnl"].sum() if len(nt) else 0.0
        row["n_trades"] = len(nt)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    df["total"] = df["actual"] + df["new_trades"]
    big = df["unexplained"].abs() > np.maximum(threshold_pct * df["actual"].abs(), threshold_abs)
    df["flag"] = big
    return df
