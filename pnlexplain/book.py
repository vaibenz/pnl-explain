"""Turning trades and marks into positions and model values."""
import numpy as np
import pandas as pd

from . import bs

KEY = ["underlying", "kind", "strike", "expiry"]


def load_marks(path):
    m = pd.read_csv(path, parse_dates=["date"])
    return m.set_index(["date", "underlying"]).sort_index()


def load_trades(path):
    t = pd.read_csv(path, parse_dates=["date", "expiry"])
    t["strike"] = t["strike"].fillna(0.0)
    return t


def positions_at(trades, date, include_today=False):
    """Net position per contract after the close of `date`
    (or before today's trades if include_today is False)."""
    mask = trades["date"] <= date if include_today else trades["date"] < date
    t = trades[mask].copy()
    t["expiry"] = t["expiry"].fillna(pd.Timestamp("2100-01-01"))
    pos = t.groupby(KEY, as_index=False).agg(qty=("qty", "sum"), multiplier=("multiplier", "first"))
    pos = pos[pos["qty"] != 0]
    # drop anything that expired before this date
    pos = pos[pos["expiry"] >= date]
    return pos.reset_index(drop=True)


def line_inputs(pos, marks, date, spot_override=None, vol_override=None, date_override=None):
    """Pricing inputs for each line on a given day.

    Vol per line comes from the day's ATM vol plus a simple linear skew in
    log moneyness, so when spot drops a fixed strike picks up vol (sticky
    moneyness). The overrides let attribution move one factor at a time.
    """
    day = marks.xs(date, level="date")
    und = pos["underlying"]
    S = und.map(day["spot"]).to_numpy(float) if spot_override is None else spot_override
    atm = und.map(day["atm_vol"]).to_numpy(float) if vol_override is None else vol_override
    skew = und.map(day["skew"]).to_numpy(float)
    K = pos["strike"].to_numpy(float)
    is_fut = (pos["kind"] == "future").to_numpy()
    Ksafe = np.where(is_fut, S, K)
    vol = np.maximum(atm + skew * np.log(Ksafe / S), 0.03)
    asof = date if date_override is None else date_override
    T = np.maximum((pos["expiry"] - asof).dt.days.to_numpy() / 365, 0.0)
    return dict(
        S=S, K=Ksafe, T=T, vol=vol, atm=atm,
        r=und.map(day["rate"]).to_numpy(float),
        q=und.map(day["div_yield"]).to_numpy(float),
        call=(pos["kind"] == "call").to_numpy(),
        fut=is_fut,
        scale=(pos["qty"] * pos["multiplier"] * und.map(day["usd_per_unit"])).to_numpy(float),
    )


def unit_value(x):
    """Per unit model value. Futures are valued at spot so differences give
    the futures P&L (no carry modelled)."""
    opt = bs.price(x["S"], x["K"], x["T"], x["r"], x["q"], x["vol"], x["call"])
    return np.where(x["fut"], x["S"], opt)


def book_value(pos, marks, date, **over):
    x = line_inputs(pos, marks, date, **over)
    return unit_value(x) * x["scale"]
