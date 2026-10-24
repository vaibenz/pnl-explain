import numpy as np
import pandas as pd
import pytest

from pnlexplain.attribution import explain_day, explain_range, new_trade_pnl, step_reval
from pnlexplain.book import load_marks, load_trades


def make_marks(spot1, vol1, skew=0.0):
    rows = []
    for d, s, v in [("2026-01-05", 100.0, 0.20), ("2026-01-06", spot1, vol1)]:
        rows.append([pd.Timestamp(d), "X", s, v, skew, 0.0, 0.0, 1, 1.0])
    m = pd.DataFrame(rows, columns=["date", "underlying", "spot", "atm_vol", "skew", "rate", "div_yield",
                                    "multiplier", "usd_per_unit"])
    return m.set_index(["date", "underlying"])


def one_option(kind="call", qty=10, expiry="2026-04-17"):
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-01-02"), "trade_id": "A", "underlying": "X", "kind": kind,
        "strike": 100.0, "expiry": pd.Timestamp(expiry), "qty": qty, "price": 5.0, "multiplier": 1,
    }])


D0, D1 = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")


def test_small_move_explained_well():
    e = explain_day(one_option(), make_marks(100.5, 0.202), D0, D1)
    assert abs(e["unexplained"].sum()) < 1e-3 * abs(e["actual"].sum()) + 1e-3


def test_second_order_helps_on_big_move():
    m = make_marks(92.0, 0.27)
    first = explain_day(one_option(), m, D0, D1, second_order=False)["unexplained"].abs().sum()
    second = explain_day(one_option(), m, D0, D1, second_order=True)["unexplained"].abs().sum()
    assert second < first


def test_no_move_is_just_theta():
    e = explain_day(one_option(), make_marks(100.0, 0.20), D0, D1)
    assert e["theta"].sum() == pytest.approx(e["actual"].sum(), rel=1e-2)  # theta itself decays a bit over the day
    assert e["theta"].sum() < 0  # long option bleeds


def test_step_reval_adds_up():
    s = step_reval(one_option(), make_marks(95.0, 0.24, skew=-0.2), D0, D1)
    assert s[["spot", "vol", "time_carry"]].sum() == pytest.approx(s["actual"])


def test_new_trade_edge_positive_for_cheap_buy():
    t = one_option()
    t.loc[0, "date"] = D1
    t.loc[0, "price"] = 0.01                 # bought for basically nothing
    nt = new_trade_pnl(t, make_marks(100.0, 0.2), D1)
    assert nt["pnl"].iloc[0] > 0


def test_sample_data_flags_the_selloff():
    t, m = load_trades("data/trades.csv"), load_marks("data/marks.csv")
    df = explain_range(t, m, second_order=False)
    worst = df["unexplained"].abs().idxmax()
    assert df.loc[worst, "flag"]
    assert df["delta"].abs().loc[worst] > df["delta"].abs().median() * 3
