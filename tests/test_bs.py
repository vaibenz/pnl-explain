import numpy as np
import pytest

from pnlexplain.bs import greeks, price

ARGS = dict(K=100.0, T=0.4, r=0.03, q=0.01)


def px(S=100.0, vol=0.25, call=True):
    return price(S, vol=vol, call=call, **ARGS)


@pytest.mark.parametrize("call", [True, False])
def test_greeks_vs_bumps(call):
    S, v, h, dv = 100.0, 0.25, 0.01, 1e-4
    g = greeks(S, vol=v, call=call, **ARGS)
    assert g["delta"] == pytest.approx((px(S + h, v, call) - px(S - h, v, call)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((px(S + 0.1, v, call) - 2 * px(S, v, call) + px(S - 0.1, v, call)) / 0.01, abs=1e-5)
    assert g["vega"] == pytest.approx((px(S, v + dv, call) - px(S, v - dv, call)) / (2 * dv) / 100, abs=1e-6)

    d_up = greeks(S, vol=v + dv, call=call, **ARGS)["delta"]
    d_dn = greeks(S, vol=v - dv, call=call, **ARGS)["delta"]
    assert g["vanna"] == pytest.approx((d_up - d_dn) / (2 * dv) / 100, abs=1e-6)

    v_up = greeks(S, vol=v + dv, call=call, **ARGS)["vega"]
    v_dn = greeks(S, vol=v - dv, call=call, **ARGS)["vega"]
    assert g["volga"] == pytest.approx((v_up - v_dn) / (2 * dv) / 100, abs=1e-6)


def test_expired_payoff():
    assert price(110, 100, 0.0, 0.0, 0.0, 0.2, True) == 10
