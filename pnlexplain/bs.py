"""Black Scholes pricing and the greeks needed for P&L explain.

Units (per one option, before qty / multiplier):
    delta  dV/dS
    gamma  d2V/dS2
    vega   per 1 vol point (sigma + 0.01)
    theta  per calendar day
    vanna  d(delta)/d(vol point)
    volga  d(vega)/d(vol point), i.e. second derivative per vol point^2
"""
import numpy as np
from scipy.stats import norm


def _d(S, K, T, r, q, vol):
    st = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / st
    return d1, d1 - st


def price(S, K, T, r, q, vol, call):
    S, K, T, vol = map(lambda x: np.asarray(x, float), (S, K, T, vol))
    call = np.asarray(call, bool)
    expired = T <= 0
    T = np.where(expired, 1e-10, T)
    d1, d2 = _d(S, K, T, r, q, vol)
    fq, fr = np.exp(-q * T), np.exp(-r * T)
    c = S * fq * norm.cdf(d1) - K * fr * norm.cdf(d2)
    p = K * fr * norm.cdf(-d2) - S * fq * norm.cdf(-d1)
    payoff = np.where(call, np.maximum(S - K, 0), np.maximum(K - S, 0))
    return np.where(expired, payoff, np.where(call, c, p))


def greeks(S, K, T, r, q, vol, call):
    S, K, T, vol = map(lambda x: np.asarray(x, float), (S, K, T, vol))
    call = np.asarray(call, bool)
    T = np.maximum(T, 1e-10)
    d1, d2 = _d(S, K, T, r, q, vol)
    fq, fr = np.exp(-q * T), np.exp(-r * T)
    n1 = norm.pdf(d1)
    sq = np.sqrt(T)

    delta = np.where(call, fq * norm.cdf(d1), fq * (norm.cdf(d1) - 1))
    gamma = fq * n1 / (S * vol * sq)
    vega = S * fq * n1 * sq / 100
    a = -S * fq * n1 * vol / (2 * sq)
    theta = np.where(
        call,
        a - r * K * fr * norm.cdf(d2) + q * S * fq * norm.cdf(d1),
        a + r * K * fr * norm.cdf(-d2) - q * S * fq * norm.cdf(-d1),
    ) / 365
    vanna = -fq * n1 * d2 / vol / 100
    volga = vega * d1 * d2 / vol / 100
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, vanna=vanna, volga=volga)
