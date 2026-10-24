# pnl-explain

Daily P&L attribution for an options book. Every morning a risk team has to be able to say *why* the book made or lost money yesterday: how much came from delta, gamma, vega and theta, how much from new trades, and whether anything is left over that the greeks can't explain. A large unexplained number usually means a missing risk, a bad mark or a booking error, so those days get flagged.

This is a companion to my [options-risk-engine](https://github.com/vaibenz/options-risk-engine). Once you know the risk in the book, the next question is always why it made or lost what it did yesterday.

## How it works

For the book held overnight from d0 to d1:

* actual is a full revaluation of every line at d1's market minus its value at d0's market
* explain uses d0 greeks and the realised moves
  * delta x dS, ½ gamma x dS², vega x dvol, theta x days
  * optional second order cross terms: vanna x dS x dvol and ½ volga x dvol²
* unexplained = actual minus the sum of the explain terms
* new trades are marked separately against the close (fill price vs end of day theo), which is the edge a market maker collects

Vol per line is ATM vol plus a linear skew in log moneyness, so when spot falls a fixed strike picks up vol even if ATM vol doesn't move. That skew roll is part of the vega term.

There is also a step revaluation (move spot, then vol, then roll the date, full reval each time). It has no Taylor error, it just hands the cross effects to whichever factor moves later, so it's a good cross check for the greeks explain.

A day is flagged when |unexplained| is above both 10% of |actual| and USD 5,000.

## Data

`data/simulate.py` generates 65 business days of marks for NKY, HSI and TWSE plus ~400 fills (a 30 line opening book, then 3 to 8 option fills a day at theo ± 1.5% edge and the odd futures hedge). Day 40 has a 4.5% sell off with a 5 point vol spike on top of the normal noise.

## Running

```
pip install -r requirements.txt
python -m pytest
python -m pnlexplain --plot                 # full history + out/pnl_explain.png
python -m pnlexplain --day 2026-09-21       # drill into one day
python -m pnlexplain --first-order          # explain without vanna / volga
```

## What came out of it

On the sample history the greeks explain 96.3% of absolute daily P&L with second order terms on. Turning vanna and volga off pushes total absolute unexplained from about $325k to $524k and flags 7 days instead of 3.

The three flagged days were interesting:

* 11 Sep and 9 Oct are expiry days. Near expiry gamma is huge and changes fast, so a Taylor expansion off yesterday's greeks breaks down on the lines expiring that day. The drill down shows the unexplained sitting in lines expiring that day: the near the money NKY puts on 11 Sep, and HSI and TWSE options on 9 Oct.
* 21 Sep is the sell off. Delta, vega and vanna are all large and the moves are big enough that third order effects show up.

New trades added about $39k over 393 fills. It's noisy day to day because fills happen intraday and are marked at the close, so the intraday spot move gets mixed in with the edge.

## What it doesn't do (yet)

* no carry on futures, no rates or dividend P&L line
* skew is linear and its slope doesn't move, so there is no skew P&L bucket
* new trades are only held from fill to close on their first day, any P&L after that lands in the normal greeks buckets the next day

---

If something here is unclear, broken, or just wrong, feel free to open an issue. I'm always happy to chat about it.
