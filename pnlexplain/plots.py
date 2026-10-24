import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOURS = {"delta": "#4c72b0", "gamma": "#55a868", "vega": "#c44e52", "theta": "#8172b2",
           "vanna": "#ccb974", "volga": "#64b5cd", "unexplained": "#999999", "new_trades": "#dd8452"}


def explain_chart(df, path):
    parts = [c for c in COLOURS if c in df.columns and df[c].abs().sum() > 0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(df))
    pos = np.zeros(len(df))
    neg = np.zeros(len(df))
    for c in parts:
        v = df[c].to_numpy() / 1000
        ax1.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COLOURS[c], label=c, width=0.8)
        ax1.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COLOURS[c], width=0.8)
        pos += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax1.plot(x, df["total"] / 1000, "k.", ms=4, label="total")
    for i in np.where(df["flag"])[0]:
        ax1.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)
    ax1.set_ylabel("USD 000s")
    ax1.legend(ncol=5, fontsize=8, loc="upper left")
    ax1.set_title("Daily P&L explain (flagged days shaded)")

    for c in ["actual", "unexplained", "new_trades"]:
        ax2.plot(x, df[c].cumsum() / 1000, label=f"cumulative {c}")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(fontsize=8)
    ax2.set_ylabel("USD 000s")
    ticks = x[:: max(len(x) // 10, 1)]
    ax2.set_xticks(ticks, [df.index[i].strftime("%d %b") for i in ticks])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
