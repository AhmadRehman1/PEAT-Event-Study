from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


UNIVERSE = {

    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ORCL": "XLK",
    "CRM": "XLK", "ADBE": "XLK", "CSCO": "XLK", "AMD": "XLK", "QCOM": "XLK",

    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",

    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",

    "WMT": "XLP", "PG": "XLP", "KO": "XLP", "COST": "XLP",

    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",

    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "V": "XLF", "MA": "XLF",

    "CAT": "XLI", "HON": "XLI", "GE": "XLI", "UPS": "XLI",

    "XOM": "XLE", "CVX": "XLE",
}
BENCH = "SPY"

YEARS_BACK = 5
BEAT_THRESHOLD = 0.02
MISS_THRESHOLD = -0.02
MIN_ABS_ESTIMATE = 0.10

HORIZONS = [1, 5, 10, 20]
PRE_DAYS = 5
POST_DAYS = max(HORIZONS)
EST_WINDOW = (-250, -31)
MIN_EST_OBS = 120
AFTER_CLOSE_HOUR = 16.0
OPEN_HOUR = 9.5

METHODS = {
    "mm": "market model (beta-adjusted S&P 500)  [PRIMARY]",
    "ma": "market-adjusted (stock minus S&P 500)",
    "sa": "sector-adjusted (stock minus sector ETF)",
}
PRIMARY = "mm"


BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_prices(symbols, start, end, cache: Path, refresh: bool) -> pd.DataFrame:
    """Daily ADJUSTED closing prices (adjusted for splits and dividends)."""
    if cache.exists() and not refresh:
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        if all(s in px.columns for s in symbols):
            return px
    import yfinance as yf

    print(f"Downloading prices for {len(symbols)} symbols from Yahoo Finance ...")
    raw = yf.download(symbols, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no price data (offline, or Yahoo blocked the request).")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.dropna(how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px


def load_earnings_yfinance(tickers, cache: Path, refresh: bool) -> pd.DataFrame:
    """Earnings history from yfinance: date+time, consensus estimate, reported EPS."""
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["announce_date"])
    import yfinance as yf

    frames, weak = [], []
    for tk in tickers:
        try:
            raw = yf.Ticker(tk).get_earnings_dates(limit=40)
        except Exception as exc:
            print(f"  ! {tk}: could not fetch earnings dates ({type(exc).__name__}: {str(exc)[:80]})")
            weak.append(tk)
            continue
        if raw is None or len(raw) == 0:
            weak.append(tk)
            continue
        df = raw.reset_index()
        est_col = next((c for c in df.columns if "estimate" in str(c).lower()), None)
        act_col = next((c for c in df.columns if "reported" in str(c).lower()), None)
        if est_col is None or act_col is None:
            print(f"  ! {tk}: unexpected columns {list(df.columns)}")
            weak.append(tk)
            continue
        ts = pd.to_datetime(df.iloc[:, 0])
        ts = ts.dt.tz_localize("America/New_York") if ts.dt.tz is None else ts.dt.tz_convert("America/New_York")
        hour = ts.dt.hour + ts.dt.minute / 60.0
        hour[(ts.dt.hour == 0) & (ts.dt.minute == 0)] = np.nan
        frames.append(pd.DataFrame({
            "ticker": tk,
            "announce_date": ts.dt.tz_localize(None).dt.normalize(),
            "announce_hour": hour,
            "eps_est": pd.to_numeric(df[est_col], errors="coerce"),
            "eps_act": pd.to_numeric(df[act_col], errors="coerce"),
        }))
    if not frames:
        raise RuntimeError("No earnings data could be fetched from yfinance.")
    out = pd.concat(frames, ignore_index=True)
    used = out.dropna(subset=["eps_est", "eps_act"]).groupby("ticker").size()
    print(f"yfinance coverage: {len(used)}/{len(tickers)} tickers with usable reports, "
          f"median {int(used.median())} reports per ticker (min {int(used.min())}, max {int(used.max())}).")
    if weak:
        print(f"  no/weak coverage for: {', '.join(weak)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def load_earnings_alphavantage(tickers, cache: Path, refresh: bool) -> pd.DataFrame:
    """Fallback source. Free key: https://www.alphavantage.co/support/ (get a free key there)
    The free tier allows ~25 requests/day, so 40 tickers needs two days (results are cached)."""
    import requests

    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("Set the ALPHAVANTAGE_API_KEY environment variable to use --source alphavantage.")
    have = pd.read_csv(cache, parse_dates=["announce_date"]) if cache.exists() and not refresh else pd.DataFrame()
    done = set(have["ticker"]) if len(have) else set()
    frames = [have] if len(have) else []
    for tk in tickers:
        if tk in done:
            continue
        js = requests.get("https://www.alphavantage.co/query",
                          params={"function": "EARNINGS", "symbol": tk, "apikey": key}, timeout=30).json()
        if "quarterlyEarnings" not in js:
            print(f"  ! stopping at {tk}: {str(js)[:120]}  (daily limit? re-run tomorrow, cache is kept)")
            break
        rows = []
        for q in js["quarterlyEarnings"]:
            rt = str(q.get("reportTime", "")).lower()
            hour = 8.0 if "pre" in rt else 17.0 if "post" in rt else np.nan
            rows.append({"ticker": tk, "announce_date": pd.to_datetime(q.get("reportedDate")),
                         "announce_hour": hour,
                         "eps_est": pd.to_numeric(q.get("estimatedEPS"), errors="coerce"),
                         "eps_act": pd.to_numeric(q.get("reportedEPS"), errors="coerce")})
        frames.append(pd.DataFrame(rows))
        time.sleep(1.2)
    if not frames:
        raise RuntimeError("Alpha Vantage returned nothing.")
    out = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def load_earnings_csv(path: str) -> pd.DataFrame:
    """Your own file. Columns: ticker, announce_date, eps_est, eps_act, [announce_hour]."""
    df = pd.read_csv(path, parse_dates=["announce_date"])
    if "announce_hour" not in df.columns:
        df["announce_hour"] = np.nan
    return df[["ticker", "announce_date", "announce_hour", "eps_est", "eps_act"]]


def make_synthetic(planted_drift: float, seed: int):
    """Fake market with a planted drift of `planted_drift` per day for 20 days after a
    beat (and the opposite after a miss). With planted_drift=0 there is NO effect, so a
    correct pipeline must find nothing. This tests the machinery, not the market."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-06-01", "2026-09-18")
    n = len(cal)
    mkt = rng.normal(0.0004, 0.010, n)
    rets = {BENCH: mkt}
    for sec in sorted(set(UNIVERSE.values())):
        rets[sec] = mkt + rng.normal(0, 0.004, n)
    rows = []
    for tk in UNIVERSE:
        beta = rng.uniform(0.7, 1.4)
        r = beta * mkt + rng.normal(0, 0.012, n)
        pos = int(rng.integers(30, 80))
        while pos < n - 30:
            after_close = rng.random() < 0.6
            est = rng.uniform(0.5, 3.0)
            surprise = rng.normal(0.04, 0.10)
            day0 = pos + 1 if after_close else pos
            r[day0] += 0.30 * np.clip(surprise, -0.3, 0.3) + rng.normal(0, 0.02)
            if abs(surprise) >= BEAT_THRESHOLD:
                r[day0 + 1: day0 + 1 + POST_DAYS] += np.sign(surprise) * planted_drift
            rows.append({"ticker": tk, "announce_date": cal[pos],
                         "announce_hour": 16.25 if after_close else 7.0,
                         "eps_est": est, "eps_act": est * (1 + surprise)})
            pos += int(rng.integers(60, 66))
        rets[tk] = r
    px = pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in rets.items()}, index=cal)
    return px, pd.DataFrame(rows)


def classify_events(earn: pd.DataFrame, beat_th: float, miss_th: float, audit: Counter) -> pd.DataFrame:
    """Surprise = (actual - estimate) / |estimate|.  beat > +2%, miss < -2%, else in-line."""
    e = earn.drop_duplicates(subset=["ticker", "announce_date"]).copy()
    audit["raw earnings rows"] = len(e)
    e = e.dropna(subset=["eps_est", "eps_act"])
    audit["dropped: missing estimate or actual (incl. future reports)"] = audit["raw earnings rows"] - len(e)
    small = e["eps_est"].abs() < MIN_ABS_ESTIMATE
    audit[f"dropped: |consensus EPS| < ${MIN_ABS_ESTIMATE:.2f}"] = int(small.sum())
    e = e[~small].copy()

    e["surprise"] = (e["eps_act"] - e["eps_est"]) / e["eps_est"].abs()
    e["label"] = np.where(e["surprise"] > beat_th, "beat", np.where(e["surprise"] < miss_th, "miss", "inline"))
    return e


def assign_day0(e: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Day 0 = the first trading day whose CLOSE already reflects the announcement.
       before open / during hours -> same day.   after close (or time unknown) -> next trading day."""
    e = e.copy()
    hr = e["announce_hour"]
    after = hr.isna() | (hr >= AFTER_CLOSE_HOUR)
    e["timing"] = np.where(hr.isna(), "unknown (assumed after close)",
                   np.where(hr >= AFTER_CLOSE_HOUR, "after close",
                   np.where(hr < OPEN_HOUR, "before open", "during hours")))
    pos = []
    for d, a in zip(e["announce_date"], after):
        pos.append(cal.searchsorted(d, side="right" if a else "left"))
    e["day0_pos"] = pos
    return e


def compute_abnormal_returns(events: pd.DataFrame, rets: pd.DataFrame, cal: pd.DatetimeIndex, audit: Counter):
    """For each event build the abnormal-return path from day -PRE_DAYS to day +POST_DAYS.

    abnormal return (AR) on a day = actual stock return - "normal" return
    cumulative abnormal return (CAR) = AR added up over the days of the window.
    """
    idx = {t: i for i, t in enumerate(range(-PRE_DAYS, POST_DAYS + 1))}
    bench = rets[BENCH].to_numpy()
    keep, out_rows, paths = [], [], {m: [] for m in METHODS}
    for i, ev in events.iterrows():
        p0 = int(ev["day0_pos"])
        lo_est, hi_est = p0 + EST_WINDOW[0], p0 + EST_WINDOW[1]
        lo_ev, hi_ev = p0 - PRE_DAYS, p0 + POST_DAYS
        if hi_ev >= len(cal):
            audit["dropped: fewer than 20 trading days of data after day 0"] += 1
            continue
        if lo_est < 1:
            audit["dropped: not enough price history before the event"] += 1
            continue
        assert hi_est < lo_ev, "LOOKAHEAD GUARD: estimation window must end before the event window starts"
        s = rets[ev["ticker"]].to_numpy()
        sec = rets[UNIVERSE[ev["ticker"]]].to_numpy()


        y, x = s[lo_est:hi_est + 1], bench[lo_est:hi_est + 1]
        ok = ~np.isnan(y) & ~np.isnan(x)
        if ok.sum() < MIN_EST_OBS:
            audit["dropped: too few valid days to estimate beta"] += 1
            continue
        yv, xv = y[ok], x[ok]
        beta = np.cov(xv, yv, ddof=1)[0, 1] / np.var(xv, ddof=1)
        alpha = yv.mean() - beta * xv.mean()

        ys, xs, ss = s[lo_ev:hi_ev + 1], bench[lo_ev:hi_ev + 1], sec[lo_ev:hi_ev + 1]
        if np.isnan(ys).any() or np.isnan(xs).any() or np.isnan(ss).any():
            audit["dropped: missing prices inside the event window"] += 1
            continue
        ar = {"mm": ys - (alpha + beta * xs), "ma": ys - xs, "sa": ys - ss}

        row = {"beta": beta}
        for m, a in ar.items():
            cum = np.cumsum(a)
            rebased = cum - cum[idx[-1]]
            paths[m].append(rebased)
            for k in HORIZONS:
                row[f"{m}_car0_{k}"] = rebased[idx[k]]
                row[f"{m}_car1_{k}"] = rebased[idx[k]] - rebased[idx[0]]
        keep.append(i)
        out_rows.append(row)
    ev2 = events.loc[keep].copy()
    ev2 = pd.concat([ev2.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)
    ev2["day0_date"] = cal[ev2["day0_pos"].astype(int)]
    paths = {m: np.vstack(v) for m, v in paths.items()}
    return ev2, paths


def t_one(x):
    """One-sample t-test: is the average different from zero?"""
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


def t_diff(a, b):
    """Welch two-sample t-test: are the two averages different from EACH OTHER?"""
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def summary_table(ev: pd.DataFrame, method: str, start: int) -> pd.DataFrame:
    rows = []
    for k in HORIZONS:
        col = f"{method}_car{start}_{k}"
        b = ev.loc[ev["label"] == "beat", col].to_numpy()
        m = ev.loc[ev["label"] == "miss", col].to_numpy()
        tb, pb = t_one(b)
        tm, pm = t_one(m)
        td, pd_ = t_diff(b, m)
        rows.append({"window": f"CAR[{start},{k}]", "k": k,
                     "n_beat": len(b), "mean_beat": b.mean(), "median_beat": np.median(b), "t_beat": tb, "p_beat": pb,
                     "n_miss": len(m), "mean_miss": m.mean(), "median_miss": np.median(m), "t_miss": tm, "p_miss": pm,
                     "spread": b.mean() - m.mean(), "t_spread": td, "p_spread": pd_})
    return pd.DataFrame(rows)


def season_test(ev: pd.DataFrame, col: str) -> dict:
    """Clustered check. Events in the same earnings season share market shocks, so they are not
    fully independent. Average within each calendar quarter first, then t-test across quarters."""
    d = ev[ev["label"].isin(["beat", "miss"])].copy()
    d["season"] = d["day0_date"].dt.to_period("Q")
    pv = d.groupby(["season", "label"])[col].mean().unstack()
    res = {"n_seasons": int(len(pv))}
    for g in ("beat", "miss"):
        if g in pv:
            res[f"mean_{g}"] = pv[g].mean()
            res[f"t_{g}"], res[f"p_{g}"] = t_one(pv[g].dropna())
    if {"beat", "miss"} <= set(pv.columns):
        sp = (pv["beat"] - pv["miss"]).dropna()
        res["mean_spread"] = sp.mean()
        res["t_spread"], res["p_spread"] = t_one(sp)
        res["n_seasons"] = int(len(sp))
    return res


def robustness_checks(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, variant, col, b, m):
        t, p = t_diff(b, m)
        rows.append({"check": check, "variant": variant, "window": col.split("_", 1)[1],
                     "n_beat": len(b), "n_miss": len(m), "mean_beat": np.mean(b), "mean_miss": np.mean(m),
                     "spread": np.mean(b) - np.mean(m), "t_spread": t, "p_spread": p})


    for meth, desc in METHODS.items():
        for k in (5, 20):
            col = f"{meth}_car1_{k}"
            add("abnormal-return definition", desc.split("  ")[0], col,
                ev.loc[ev.label == "beat", col].to_numpy(), ev.loc[ev.label == "miss", col].to_numpy())

    col = f"{PRIMARY}_car1_20"
    for th in (0.02, 0.05, 0.10):
        add("beat/miss threshold", f"+/-{th:.0%}", col,
            ev.loc[ev.surprise > th, col].to_numpy(), ev.loc[ev.surprise < -th, col].to_numpy())

    cut = ev["day0_date"].sort_values().iloc[len(ev) // 2]
    for name, sub in (("first half", ev[ev.day0_date < cut]), ("second half", ev[ev.day0_date >= cut])):
        add("sub-period", f"{name} (to/from {cut.date()})", col,
            sub.loc[sub.label == "beat", col].to_numpy(), sub.loc[sub.label == "miss", col].to_numpy())

    known = ev[~ev.timing.str.startswith("unknown")]
    add("timing", "only events with a known announcement time", col,
        known.loc[known.label == "beat", col].to_numpy(), known.loc[known.label == "miss", col].to_numpy())
    return pd.DataFrame(rows)


def make_chart(paths: np.ndarray, ev: pd.DataFrame, out_png: Path, watermark: str | None):
    T = np.arange(-PRE_DAYS, POST_DAYS + 1)
    lab = ev["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = {}
    for label, color, name in (("beat", BLUE, "Beats"), ("miss", ORANGE, "Misses")):
        P = paths[lab == label] * 100
        n = len(P)
        mean = P.mean(axis=0)
        se = P.std(axis=0, ddof=1) / np.sqrt(n)
        ax.fill_between(T, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.12, lw=0)
        ax.plot(T, mean, color=color, lw=2, solid_capstyle="round", label=name)
        ax.plot([T[-1]], [mean[-1]], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
        ends[label] = (mean, n)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.25, ymax, "Day 0: first close that\nreflects the results", ha="left", va="top", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.set_xlabel("Trading days relative to the earnings reaction (day 0)", color=INK2, fontsize=10.5, labelpad=8)
    ax.set_ylabel("Cumulative abnormal return", color=INK2, fontsize=10.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    ax.set_xticks(range(-PRE_DAYS, POST_DAYS + 1, 5))
    ax.set_xlim(-PRE_DAYS, POST_DAYS + 3.5)

    (mb, nb), (mm_, nm) = ends["beat"], ends["miss"]
    i0 = PRE_DAYS
    for mean, txt in ((mb, f"Beats {mb[-1]:+.2f}%"), (mm_, f"Misses {mm_[-1]:+.2f}%")):
        ax.text(POST_DAYS + 0.6, mean[-1], txt, va="center", ha="left", fontsize=10, color=INK)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=10, labelcolor=INK2)
    for h in leg.get_lines():
        h.set_linewidth(2)

    fig.text(0.075, 0.955, "After an earnings surprise, do stocks keep drifting?", fontsize=15,
             fontweight="semibold", color=INK, ha="left")
    drift_b = (mb[-1] - mb[i0]); drift_m = (mm_[-1] - mm_[i0])
    fig.text(0.075, 0.915,
             f"Average cumulative abnormal return. Drift after day 0 (days +1 to +{POST_DAYS}): "
             f"beats {drift_b:+.2f}%, misses {drift_m:+.2f}%", fontsize=10.5, color=INK2, ha="left")
    fig.text(0.075, 0.015,
             "Abnormal return = stock return minus beta-adjusted S&P 500 return (beta estimated only from data before the event).\n"
             f"Shaded bands = 95% confidence interval of the average. n = {nb} beats, {nm} misses; in-line reports (|surprise| <= {BEAT_THRESHOLD:.0%}) excluded.",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=30, color="#b0aea5", alpha=0.35, ha="center", va="center", rotation=25)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.86, bottom=0.17)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def pct(x, d=2):
    return "n/a" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def pv(p):
    return "n/a" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def peq(p):
    """'p = 0.012' or 'p < 0.001' for use inside sentences."""
    return "p n/a" if pd.isna(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def stars(p):
    return "" if pd.isna(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def results_markdown(ev, tab1, tab0, seas, robust, audit, universe_n, run_label, cfg) -> str:
    n_beat, n_miss = int((ev.label == "beat").sum()), int((ev.label == "miss").sum())
    n_in = int((ev.label == "inline").sum())
    n_co = ev.loc[ev.label.isin(["beat", "miss"]), "ticker"].nunique()
    d0, d1 = ev.day0_date.min().date(), ev.day0_date.max().date()
    L = []
    L.append(f"_{run_label}_\n")
    L.append(f"**Sample: {n_beat + n_miss} usable events ({n_beat} beats, {n_miss} misses) across {n_co} companies** "
             f"(of {universe_n} in the universe), reactions from {d0} to {d1}. "
             f"A further {n_in} in-line reports (|surprise| <= {cfg['beat']:.0%}) were excluded by design.\n")

    def block(tab, title, note):
        L.append(f"**{title}**  \n{note}\n")
        L.append("| Window | Beats: avg | t-stat | p | Misses: avg | t-stat | p | Beat minus miss | t-stat | p |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in tab.iterrows():
            L.append(f"| {r.window} | {pct(r.mean_beat)} | {r.t_beat:.2f} | {pv(r.p_beat)}{stars(r.p_beat)} | "
                     f"{pct(r.mean_miss)} | {r.t_miss:.2f} | {pv(r.p_miss)}{stars(r.p_miss)} | "
                     f"{pct(r.spread)} | {r.t_spread:.2f} | {pv(r.p_spread)}{stars(r.p_spread)} |")
        L.append("")

    block(tab1, "Table 1 - Post-announcement DRIFT (starts at the close of day 0; the headline result)",
          "Average cumulative abnormal return from the day after the reaction. `***` p<0.01, `**` p<0.05, `*` p<0.10.")
    block(tab0, "Table 2 - Full reaction window (includes the day-0 jump)",
          "Shown for context: this mostly measures the immediate price reaction, which is NOT the anomaly.")

    r = tab1[tab1.k == POST_DAYS].iloc[0]
    L.append("**Plain-English verdict (generated automatically from the numbers above)**\n")
    v = []
    v.append(f"Over the 20 trading days after the reaction, beats moved {pct(r.mean_beat)} versus the market-adjusted "
             f"norm (t = {r.t_beat:.2f}, {peq(r.p_beat)}) - "
             f"{'statistically distinguishable from zero' if r.p_beat < 0.05 else 'NOT statistically distinguishable from zero'} at the 5% level.")
    v.append(f"Misses moved {pct(r.mean_miss)} (t = {r.t_miss:.2f}, {peq(r.p_miss)}) - "
             f"{'statistically distinguishable from zero' if r.p_miss < 0.05 else 'NOT statistically distinguishable from zero'} at the 5% level.")
    v.append(f"The beat-minus-miss gap was {pct(r.spread)} (t = {r.t_spread:.2f}, {peq(r.p_spread)}), i.e. "
             f"{'the two groups do drift differently' if r.p_spread < 0.05 and r.spread > 0 else 'we cannot conclude the two groups drift differently'} at the 5% level.")
    sp = seas
    if not pd.isna(sp.get("p_spread", np.nan)):
        v.append(f"Robustness - clustering by earnings season ({sp['n_seasons']} quarters): the 20-day gap is "
                 f"{pct(sp['mean_spread'])} (t = {sp['t_spread']:.2f}, {peq(sp['p_spread'])}) - "
                 f"{'still significant' if sp['p_spread'] < 0.05 else 'not significant'} at 5%.")
    L.append("\n".join(f"- {s}" for s in v) + "\n")

    L.append("**Robustness checks** (20-day and 5-day drift, beat minus miss)\n")
    L.append("| Check | Variant | Window | n beats | n misses | Beat minus miss | t-stat | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, x in robust.iterrows():
        L.append(f"| {x.check} | {x.variant} | {x.window} | {x.n_beat} | {x.n_miss} | {pct(x.spread)} | "
                 f"{x.t_spread:.2f} | {pv(x.p_spread)}{stars(x.p_spread)} |")
    L.append("")
    L.append("**Data audit** (how many events were removed, and why)\n")
    for k, v_ in audit.items():
        L.append(f"- {k}: {v_}")
    L.append("")
    return "\n".join(L)


def inject_into_readme(readme: Path, md: str):
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not readme.exists():
        return False
    txt = readme.read_text(encoding="utf-8")
    if a not in txt or b not in txt:
        return False
    head, rest = txt.split(a, 1)
    _, tail = rest.split(b, 1)
    readme.write_text(f"{head}{a}\n{md}\n{b}{tail}", encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["yfinance", "alphavantage", "csv"], default="yfinance",
                    help="where earnings data comes from (prices always come from yfinance)")
    ap.add_argument("--earnings-csv", help="path to your own earnings CSV (with --source csv)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached data and re-download")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data with a planted drift")
    ap.add_argument("--planted-drift", type=float, default=0.0005, help="selftest: daily drift planted after beats/misses")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beat-threshold", type=float, default=BEAT_THRESHOLD)
    ap.add_argument("--no-readme", action="store_true", help="do not write results into README.md")
    args = ap.parse_args()

    beat_th, miss_th = args.beat_threshold, -args.beat_threshold
    audit: Counter = Counter()
    out_dir = ROOT / ("results_selftest" if args.selftest else "results")
    out_dir.mkdir(exist_ok=True)


    if args.selftest:
        print(f"SELF-TEST: synthetic data, planted drift = {args.planted_drift:+.4%} per day for {POST_DAYS} days.")
        px, earn = make_synthetic(args.planted_drift, args.seed)
        today = px.index[-1].date()
    else:
        today = date.today()
        symbols = list(UNIVERSE) + [BENCH] + sorted(set(UNIVERSE.values()))
        start_px = today - timedelta(days=int(365.25 * YEARS_BACK) + 500)
        px = load_prices(symbols, start_px, today + timedelta(days=1), DATA_DIR / "prices.csv", args.refresh)
        if args.source == "yfinance":
            earn = load_earnings_yfinance(list(UNIVERSE), DATA_DIR / "earnings_yfinance.csv", args.refresh)
        elif args.source == "alphavantage":
            earn = load_earnings_alphavantage(list(UNIVERSE), DATA_DIR / "earnings_alphavantage.csv", args.refresh)
        else:
            earn = load_earnings_csv(args.earnings_csv)
    event_start = pd.Timestamp(today) - pd.DateOffset(years=YEARS_BACK)
    earn = earn[(earn["announce_date"] >= event_start) & (earn["announce_date"] <= pd.Timestamp(today))]
    earn = earn[earn["ticker"].isin(UNIVERSE)]


    cal = px[BENCH].dropna().index
    rets = px.reindex(cal).pct_change(fill_method=None)


    events = classify_events(earn, beat_th, miss_th, audit)
    events = assign_day0(events, cal)
    ev, paths = compute_abnormal_returns(events, rets, cal, audit)
    if ev.empty:
        sys.exit("No usable events - check the data audit above.")
    audit["events kept for analysis (all labels)"] = len(ev)
    print("\nDATA AUDIT")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    print("\nTiming of announcements:", {k: int(v) for k, v in ev["timing"].value_counts().items()})
    print("Labels:", {k: int(v) for k, v in ev["label"].value_counts().items()})


    tab1 = summary_table(ev, PRIMARY, start=1)
    tab0 = summary_table(ev, PRIMARY, start=0)
    seas = season_test(ev, f"{PRIMARY}_car1_{POST_DAYS}")
    robust = robustness_checks(ev)

    ev.drop(columns=["day0_pos"]).to_csv(out_dir / "event_level_results.csv", index=False)
    tab1.to_csv(out_dir / "summary_drift_car1_k.csv", index=False)
    tab0.to_csv(out_dir / "summary_full_window_car0_k.csv", index=False)
    robust.to_csv(out_dir / "robustness_checks.csv", index=False)

    label = ("SYNTHETIC SELF-TEST - NOT REAL MARKET DATA" if args.selftest
             else f"Generated {today}. Source: yfinance prices, {args.source} earnings.")
    md = results_markdown(ev, tab1, tab0, seas, robust, audit, len(UNIVERSE), label,
                          {"beat": beat_th})
    (out_dir / "results_summary.md").write_text(md, encoding="utf-8")
    make_chart(paths[PRIMARY][(ev.label.isin(["beat", "miss"])).to_numpy()],
               ev[ev.label.isin(["beat", "miss"])], out_dir / "car_beats_vs_misses.png",
               "SYNTHETIC SELF-TEST\nNOT REAL DATA" if args.selftest else None)
    print("\n" + md)
    if not args.selftest and not args.no_readme:
        if inject_into_readme(ROOT / "README.md", md):
            print("README.md results section updated.")
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()