---
title: NSE Daily + Intraday — Data Understanding, Feature Engineering & Model Selection
date: 2026-10-08
idea: Feature-engineer the NSE bhavcopy (daily, 2000–2026) and the HF 1-minute dataset (2022–2026) into a tabular ML matrix, then train/tune a model to predict next-bar return, direction, and realized volatility.
verdict: GO on the pipeline, CONDITIONAL on the target — the data is rich and clean enough to engineer a strong feature matrix, but the single highest-value "feature engineering" action is fixing the raw-vs-adjusted price mismatch. Direction is near the noise floor; volatility and meta-labeling are the tractable targets.
confidence: High on data facts (directly inspected) and on model family choice; Med on achievable directional edge.
---

# NSE Daily + Intraday — Data, Features & Model Selection

> **How to read this.** This is the technical companion to
> [`~/research/nse-intraday-ml-forecasting.md`](./nse-intraday-ml-forecasting.md).
> Every data fact below was obtained by **inspecting the actual Parquet files in
> this repo on 2026-10-08** (schemas, row counts, distributions, specific rows) —
> not from documentation. External claims are tagged `[Fact|Opinion|Inference|Estimate, High|Med|Low]`
> with a URL. "Raw" file statistics are tagged `[Fact, High — inspected]`.
> Local paths are shown as `data/...` relative to `/home/samarth/thense`.

## Table of contents
1. [TL;DR — the 15 things that matter](#1-tldr)
2. [What is actually in the data](#2-what-is-actually-in-the-data)
3. [Data-quality landmines (fix these first)](#3-data-quality-landmines)
4. [How much can we improve the input? (honest answer)](#4-how-much-can-we-improve-the-input)
5. [Feature engineering blueprint — daily layer](#5-feature-engineering-blueprint--daily-layer)
6. [Feature engineering blueprint — intraday (1-min) layer](#6-feature-engineering-blueprint--intraday-1-min-layer)
7. [Cross-asset, cross-sectional & calendar features](#7-cross-asset-cross-sectional--calendar-features)
8. [Targets & labels (this choice matters more than the model)](#8-targets--labels)
9. [Which model is "perfect" — model selection](#9-which-model-is-perfect--model-selection)
10. [Tuning: GridSearchCV vs RandomizedSearch vs Optuna](#10-tuning-gridsearchcv-vs-randomizedsearch-vs-optuna)
11. [Validation protocol (non-negotiable)](#11-validation-protocol)
12. [Concrete build plan & next steps](#12-concrete-build-plan--next-steps)
13. [Risks & unknowns](#13-risks--unknowns)
14. [Sources](#14-sources)

---

## 1. TL;DR

1. **Two datasets, two different worlds:** `data/processed/bhavcopy/bhavcopy.parquet`
   is **10,672,269 rows / 8,897 symbols / 6,625 days / 2000-01-03 → 2026-10-07 / 295 MB**,
   and the HF minute data is **713,452,456 rows / 2,535 symbols / 2022-01-03 → 2026-01-21 / 9.7 GB (8 shards)**. `[Fact, High — inspected]`
2. **They do NOT agree on prices.** Bhavcopy is **raw/unadjusted**; the HF minute *and*
   daily files are **corporate-action back-adjusted**. On Reliance's 1:1 bonus
   (ex-date 2024-10-28) bhavcopy shows close 1334.35 vs prev_close 2655.70 → a
   **fake −49.76% return**, while the HF minute close that day is 1335.00 (already adjusted). `[Fact, High — inspected]`
3. **Fixing that mismatch is the single biggest "feature engineering" win available.**
   In bhavcopy `EQ` alone there are **2,180 stock-days with a >2× or <0.5× overnight ratio** (clear corporate actions) and **7,949 with a >25% move**; leaving them in teaches the model fake crashes and gap-ups. `[Fact, High — inspected]`
4. **The 1-minute data is an almost-perfect grid:** Reliance has **994 days at exactly 375 bars (09:15–15:29 IST)**; NIFTY has 906 such days. It is **left-labeled, session-aligned, no pre-open, OI always 0 for equities**. `[Fact, High — inspected]`
5. **But it has off-grid contamination:** NIFTY 1-min contains bars at 09:07 (84), 15:30 (46), 15:31 (84), 15:39 (5), 15:59 (5) and evening bars 18:00–19:14 on a handful of dates — **404 of 375,667 bars (~0.11%) off-grid**. Must be filtered and re-indexed onto a canonical 375-slot grid. `[Fact, High — inspected]`
6. **Special sessions exist:** 60-bar days are Diwali **Muhurat** sessions (2022-10-24, 2023-11-12, 2024-11-01, 2025-10-21); 105-bar days are special Saturday sessions (2024-03-02, 2024-05-18). These need `is_special_session` flags, not silent inclusion. `[Fact, High — inspected]`
7. **Minute data is survivorship-biased; bhavcopy is the point-in-time truth.** Minute/day HF = 2,535 currently-active symbols; bhavcopy = 8,897 including delisted names. Use bhavcopy for cross-sectional universe, HF for intraday history. `[Fact, High — inspected]`
8. **Feature target: ~300–600 engineered features is realistic**, benchmarked against a shipped industry baseline of **Alpha158 (158 features)** and **Alpha360 (360 raw)** from Microsoft Qlib. `[Fact, High]` [Qlib loader.py](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
9. **The "perfect" model is not a neural net.** The evidence is overwhelming that **gradient-boosted trees (LightGBM first, CatBoost/XGBoost close)** win on tabular financial features at this scale. Tree models beat deep nets across 45 tabular benchmarks, tuning doesn't close the gap, and Qlib's own benchmark shows LightGBM IC 0.0448 vs TFT IC 0.0358 on Alpha158. `[Fact, High]` [Grinsztajn 2022](https://arxiv.org/abs/2207.08815) · [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
10. **Model #1 = LightGBM** (fast, robust to uninformative features, native categorical + missing handling, best-in-class on noisy tabular data). **Model #2 = a sequence net (TCN/GRU or iTransformer)** only on the *raw sequence* branch, then **ensemble**. This mirrors what won Jane Street 2020 / Ubiquant 2022 (GBDT + NN blends). `[Fact, High]` [Jane Street 1st](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) · [Ubiquant](https://zizouhe.github.io/reading-notes/notes/kaggle-fin.html)
11. **Zero-shot time-series foundation models (Chronos/TimesFM/Moirai) are not worth it as a primary signal.** They beat a random walk only "small and sparse" on returns and generally fail to beat a Log-HAR volatility baseline. `[Fact, Med]` [TSFM for returns](https://arxiv.org/html/2606.27100v1) · [RV with TSFMs](https://arxiv.org/pdf/2607.05291)
12. **Do NOT use plain `GridSearchCV`.** Random search beats grid search (Bergstra & Bengio), and Optuna TPE + a pruner decisively beats both under a fixed budget (with pruning it explored ~1,278 trials vs ~36). Use **Optuna TPE + MedianPruner/ASHA with a purged-CV objective**. `[Fact, High]` [Bergstra 2012](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) · [Optuna](https://arxiv.org/abs/1907.10902)
13. **Accuracy is a lie for direction.** An always-up rule hit ~80% in a rising market. Report **AUC, Rank IC / ICIR, log-loss, precision@top-k and net-of-cost Sharpe** instead. `[Fact, Med]` [When Directional Accuracy Lies](https://arxiv.org/abs/2607.12248v2)
14. **Expected "improvement from features" is real but bounded.** Volatility forecasts are genuinely predictable (HAR-RV class); directional IC on liquid NSE names realistically lands ~0.01–0.05 and *often* does not survive ~6–12 bps round-trip NSE costs. Feature engineering raises the ceiling on *volatility/risk* far more than on *direction*. `[Inference, High]`
15. **No modeling library is installed yet.** `sklearn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `shap`, `statsmodels`, `arch`, `fracdiff`, `sktime`, `torch` are all **MISSING** from `.venv`. Add them before Phase 1. `[Fact, High — inspected]`

---

## 2. What is actually in the data

Everything in this section was read from the files with PyArrow/DuckDB metadata and
targeted queries. Full-data scans were avoided; schema + row-group statistics + a
handful of filtered queries were used, exactly as requested.

### 2.1 Inventory

| Dataset | Path | Rows | Size | Coverage | Symbols |
|---|---|---|---:|---|---:|
| Bhavcopy (consolidated) | `data/processed/bhavcopy/bhavcopy.parquet` | **10,672,269** | 295 MB | 2000-01-03 → 2026-10-07 (6,625 days) | 8,897 |
| Bhavcopy (per-day raw) | `data/raw/bhavcopy/<year>/<date>.parquet` | — | 723 MB | same | — |
| HF minute | `data/raw/hf/xxparthparekhxx__.../minute/train-00*.parquet` | **713,452,456** | 9.7 GB (8 shards) | 2022-01-03 → 2026-01-21 | 2,535 |
| HF daily | `data/raw/hf/xxparthparekhxx__.../day/train-00000.parquet` | **6,968,616** | 118 MB | 2000-01-01 → 2026-01-22 | 2,535 |
| Indices / macro | `data/raw/indices/*.parquet` | 3–25k each | 3 MB | see §2.4 | 10 series |
| Security master | `data/reference/security_master.parquet` | 2,599 | 0.1 MB | current listed set | 2,599 |
| Options (partial) | `data/raw/hf/rissin__nse-options-intraday/` | incomplete | 109 MB | NIFTY F&O daily 2001→; 1-min partial | — |

`[Fact, High — inspected]`

### 2.2 Bhavcopy schema and semantics

```
date: timestamp[us], symbol: string, series: string, isin: string,
open, high, low, close, last, prev_close: double,
volume: int64, turnover: double, trades: int64, source_format: string
```

- **Row counts grow with market participation:** 246,681 rows (2000, 1,265 symbols)
  → 760,926 rows (2025, 4,538 symbols). `[Fact, High — inspected]`
- **`series` is a category, not a filter to ignore.** Top: `EQ` 8,777,698 rows /
  5,036 symbols, then `BE` 768,163, `SM` 330,594, `GB`, `BZ`, `ST`, `GS`, and many
  `N1..N9`, `TB` debt entries. **Restrict the modelling universe to `EQ`
  (and optionally `BE`) to avoid bonds, ETFs and odd lots.** `[Fact, High — inspected]`
- **Missingness maps to schema eras:** `isin` and `trades` are null for exactly
  **2,909,277 rows** — the pre-~2010 legacy format. `close`, `volume`, `prev_close`
  have **zero** nulls; `high < low` never occurs; no zero/negative closes. Very clean
  on the mechanical side. `[Fact, High — inspected]`
- **`source_format` ∈ {old, new}`.** The 2024→ UDiFF files carry ISIN/trades; older
  files don't. `[Fact, High — inspected]`
- **`prev_close` is the *raw* prior close, not adjusted** (verified on Reliance
  bonus day) → you must build your own adjustment factors. `[Fact, High — inspected]`
- The daily bar in bhavcopy is **not** the same as the HF daily bar (bhavcopy has
  `turnover`, `trades`, `prev_close`, `isin`; HF has `oi` and UTC midnight timestamps).

### 2.3 HF minute schema and semantics

```
symbol: large_string, timestamp: timestamp[us, tz=UTC],
open, high, low, close: float, volume: int64, oi: int64
```

- Timestamps are **UTC**; `2022-01-03 03:45:00+00:00` = **09:15 IST**. Add +5:30 (or
  convert with `tz='Asia/Kolkata'`). `[Fact, High — inspected]`
- Data is **sorted by symbol then timestamp**, sharded alphabetically
  (shard 0 = `20MICRONS`→`BOMDYEING`, shard 7 = `WELENT`→`ZYDUSWELL`), so a single
  symbol is contiguous within one shard. `[Fact, High — inspected]`
- Session = **09:15–15:29 IST = exactly 375 one-minute bars**; each bar is
  left-labeled (the 09:15 bar covers [09:15, 09:16)). `[Fact, High — inspected]`
- **`oi` is 0 for all equity rows** (Reliance: 376,380/376,380 zero). Open interest
  only exists for derivatives; don't waste a feature slot on equity `oi`. `[Fact, High — inspected]`
- `volume` is per-minute; 15 zero-volume bars in Reliance's 375,380 (negligible).
  No negative volumes, no null OHLC, no `high<low`. `[Fact, High — inspected]`

### 2.4 Index / macro files (daily, Yahoo-sourced)

| File | Rows | Start | Notes |
|---|---:|---|---|
| `nifty50.parquet` | 4,675 | 2007-09-17 | Nifty 50 spot |
| `banknifty.parquet` | 4,690 | 2007-09-17 | Bank Nifty spot |
| `india_vix.parquet` | 4,556 | 2008-03-03 | India VIX |
| `sensex.parquet` | 7,210 | 1997-07-01 | BSE Sensex |
| `sp500.parquet` | 24,809 | 1927-12-30 | US S&P 500 |
| `nasdaq.parquet` | 14,035 | 1971-02-05 | Nasdaq Comp |
| `dxy.parquet` | 14,159 | 1971-01-04 | Dollar index |
| `usdinr.parquet` | 5,926 | 2003-12-01 | USD/INR |
| `brent.parquet` | 4,777 | 2007-07-30 | Brent crude |
| `gold.parquet` | 6,551 | 2000-08-30 | Gold |

All share schema `date, close, high, low, open, volume` (volume often 0 for indices).
`[Fact, High — inspected]`

### 2.5 Options (partial — deprioritise)

`rissin/nse-options-intraday` is **incomplete** in this repo: `kotak_live/NIFTY/NIFTY_2026.parquet`
has only **4 synthetic-looking rows**, and most upstox/1-min files are `.incomplete`
locks. The **`historical_daily/NIFTY/NIFTY_2001.parquet`** files are real NSE F&O
bhavcopy data (`source=nse_bhavcopy_legacy`, granularity `1d`, ~8,600 rows/year).
Treat options as a *future* feature source (ATM IV, skew, IV−RV, OI, PCR) requiring
a full re-download, not as a ready input. `[Fact, High — inspected]`

---

## 3. Data-quality landmines

> **This section is the most important part of the report.** Each item below is a
> way a model silently learns noise. All were verified in the files.

### 3.1 The killer: raw vs adjusted prices

| Symbol / date | Bhavcopy (raw) | HF minute (adjusted) |
|---|---|---|
| RELIANCE 2024-10-25 | close 2655.70 | 1328.15 |
| RELIANCE 2024-10-28 (bonus ex-date) | close 1334.35, prev_close 2655.70 → **−49.76%** | 1335.00 (continuous) |
| RELIANCE 2024-10-29 | close 1340.00 | 1340.00 |

`[Fact, High — inspected]`

**Consequences if ignored:** any model trained on bhavcopy-derived returns learns
that Reliance (and every other bonus/split name) occasionally crashes ~50% and then
recovers; labelers see a "short" signal at the exact date; cross-asset alignment
produces a −50% gap where a real overnight gap was ~0%. The HF dataset is already
back-adjusted (good for continuity), but back-adjustment folds *future* corporate
actions into *past* levels, which is itself a (mild) look-ahead for level-based
features — safe for returns, cautionary for ratios.

**Fix (pick one, documented):**
- **(A) Return-space modelling:** compute all features/labels from *returns* and
  ratios, never from raw price levels. This is the pragmatic default and what most
  of the feature list in §5 uses.
- **(B) Build a point-in-time adjustment factor** from the bhavcopy ratio itself:
  `adj_factor = prev_close / prior_raw_close`, cumulative backward; adjust only rows
  ≤ current date. Then level features are safe.
- **(C) Prefer the HF adjusted series** for 2022–2026 intraday work, and use
  bhavcopy only for universe/history/ex-post delisted names.

Never merge the two price levels without adjusting. `[Inference, High]`

### 3.2 Magnitude of the corporate-action problem

In bhavcopy `EQ` with `prev_close > 0`:

| Threshold | Rows | Share |
|---|---:|---:|
| \|close/prev_close − 1\| > 5% | 842,742 | 9.6% |
| \|close/prev_close − 1\| > 25% | 7,949 | 0.09% |
| ratio < 0.5 or > 2 (clear action) | **2,180** | 0.02% |
| ratio < 0.2 | 504 | — |
| ratio > 1.96 | 438 | — |

Big-price-action events by year: 122 (2000), 336 (2001), 139 (2015), 143 (2016),
85 (2024), 74 (2025), 73 (2026 YTD). Extreme examples: `NIRLON` 2026-04-20
1.40 → 539.79 (×385); `HAWKINCOOK` 37.10 → 7573.60; `LICMFGOLD` 14,401.80 → 143.20.
`[Fact, High — inspected]`

### 3.3 Off-grid and special-session bars

- **NIFTY 1-min off-grid:** 84 bars at 09:07, 46 at 15:30, 84 at 15:31, 5 at 15:39,
  5 at 15:59, and evening bars 18:00–19:14 (a handful of dates) — **404 off-grid of
  375,667 (0.11%)**. `[Fact, High — inspected]`
- **Special sessions:** exactly 60 bars on 2022-10-24, 2023-11-12, 2024-11-01,
  2025-10-21 (Diwali **Muhurat**, evening session); 105 bars on 2024-03-02,
  2024-05-18 (special Saturday sessions). `[Fact, High — inspected]`
- **Fix:** filter to `09:15 ≤ t ≤ 15:29`, reindex each symbol-day to a 375-slot grid,
  and add `is_special_session` / `session_length` features so the model isn't fed a
  mis-aligned row (feature at position 300 means a different clock time on a 60-bar
  day). Drop or flag sessions with < ~300 bars for intraday-bar-indexed models.

### 3.4 Survivorship and universe

- HF minute/day covers **2,535 currently-listed symbols** → survivorship-biased.
  Bhavcopy covers **8,897** including delisted names → point-in-time. `[Fact, High — inspected]`
- Do **not** learn "the Nifty 50" from today's constituent list: use point-in-time
  membership if you build a cross-sectional/index model. `[Inference, High]`
- The HF daily dataset **stops at 2026-01-22** while bhavcopy runs to **2026-10-07**
  — there is a **~8.5-month tail** where you only have raw daily data. Plan the
  train/validation boundary accordingly. `[Fact, High — inspected]`

### 3.5 Session / timezone / bar-timestamp traps

- Bhavcopy `date` is a plain date; HF daily `timestamp` is UTC midnight of the IST
  trading day; HF minute is UTC. **Normalise everything to `Asia/Kolkata` and a
  trading-session calendar** before joining. `[Fact, High — inspected]`
- A bar timestamped 09:30 includes trades up to 09:31; acting on 09:30 close
  information at 09:30 is look-ahead — fill at 09:31 (or next bar open).
  `[Fact, High]` [Quant Memo look-ahead](https://quantmemo.com/concepts/look-ahead-bias)
- Rolling windows must reset at session boundaries (a 20-bar MA must not silently
  span the overnight gap). Use a per-symbol **sequential bar index**, not wall-clock
  time. `[Inference, High]` [JP intraday daily_model](https://github.com/yfuka86/finance/blob/33346dc6/trading/jp_intraday/daily_model.py)
- Full-sample scaling / target encoding / PCA leaks; fit every transformer inside
  the training fold only. `[Fact, High]` [Quant Memo](https://quantmemo.com/concepts/look-ahead-bias)

---

## 4. How much can we improve the input?

Honest, tiered answer. The user's question is "how much can we improve the input we
give the model" — the answer is that engineering the **data contract** and
**feature set** yields large, verifiable gains; engineering more exotic features
yields diminishing returns; and the *label/target* choice dominates the model choice.

| Lever | Expected effect | Evidence / reasoning |
|---|---|---|
| **Fix raw-vs-adjusted mismatch** | Removes 2,180 catastrophic fake returns + 842k spurious >5% bars; likely the difference between a broken and a usable model | Inspected; §3.1–3.2 `[Inference, High]` |
| **Session-grid alignment** | Removes ~1.4% mis-aligned rows and prevents spurious overnight-window features | Inspected; §3.3 `[Inference, High]` |
| **Go from raw OHLCV (5 cols) to Alpha158-class features (158)** | This is the documented industry baseline; Qlib shows LightGBM IC rises from ~0.02 (weak raw) to 0.0448 on Alpha158 | [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) `[Fact, High]` |
| **Add intraday + volatility-estimator + cross-asset + cross-sectional features (→ ~300–600)** | Modest further IC gain, but a large gain in *volatility/risk* R² and regime detection | `[Inference, Med]` |
| **Fractional differentiation of levels** | Preserves memory vs returns; RMSE/MAE improvement vs integer differencing in an ANN study | [Springer 2021](https://link.springer.com/article/10.1007/s12572-021-00299-5) `[Fact, Med]` |
| **Feature selection (MDA/SFI/SHAP) from ~500 → ~50–80** | Reduces overfitting and training time; Grinsztajn shows GBT is robust to up to ~50% uninformative features, but selection still helps stability | [Grinsztajn 2022](https://arxiv.org/abs/2207.08815) `[Fact, High]` |
| **Change target from "exact price" to realized-vol / triple-barrier / meta-label** | The largest single modelling gain; converts an unpredictable target into a partially predictable one | [AFML Ch.3](https://www.wiley.com/en-ie/advances-in-financial-machine-learning-p-9781119482086) `[Inference, High]` |
| **Hyperparameter tuning (Optuna vs defaults)** | Meaningful but bounded; rarely more than a few % of IC. Tuning does not rescue a bad feature set | [Optuna](https://arxiv.org/abs/1907.10902) `[Inference, High]` |
| **Adding deep sequence model on top of GBDT** | Ensemble diversity; competitors repeatedly blend GBDT+NN for small Sharpe gains | [Ubiquant](https://zizouhe.github.io/reading-notes/notes/kaggle-fin.html) `[Fact, Med]` |

**Rule of thumb to state to stakeholders `[Estimate, Med]`:**
- 70% of the achievable improvement comes from **data correctness + target design**
  (§3 + §8), not from the model.
- 20% comes from **feature breadth** (Alpha158-class + intraday/vol/cross-asset).
- 10% comes from **model choice + tuning**.

And the ceiling: on liquid NSE names, a genuine out-of-sample **directional Rank IC
of ~0.01–0.05** is a normal "good" result; an R² of 1–2% on next-bar *returns* is
respectable; realized-volatility R² can be far higher (HAR-RV is a strong baseline).
`[Inference, High]` backed by [HN: "R² of 1% … something to write home about"](https://news.ycombinator.com/item?id=14943605)
and the Qlib IC values above.

---

## 5. Feature engineering blueprint — daily layer

Build a **per-symbol daily table** indexed by `(symbol, trading_date)` with features
known at or before the close of day *t*, and labels from *t+1* forward. Anchor the
feature vocabulary to **Qlib's Alpha158**, which is a battle-tested, reproducible set
(9 kbar + 4 price + 145 rolling = 158). Exact expressions from
[`qlib/contrib/data/loader.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py). `[Fact, High]`

### 5.1 kbar features (9) — candle geometry (scale-free)

| Name | Formula |
|---|---|
| KMID | `(close − open)/open` |
| KLEN | `(high − low)/open` |
| KMID2 | `(close − open)/(high − low + ε)` |
| KUP | `(high − max(open,close))/open` |
| KUP2 | `(high − max(open,close))/(high − low + ε)` |
| KLOW | `(min(open,close) − low)/open` |
| KLOW2 | `(min(open,close) − low)/(high − low + ε)` |
| KSFT | `(2·close − high − low)/open` |
| KSFT2 | `(2·close − high − low)/(high − low + ε)` |

`[Fact, High]` (same source). These are the highest signal-per-cost features because
they are already stationary and dimensionless.

### 5.2 Price-normalised features (4)
`OPEN0=open/close`, `HIGH0=high/close`, `LOW0=low/close`, `VWAP0=vwap/close`.
`[Fact, High]`

### 5.3 Rolling operator features (29 operators × windows {5,10,20,30,60} = 145)

The full operator list (from the same loader): `ROC, MA, STD, BETA, RSQR, RESI, MAX,
MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP,
SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD`. Notable ones:
- **RSV** = stochastic oscillator `(close − min(low,d))/(max(high,d) − min(low,d))`
- **IMAX/IMIN/IMXD** = Aroon-like "bar index of rolling high/low"
- **SUMP/SUMN/SUMD** = RSI-like up/down volume/return ratios
- **CORR/CORD** = price–volume correlation
- **VMA/VSTD/WVMA** = volume MA / volume std / volume-weighted volatility
`[Fact, High]`

### 5.4 Volatility-estimator family (add these — Qlib omits most)

Daily range estimators are more efficient than close-to-close squared returns and
should be added explicitly `[Fact, High]` ([Yang-Zhang/repub PDF](https://repub.eur.nl/pub/7582/ei2006-10.pdf), [arXiv 2312.01426](https://arxiv.org/pdf/2312.01426v2)):

| Estimator | Formula | Efficiency vs close-close |
|---|---|---|
| Parkinson | `σ²_P = (1/(4 ln 2))·[ln(H/L)]²` | ~5× |
| Garman–Klass | `σ²_GK = 0.5[ln(H/L)]² − (2ln2−1)[ln(C/O)]²` | ~7.4× |
| Rogers–Satchell | `σ²_RS = ln(H/O)[ln(H/O)−ln(C/O)] + ln(L/O)[ln(L/O)−ln(C/O)]` | ~6×, drift-independent |
| Yang–Zhang | `V_YZ = V_o + k·V_c + (1−k)·V_RS` | min error, gap-independent |
| Realized vol (intraday) | `RV = Σ r_i²` (1-min returns) | uses the 713M-row data |
| Bipower variation | `BV = (π/2)·Σ|r_i||r_{i−1}|` | jump-robust |
| Realized semivariance | `RS± = Σ r_i²·1{r_i≷0}`, `SignedJump = RS+ − RS−` | asymmetry signal |
| **HAR-RV** | regress `RV_t` on daily/weekly(5)/monthly(22) RV lags | the strong baseline |

`[Fact, High]` [Patton & Sheppard 2015](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf) · [HAR-RV](https://metricgate.com/docs/har-rv-realized-volatility-model)

### 5.5 Technical-indicator family (the ones with documented evidence)
`RSI(14/21,50)`, `MACD(12,26,9)`, `Bollinger(20,2)`, `ATR(14)`, `ADX`, `CCI`,
`Williams %R`, `OBV`, `Stochastic`. Evidence is mixed but non-zero: MACD/RSI showed
gross profitability on the LSE ([Chong & Ng 2008](https://www.tandfonline.com/doi/full/10.1080/13504850600993598));
in a 5-market OECD test MACD had no predictability in 4/5 but RSI(21,50) worked in
Milan ([MDPI JRFM 2014](https://www.mdpi.com/1911-8074/7/1/1)); >28,000 technical
rules were profitable net of costs in China ([Jiang et al. 2019](https://onlinelibrary.wiley.com/doi/10.1111/irfi.12161)).
Treat these as **candidate features to be selected, not guaranteed alpha**. `[Fact, Med]`

### 5.6 Volume / liquidity family
log dollar-volume, turnover (`volume/prev_volume`), volume z-score vs 20d,
**Amihud illiquidity** `ILLIQ = mean(|r|/dollar_volume)`, OBV slope, VWAP deviation
`(close − VWAP)/VWAP`, anchored-VWAP distance, delivery-% (not yet in bhavcopy —
would need `sec_bhavdata_full`). `[Fact, High]` [Amihud 2002](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)

### 5.7 Fractional differentiation (optional but recommended)
For log-price, log-volume and log-dollar-volume, compute FFD with the **minimum d
that passes ADF** (frequently 0.1–0.4) to keep long memory that plain returns
destroy. Use [`fracdiff`](https://github.com/fracdiff/fracdiff) `FracdiffStat`.
`[Fact, Med]` [López de Prado AFML Ch.5](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c05.xhtml) · [Springer 2021](https://link.springer.com/article/10.1007/s12572-021-00299-5)

---

## 6. Feature engineering blueprint — intraday (1-min) layer

The intraday branch is where the 713M-row dataset earns its keep. Everything must be
computed on a **per-symbol, session-reset bar index** and **lagged** so bar *t* sees
only bars ≤ *t−1*. `[Inference, High]`

### 6.1 Resampling to decision horizons
Don't model raw 1-minute direction — the noise/cost ratio is worst there. Build
features at each bar but aggregate labels/returns at **5-min, 15-min, 60-min** (and
daily). Resample OHLC correctly: `open=first`, `high=max`, `low=min`, `close=last`,
`volume=sum`. `[Inference, High]`

### 6.2 Intraday microstructure / session features
Derive from the 375-bar session:

| Feature | Definition | Source |
|---|---|---|
| `time_since_open` | bar index / 375 | `[Inference, High]` |
| `overnight_gap` | `(open_09:15 − prev_close)/prev_close` | `[Fact, High]` [Barardehi et al.](https://experts.illinois.edu/en/publications/what-drives-momentum-and-reversal-evidence-from-day-and-night-sig) |
| `first_15/30m_return` | `close_{09:30}/open_{09:15}−1` | `[Fact, High]` [Gao et al. JFE 2018](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351) |
| `last_30m_return` | `close_{15:30}/close_{15:00}−1` | `[Fact, High]` (same) |
| `intraday_momentum` | `sign(first_30m)·last_30m` | `[Fact, High]` (same) |
| `ORB_high/low_dist` | distance to opening-range (first 15/30m) high/low | `[Fact, Med]` |
| `intraday_RV_k` | `Σ r_i²` over the last k bars (5/15/30/60) | `[Fact, High]` |
| `realized_skew`, `realized_kurt` | 3rd/4th moments of intraday returns | `[Fact, High]` |
| `up/down_vol_ratio` | `RS+/RS−` | `[Fact, High]` [Patton & Sheppard](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf) |
| `vwap_dev_session` | `(close − session_VWAP)/session_VWAP` **using only bars ≤ t** | `[Inference, High]` |
| `volume_profile_share` | bar volume / cumulative session volume | `[Inference, High]` |
| `time_of_day` dummies | e.g. 09:15–09:30, 09:30–10:00, …, 15:00–15:30 | `[Fact, High]` [Sampath & Gopalaswamy 2020](https://journals.sagepub.com/doi/full/10.1177/0972652720930586) |
| `is_expiry_week`, `days_to_expiry` | regime-segmented (see §7) | `[Fact, High]` [NSE circular](https://avantiscdnprodstorage.blob.core.windows.net/legalupdatedocs/43858/NSE-notified-regarding-the-Revision-in-Expiry-Day-of-Index-and-Stock-Derivatives-Contracts-June242025.pdf) |

**NSE-specific intraday structure to encode:** volume/volatility are **U-shaped**
(high at open and close); NSE's intraday vol peak-to-low ratio is **4–5×** (vs ~2×
for Nikkei/S&P); on F&O maturity the last 30 min (15:00–15:15, 15:15–15:30) sees
volatility jumps of ~80% and ~22% because settlement uses the 15:00–15:30 VWAP.
`[Fact, High]` [IIMA repository](http://vslir.iima.ac.in:8080/xmlui/handle/11718/6511) · [Sampath & Gopalaswamy](https://journals.sagepub.com/doi/full/10.1177/0972652720930586)

### 6.3 Rolling stats, computed only on past bars
For k ∈ {5, 15, 30, 60, 120} bars: return lags, sum, std, skew, z-score
`(r_t − μ_k)/σ_k`, `close/max(high,k)`, `close/min(low,k)`, position-in-range, return
autocorrelation, signed-volume sums. All `shift(1)`. `[Inference, High]`

### 6.4 Don't feed the model raw 1-min prices
Raw price levels in intraday features are non-stationary and non-comparable across
symbols. Use ratios/differences, or the Alpha360-style "each price normalised by the
latest close" scheme. Qlib's sequence models win on Alpha360 (raw normalised
sequences) — that is the correct way to hand sequences to a deep net. `[Fact, High]`
[Qlib Alpha360](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)

---

## 7. Cross-asset, cross-sectional & calendar features

### 7.1 Cross-asset / macro (already downloaded)
- India VIX level, log-change, and **IV−RV** (implied minus realized) — the latter is
  model-friendly and free here. `[Fact, High]`
- Nifty 50 / Bank Nifty returns and realized vol; Sensex for confirmation.
- Overnight global cues: S&P 500, Nasdaq returns (lagged to the prior US close),
  DXY, Brent, Gold, USD/INR. These are known *before* the NSE open. `[Inference, High]`
- **Alignment rule:** use the most recent *completed* foreign session as of the NSE
  open; never forward-fill a foreign close that hasn't happened. `[Inference, High]`

### 7.2 Cross-sectional (rank / relative) features
For each date/bar, across the universe: `ret_rank_cs`, `vol_rank_cs`, z-scores,
market beta (60/120d), idiosyncratic vol, stock-minus-index return, sector-relative
return (needs a sector map), and **cross-sectional dispersion** (std of returns),
which has documented market-return forecasting power.
`[Fact, High]` [Maio 2016](https://ideas.repec.org/a/eee/finmar/v29y2016icp87-109.html) · Qlib `CSZScoreNorm`/`CSRankNorm` in [processor.py](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py)

### 7.3 Calendar (NSE-specific, regime-aware)
- `day_of_week` (beware: effects are time-varying and weak in India
  `[Fact, Med]` [NSE research](https://archives.nseindia.com/content/research/res_paper_final228.pdf)),
  `month`, `day_of_month`, `turn_of_month`, pre/post-holiday dummies.
- **Expiry regime split is essential:**
  - Before 20 Nov 2024: Bank Nifty/FinNifty/MidcpNifty/NiftyNext50 weeklies existed.
  - From 20 Nov 2024: only Nifty weekly survives (one weekly per exchange).
  - From 28 Aug 2025 / 1 Sep 2025: NSE weekly expiry moved **Thursday → Tuesday**.
  So `days_to_expiry` must be computed from **actual contract expiry dates**, and
  `is_expiry_day` must be **regime-segmented**, not a fixed weekday.
  `[Fact, High]` [NSE circular](https://avantiscdnprodstorage.blob.core.windows.net/legalupdatedocs/43858/NSE-notified-regarding-the-Revision-in-Expiry-Day-of-Index-and-Stock-Derivatives-Contracts-June242025.pdf) · [CNBC-TV18](https://www.cnbctv18.com/market/nse-to-shift-expiry-of-derivatives-to-tuesdays-starting-august-28-19625489.htm)
- `is_muhurat`, `is_special_session`, `session_length` (from §3.3).
- **NSE holidays:** pull the official holiday list per year for accurate trading-day
  counts (the data already encodes holidays as missing days, but explicit dummies help).
  `[Fact, Med]` [NSE holidays](https://www.nseindia.com/resources/exchange-communication-holidays)

---

## 8. Targets & labels

> The target choice dominates the model. Pick it before writing a single model line.

| Target | Definition | When to use | Verdict |
|---|---|---|---|
| **Realized volatility** | `RV_{t→t+h}` from intraday returns; predict next-day/hour RV | First model — genuinely predictable, strong HAR baseline | **Start here** `[Inference, High]` |
| **Log-return regression** | `r_{t+h} = log(P_{t+h}/P_t)` | Continuous signal, IC evaluation | Useful; MSE dominated by vol outliers |
| **Direction classification** | `1[r_{t+h} > cost]` | Bet side | Near 50/50 at 5-min; use only with meta-labeling |
| **Triple-barrier** | label = first of {profit-take, stop-loss, time} hit | Realistic exit logic + gives each label a span for purging | High ROI `[Fact, High]` [AFML Ch.3](https://www.wiley.com/en-ie/advances-in-financial-machine-learning-p-9781119482086) |
| **Meta-labeling** | primary sets side; secondary predicts success → sizing | Decouples direction from size; improves precision | High ROI `[Fact, High]` [Hudson & Thames](https://hudsonthames.org/meta-labeling-a-toy-example/) |
| **Quantile forecast** | predict 5/50/95th pct | Risk/sizing | Good complement |

**Triple-barrier details:** scale barriers by an **entry-time** volatility estimate
(e.g. EWM std using only data ≤ t0), typical `pt_sl=[1,1]` or `[2,1]`, vertical
barrier = h bars. On intraday data, detect barrier touches with **High/Low**, not
just closes. `[Fact, High]` [mlfinpy Labelling](https://mlfinpy.readthedocs.io/en/latest/Labelling.html)

**Sample weights:** overlapping labels are not IID — compute concurrency
`c_t`, average uniqueness `u_i = mean(1/c_t)`, and weight `w_i ∝ |r_i|/u_i` (optionally
time-decayed). Empirical mean uniqueness ≈ 0.41–0.44, i.e. effective sample size is
much smaller than row count. `[Fact, Med]` [Daru Finance](https://daru.finance/research-review/lopez-de-prado/sample-uniqueness-and-bootstrapping)

---

## 9. Which model is "perfect" — model selection

### 9.1 The verdict

**Primary model: LightGBM** on the engineered tabular matrix. **Secondary: CatBoost
and XGBoost** as near-equal variants. **Optional sequence branch: TCN/GRU or
iTransformer** on normalised raw sequences. **Ensemble the two** (rank-average or
stacking via purged CV). Skip zero-shot foundation models.

### 9.2 Why (evidence)

- **Tabular benchmarks:** tree ensembles remain state-of-the-art on medium tabular
  data; the gap "does not disappear after tuning hyperparameters," and deep nets
  struggle with uninformative features and irregular target functions — exactly the
  financial-tabular situation. `[Fact, High]` [Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)
- **Finance-specific:** Gu, Kelly & Xiu find "shallow learning outperforms deeper
  learning" for return prediction due to low signal-to-noise.
  `[Fact, High]` [NBER w25398](https://www.nber.org/system/files/working_papers/w25398/revisions/w25398.rev0.pdf)
- **Qlib's own benchmark on engineered features (Alpha158, CSI300):**
  LightGBM IC **0.0448** / ICIR 0.366 / RankIC 0.0469; CatBoost IC 0.0481;
  XGBoost IC 0.0498; whereas TFT IC 0.0358 (RankIC 0.0116), GRU 0.0315, LSTM 0.0318,
  Transformer 0.0264, TabNet 0.0204. **LightGBM-class models win on engineered
  tabular features.** `[Fact, High]` [Qlib benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- **On raw sequences (Alpha360), sequence models overtake trees:** GRU IC 0.0493,
  ALSTM 0.0497, HIST 0.0522 vs LightGBM 0.0400. This is why the sequence branch is
  worth having **only** on the raw-sequence representation. `[Fact, High]` (same)
- **Competitions on noisy financial tabular data:** Jane Street 2020 1st = supervised
  autoencoder + MLP blended with XGBoost; Ubiquant 2022 1st = LGBM + TabNet, 2nd =
  pure LightGBM with purged K-fold, 3rd = Transformer; Optiver 2021 1st = blend of
  LightGBM, MLP and 1D-CNN. **GBDT is the stable base; NNs add diversity.**
  `[Fact, High]` [Jane Street 1st](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) · [Ubiquant](https://zizouhe.github.io/reading-notes/notes/kaggle-fin.html)
- **Foundation TS models underperform:** on financial returns TSFMs beat a random
  walk only "small and sparse"; for realized volatility only Tiny Time Mixers barely
  beats Log-HAR. `[Fact, Med]` [TSFM returns](https://arxiv.org/html/2606.27100v1) · [RV with TSFMs](https://arxiv.org/pdf/2607.05291)

### 9.3 Model comparison table

| Model | Role | Pros | Cons | Verdict for NSE |
|---|---|---|---|---|
| **LightGBM** | **Primary** | Fast on 700M rows, handles missing/categoricals, robust to useless features, SHAP support | Needs careful CV | **Best default** |
| CatBoost | Strong alt | Great categorical handling, ordered boosting reduces overfit, often best on noisy finance | Slower | Use in ensemble |
| XGBoost | Strong alt | Mature, `scale_pos_weight`, well-tuned ecosystem | Slightly slower than LGBM | Use in ensemble |
| Ridge / logistic | Baseline | Fast, interpretable, hard to beat at 5-min | Linear only | Always report |
| HAR / EWMA / GARCH | Baseline | The volatility benchmark you must beat | No cross-features | **Mandatory baseline** |
| LSTM/GRU/TCN | Sequence branch | Win on raw sequences (Alpha360) | Needs large data + GPU, overfits | Optional |
| TFT / PatchTST / N-BEATS | Multi-horizon DL | Quantile/multi-horizon output | Weak Rank IC in Qlib benchmark | Only if ensemble |
| iTransformer | Sequence branch | Strong recent results, direct correlation loss | Needs careful training | Optional |
| Chronos / TimesFM / Moirai | N/A | Zero-shot | Do not beat random walk / HAR on finance | **Skip (or feature only)** |
| Random Forest / ExtraTrees | Diversity | Cheap | Usually behind boosting | Occasional ensemble |

### 9.4 Class-imbalance handling (direction models)
- LightGBM: `scale_pos_weight ≈ neg/pos` or `is_unbalance=true` (not both; both hurt
  probability calibration). `[Fact, High]` [LightGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- Prefer **keeping calibrated probabilities** and tuning the **decision threshold /
  top-k cut** on a validation metric, rather than only reweighting the loss.
  `[Inference, Med]`
- Focal loss exists (`FL = −(1−p_t)^γ log p_t`) but is designed for extreme
  imbalance (1:1000); NSE direction is rarely that skewed. `[Fact, High]` [Lin et al. 2017](https://arxiv.org/abs/1708.02002)

---

## 10. Tuning: GridSearchCV vs RandomizedSearch vs Optuna

### 10.1 Verdict
**Do not use plain `GridSearchCV`.** Use **random search as the honest baseline**, then
**Optuna (TPE + MedianPruner/ASHA)** with a **purged-CV objective**. Grid search is
only acceptable for 1–2 parameters as a sanity check.

### 10.2 Why
- **Bergstra & Bengio (JMLR 2012):** random search is more efficient than grid search
  because only a few hyperparameters matter and which ones differ per dataset; grid
  "is a poor choice." `[Fact, High]` [paper](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- **Optuna (KDD 2019):** TPE + CMA-ES is statistically worse than random in only
  1/56 cases; with pruning it explored ~1,278 trials vs ~36 without, in the same wall
  clock. `[Fact, High]` [arXiv 1907.10902](https://arxiv.org/abs/1907.10902)
- **Hyperband/ASHA** exploit early stopping and outperform random search; SHA/ASHA/BOHB
  are comparable to each other. `[Fact, High]` [ASHA](https://arxiv.org/abs/1810.05934)

### 10.3 Recommended LightGBM search space `[Estimate, Med]`

| Parameter | Range / distribution |
|---|---|
| `learning_rate` | log-uniform 0.01–0.2 (then lower it later) |
| `num_leaves` | int 15–511 |
| `max_depth` | int 3–12 (or −1) |
| `min_child_samples` | int 10–200 (large values regularise noisy finance) |
| `feature_fraction` (`colsample_bytree`) | uniform 0.4–1.0 |
| `bagging_fraction` + `bagging_freq` | uniform 0.4–1.0; freq 1–7 |
| `lambda_l1`, `lambda_l2` | log-uniform 1e-8–1e4 (Qlib's tuned example used ~206 / ~581) |
| `min_split_gain` | 0–1 |
| `scale_pos_weight` | fixed at neg/pos for classification; threshold tuned separately |
| `objective` | `regression` (L1/L2/Huber) for returns; `binary` for direction; `poisson`/L2 for RV |

Qlib's own tuned LightGBM example (`colsample 0.8879, lr 0.2, subsample 0.8789,
λ1 205.7, λ2 580.98, max_depth 8, leaves 210`) is a good warm start.
`[Fact, High]` [Qlib config](https://github.com/microsoft/qlib/blob/7ccf3f76/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_multi_freq.yaml)

### 10.4 Tuning protocol
1. Fixed **purged/embargoed walk-forward** folds (or CPCV) — never shuffle.
2. Objective = **Rank IC / AUC on OOS folds** (regression: negative RMSE on the
   *vol-normalised* target).
3. `RandomizedSearchCV(n_iter≈100)` for a cheap landscape read.
4. `optuna.create_study(direction=..., sampler=TPESampler, pruner=MedianPruner())`,
   `n_trials≈500–2000` with early pruning; log to SQLite.
5. Re-tune `learning_rate ↓` with `num_leaves` held after a good region is found.
6. **Nested CV / hold-out** for the final number: never tune on the test fold.

### 10.5 Feature selection (after tuning)
- Compute **MDI** (`feature_importances_`), **MDA** (`permutation_importance`) and
  **SFI** (single-feature purged CV) + **SHAP**; prefer MDA/SFI because MDI is biased
  toward correlated/high-cardinality features. `[Fact, High]` [sklearn permutation_importance](https://scikit-learn.org/stable/modules/permutation_importance.html) · [SHAP](https://github.com/shap/shap)
- Cluster correlated features (or PCA-orthogonalise) before interpreting importance.
- GBT is robust to up to ~50% uninformative features (Grinsztajn), but selection
  still improves stability and latency. `[Fact, High]` [Grinsztajn](https://arxiv.org/abs/2207.08815)

---

## 11. Validation protocol

Non-negotiable, in order:
1. **Purged K-Fold + embargo:** remove training rows whose label window overlaps the
   test fold; drop a post-test buffer for serial correlation. A controlled experiment
   showed shuffled CV reports R²≈0.92 with 100% overlap vs no skill under PurgedKFold.
   `[Fact, High]` [purgedcv](https://github.com/eslazarev/purged-cross-validation)
2. **Combinatorial Purged CV (CPCV):** N blocks, k held out → `C(N,k)` folds and
   `C(N−1,k−1)` backtest paths (e.g. N=6,k=2 → 15 folds, 5 paths). Gives a
   *distribution* of OOS performance, not a single path. `[Fact, High]` [Wikipedia CPCV](https://en.wikipedia.org/wiki/Purged_cross-validation) · [purgedcv](https://eslazarev.github.io/purged-cross-validation/)
3. **Deflated Sharpe Ratio (DSR) + PBO:** adjust the Sharpe for the number of trials;
   report probability of backtest overfitting. `[Fact, High]` [Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [PBO](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
4. **Naive baselines always:** always-up, random-walk, buy-and-hold, EWMA/HAR vol.
5. **Fit all scalers/encoders inside the fold.**
6. **Net-of-cost evaluation:** NSE round-trip ≈ 10.6 bps equity / 5.9 bps index
   futures (from the parent dossier). A 5-min Nifty σ ≈ 11–12 bps — friction ≈ one
   full standard deviation. `[Estimate, Med]` (see parent dossier §7.5)
7. **Paper-trade** before capital; check live-vs-backtest fill match.

**Metrics that matter:** AUC / PR-AUC, **Rank IC & ICIR** (the Qlib/industry
standard), log-loss (calibration), precision@top-k, top-minus-bottom decile spread,
and net-of-cost Sharpe. **Not accuracy** — an always-up rule hit ~80% in a rising
market. `[Fact, Med]` [When Directional Accuracy Lies](https://arxiv.org/abs/2607.12248v2) · [Qlib metrics](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)

---

## 12. Concrete build plan & next steps

**Phase A — data contract (highest ROI, do first).** `[Inference, High]`
1. Build `eq_universe` from bhavcopy (`series ∈ {EQ, BE}`) with point-in-time listing.
2. Build a corporate-action adjustment map from the bhavcopy ratio (`adj_factor`),
   OR commit to return-space modelling. Validate on the 5 known events in §3.1.
3. Build a canonical **375-slot session grid** for the minute data; drop/flag
   off-grid and special-session bars.
4. Align indices/macro to the IST trading calendar; compute foreign-session lags.
5. Freeze a **golden feature/label join sample** (e.g. 50 symbols × 2 months) and
   eyeball it against raw files.

**Phase B — baseline feature matrix (1–2 weeks).**
6. Implement Alpha158 + the volatility-estimator family + Amihud/OBV/VWAP for daily.
7. Add intraday session features (gap, first/last 30m, RV_k, time-of-day) from minute.
8. Add cross-asset (VIX, IV−RV, S&P/Nasdaq/DXY/USDINR/Brent/Gold lags) and
   cross-sectional rank features.
9. Add fractional-differenced levels for key series.

**Phase C — targets & baselines.**
10. Labels: next-day RV, next-day return, triple-barrier; then meta-label.
11. Baselines: HAR/EWMA vol, logistic/ridge, naive always-up.

**Phase D — models & tuning.**
12. LightGBM with Optuna TPE + MedianPruner on purged CV; CatBoost/XGBoost in ensemble.
13. Optional TCN/GRU on Alpha360-style sequences; rank-average the ensemble.
14. Feature selection via MDA/SFI/SHAP.
15. CPCV + DSR/PBO; net-of-cost backtest.

**Immediate environment gap:** add `scikit-learn`, `lightgbm`, `xgboost`, `catboost`,
`optuna`, `shap`, `statsmodels`, `arch`, `fracdiff`, `polars`, `sktime` (and `torch`
only if doing the sequence branch) to `pyproject.toml`; none are installed.
`[Fact, High — inspected]`

---

## 13. Risks & unknowns

| # | Risk / unknown | Type | Severity | Likelihood | Note |
|---|---|---|---|---|---|
| 1 | Raw-vs-adjusted merge poisoning | Data | High | High unless fixed | §3.1 |
| 2 | Directional signal below NSE costs | Market | High | High | 6–12 bps friction vs ~11 bps 5-min σ |
| 3 | Overfitting / backtest overstatement | Method | High | Med-High | DSR/PBO required |
| 4 | Survivorship bias in HF data | Data | Med-High | High | Use bhavcopy universe |
| 5 | Off-grid/special-session misalignment | Data | Med | High | §3.3 |
| 6 | Expiry-regime changes (2024/2025) | Feature | Med | Certain | Must segment |
| 7 | Non-stationarity / regime decay | Model | High | High | Walk-forward, refit |
| 8 | Foundation-model hype | Model | Low | Med | Skip as primary |
| 9 | Options features unavailable (partial download) | Data | Med | High | Re-download if wanted |
| 10 | 8.5-month HF/bhavcopy tail mismatch | Data | Low-Med | Certain | Plan boundary |

Ranked by severity × likelihood: risk 1 and 2 dominate. `[Inference, High]`

---

## 14. Sources

**Primary — local data (inspected 2026-10-08):**
- `data/processed/bhavcopy/bhavcopy.parquet` (schema, 10,672,269 rows, ratio distribution, Reliance bonus)
- `data/raw/hf/xxparthparekhxx__indian-stock-market-minute-data/minute/train-0000{0,4,5}.parquet` (schema, grid, off-grid bars)
- `data/raw/hf/xxparthparekhxx__indian-stock-market-minute-data/day/train-00000.parquet`
- `data/raw/indices/*.parquet`, `data/reference/security_master.parquet`
- `data/raw/hf/rissin__nse-options-intraday/` (partial)

**Model selection / benchmarks:**
- Grinsztajn, Oyallon & Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022 — https://arxiv.org/abs/2207.08815
- Microsoft Qlib model-zoo benchmarks (Alpha158/Alpha360, IC/ICIR) — https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md
- Qlib feature loader (Alpha158/Alpha360 exact expressions) — https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
- Gu, Kelly & Xiu, *Empirical Asset Pricing via Machine Learning* — https://www.nber.org/system/files/working_papers/w25398/revisions/w25398.rev0.pdf
- Kaggle Jane Street 2020 1st place (AE+MLP & XGBoost) — https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s
- Kaggle Ubiquant 2022 winning solutions summary — https://zizouhe.github.io/reading-notes/notes/kaggle-fin.html
- PatchTST — https://arxiv.org/abs/2211.14730
- TSFMs for returns — https://arxiv.org/html/2606.27100v1 ; RV with TSFMs — https://arxiv.org/pdf/2607.05291
- When Directional Accuracy Lies — https://arxiv.org/abs/2607.12248v2

**Feature engineering:**
- Qlib data processors (`CSZScoreNorm`, `CSRankNorm`) — https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py
- Yang–Zhang / range estimators — https://repub.eur.nl/pub/7582/ei2006-10.pdf ; https://arxiv.org/pdf/2312.01426v2
- Patton & Sheppard (realized semivariance) — https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf
- HAR-RV — https://metricgate.com/docs/har-rv-realized-volatility-model
- Amihud illiquidity — https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf
- Fractional differentiation — https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c05.xhtml ; https://link.springer.com/article/10.1007/s12572-021-00299-5 ; https://github.com/fracdiff/fracdiff
- Technical-analysis evidence — https://www.tandfonline.com/doi/full/10.1080/13504850600993598 ; https://www.mdpi.com/1911-8074/7/1/1 ; https://onlinelibrary.wiley.com/doi/10.1111/irfi.12161
- NSE intraday structure — https://journals.sagepub.com/doi/full/10.1177/0972652720930586 ; http://vslir.iima.ac.in:8080/xmlui/handle/11718/6511
- Gao et al. intraday momentum — https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351
- Feature importance — https://scikit-learn.org/stable/modules/permutation_importance.html ; https://github.com/shap/shap

**Tuning & validation:**
- Bergstra & Bengio (2012) — https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
- Optuna (Akiba et al. 2019) — https://arxiv.org/abs/1907.10902
- Hyperband / ASHA — https://arxiv.org/abs/1810.05934
- Purged CV / CPCV — https://en.wikipedia.org/wiki/Purged_cross-validation ; https://github.com/eslazarev/purged-cross-validation ; https://eslazarev.github.io/purged-cross-validation/
- Bailey & López de Prado, Deflated Sharpe — https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf ; PBO — https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- López de Prado, *Advances in Financial Machine Learning* (Ch. 3–8) — https://www.wiley.com/en-ie/advances-in-financial-machine-learning-p-9781119482086
- Meta-labeling — https://hudsonthames.org/meta-labeling-a-toy-example/ ; https://en.wikipedia.org/wiki/Meta-Labeling
- Triple-barrier — https://mlfinpy.readthedocs.io/en/latest/Labelling.html
- Sample uniqueness — https://daru.finance/research-review/lopez-de-prado/sample-uniqueness-and-bootstrapping
- Look-ahead leakage — https://quantmemo.com/concepts/look-ahead-bias ; https://thedatascientist.com/the-data-leakage-traps-hiding-in-financial-market-data-and-how-to-build-a-leak-free-dataset/

**NSE-specific:**
- NSE expiry-to-Tuesday circular — https://avantiscdnprodstorage.blob.core.windows.net/legalupdatedocs/43858/NSE-notified-regarding-the-Revision-in-Expiry-Day-of-Index-and-Stock-Derivatives-Contracts-June242025.pdf
- CNBC-TV18 expiry shift — https://www.cnbctv18.com/market/nse-to-shift-expiry-of-derivatives-to-tuesdays-starting-august-28-19625489.htm
- NSE timings & holidays — https://www.nseindia.com/static/market-data/market-timings ; https://www.nseindia.com/resources/exchange-communication-holidays

*All external URLs accessed 2026-10-08. Local file facts obtained by direct inspection the same day. This is research, not investment advice.*
