---
title: NSE Next-Day Prediction — Trends, Feature Engineering & Model Selection
date: 2026-10-15
idea: Predict "tomorrow" (the next trading session) for NSE instruments from the repo's daily bhavcopy (2000→2026, 10.67M rows). Research how daily price "trends" actually behave, which statistics add useful features, and which ML model gives the best next-day accuracy — with an honest read on what accuracy is achievable.
verdict: CONDITIONAL-GO — build the feature + validation pipeline; target next-day volatility (real, forecastable) and a small directional/meta-label edge. Expect ~51–55% out-of-sample directional accuracy, NOT 70%+. Best single model = tuned gradient-boosted trees (LightGBM → CatBoost/XGBoost), with a compact NN only as an ensemble complement.
confidence: High on data facts (directly inspected) and on model family; High that published 70–98% NSE accuracies are leakage; Med on any net-of-cost tradeable edge.
---

# NSE Next-Day Prediction — Trends, Feature Engineering & Model Selection

> **How to read this.** This is a focused follow-up to the two existing dossiers,
> [`nse-intraday-ml-forecasting.md`](./nse-intraday-ml-forecasting.md) and
> [`nse-feature-engineering-model-selection.md`](./nse-feature-engineering-model-selection.md).
> Everything tagged `[Fact, High — computed]` was measured **directly on the Parquet files
> in `/home/samarth/thense` on 2026-10-15** with read-only PyArrow/Pandas queries — no model
> was trained. External claims carry a source URL and are tagged `[Fact|Opinion|Inference|Estimate, High|Med|Low]`.
> Direct quotes are ≤25 words and attributed. `[unverified]` = could not confirm against a primary source.
> This is research, not investment, legal, or tax advice.

## Table of contents
1. [TL;DR — the 15 things that matter](#1-tldr)
2. [The idea, unpacked (incl. the "week-wise" question)](#2-the-idea-unpacked)
3. [How "trends" actually behave in your data — measured](#3-how-trends-actually-behave)
4. [Market dynamics: the theory behind the measurements](#4-market-dynamics-theory)
5. [Feature engineering — the statistics toolbox](#5-feature-engineering)
6. [Which ML model gives the best next-day accuracy](#6-model-selection)
7. [The accuracy trap — why most "90% NSE models" are wrong](#7-the-accuracy-trap)
8. [Validation & testing protocol (you have the data; don't fool yourself)](#8-validation-protocol)
9. [Costs — what a next-day signal must beat](#9-costs)
10. [Technical feasibility & build plan](#10-technical-feasibility)
11. [Business & pricing (brief)](#11-business--pricing)
12. [Differentiation, scope & the sharpest wedge](#12-differentiation--scope)
13. [Risks & unknowns](#13-risks--unknowns)
14. [Opportunities & novel additions](#14-opportunities--novel-additions)
15. [Recommended next steps](#15-recommended-next-steps)
16. [Open questions for planning](#16-open-questions-for-planning)
17. [Sources](#17-sources)

---

## 1. TL;DR

1. **Your data is daily, not weekly.** The bhavcopy has **10,672,269 rows / 8,897 symbols / 6,625 trading days, 2000-01-03 → 2026-10-07**; the `EQ` series alone is **8,777,698 rows / 5,036 symbols**. ISO-week counts: **1,075 weeks with 5 sessions, 284 with 4, 38 with 3** (mean **4.742** sessions/week). It *looks* "week-uniform" because ~91% of weeks have 5 sessions. `[Fact, High — computed]`
2. **Daily NSE is very close to a random walk in the *level*, but strongly predictable in *volatility*.** Nifty 50 (2007–2026): lag-1 return autocorrelation **+0.037**, Hurst **0.513**, variance ratios **0.999–1.061** — i.e. essentially no usable trend in raw returns. But lag-1 autocorrelation of **|returns| is +0.279**, and it stays ~0.28 out to lag 10 — textbook **volatility clustering**. `[Fact, High — computed]`
3. **There is a tiny but real positive short-term momentum.** Across 150 liquid EQ stocks (523,066 daily returns), mean per-stock lag-1 autocorrelation = **+0.0325**, and **76.4% of stocks are positive**. Weekly returns show ~0 autocorrelation (**+0.003**). So the exploitable structure is *daily* (small) and *cross-sectional*, not weekly. `[Fact, High — computed]`
4. **Return distributions are fat-tailed and left-skewed.** Nifty daily excess kurtosis **= 15.2** (normal = 0), skew **−0.28**. Any model and any risk metric must assume fat tails; normal-based confidence intervals will lie. `[Fact, High — computed]`
5. **The base rate is your real benchmark, and it drifts.** Nifty up-day share = **52.89%** overall but ranges **43.0% (2011) to 58.8% (2020)** by year. An "always up" rule already scores 52.89%. **Any accuracy near 53% is zero edge.** `[Fact, High — computed]`
6. **Volatility is the tractable target; direction is near the noise floor.** India VIX *level* correlates **0.463** with next-day |Nifty return|; today's |return| correlates **0.268** with tomorrow's. But VIX *returns* correlate **0.0** with next-day Nifty *return*. Predict volatility first, direction second. `[Fact, High — computed]`
7. **The best free cross-asset feature is overnight US equity.** Previous-day S&P 500 return correlates **+0.228** and Nasdaq **+0.239** with the *next* Nifty session return (same-day 0.303). This is a genuine, tradeable-by-timing feature, not noise. USDINR prev-day was ~0 (−0.019). `[Fact, High — computed]`
8. **Best single model = tuned gradient-boosted trees.** On Microsoft Qlib's CSI300 benchmark (20 seeds), **LightGBM IC 0.0448** and **XGBoost 0.0498 / CatBoost 0.0481** beat **TFT 0.0358**, **TabNet 0.0099–0.0204**, and a plain Transformer. Grinsztajn et al. (45 datasets) and Shwartz-Ziv & Armon (11 datasets) both find trees remain SOTA on tabular data. `[Fact, High]` [Qlib](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) · [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) · [arXiv:2106.03253](https://arxiv.org/abs/2106.03253)
9. **Deep nets are a complement, not the base model.** Every recent financial-ML competition winner (Jane Street 2020, Ubiquant 2022, Optiver) used **LightGBM/XGBoost in the winning ensemble**, usually with one NN (MLP/TabNet) as a secondary. Gu, Kelly & Xiu (2020) found NN performance *peaked at 3 hidden layers and declined*. `[Fact, High]` [Jane Street 1st](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s) · [Ubiquant 1st](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy)
10. **Do NOT use zero-shot time-series foundation models** (Chronos/TimesFM/Moirai) as the primary signal. The first comprehensive study in finance found off-the-shelf TSFMs "perform poorly in zero-shot and fine-tuning settings"; only finance-domain pretraining **from scratch** roughly closed the gap to a CatBoost. `[Fact, High]` [arXiv:2511.18578](https://arxiv.org/abs/2511.18578)
11. **Reported 70–98% NSE accuracies are almost certainly leakage.** A 2025 Nifty XGBoost paper claims ~70% next-day accuracy; a CNN-Attention-GRU paper claims **98%** 1-day accuracy. Leak-free daily equity direction is **~51–55%**. Treat anything >60% as a bug until proven with purged, embargoed CV. `[Fact, High]` [TechRxiv 2025](https://doi.org/10.36227/techrxiv.175281141.15130016/v1) · [CAGTRADE 2025](https://ideas.repec.org/a/kap/apfinm/v32y2025i2d10.1007_s10690-024-09463-w.html)
12. **Even the best models explain <1% of daily return variance.** Gu, Kelly & Xiu (2020): monthly out-of-sample R² of **0.40%** for the best neural nets, **−3.46%** for full OLS, and only **0.16%** for OLS-3. Fama (1991) had already established that less than ~1% of daily individual-stock variance is explained by expected returns. `[Fact, High]` [Gu-Kelly-Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)
13. **Validation is where most projects die.** Use **purged K-fold + embargo**, **Combinatorial Purged CV (CPCV)**, rolling/anchored walk-forward, and the **Deflated Sharpe Ratio / PBO**. A controlled simulation showed a random forest reporting CV **IC +0.31 to +0.56** when the true skill was **exactly zero**, purely from overlap + persistence. `[Fact, Med]` [Purged CV paper](https://purged-cv.marketmaker.cc/paper.pdf) · [Bailey et al. 2014](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)
14. **A next-day signal must beat NSE costs.** Round-trip statutory cost ≈ **22 bps** (cash delivery), **~3.4 bps** (intraday equity), **~5.5 bps** (Nifty futures), **~16 bps of premium** (options), before brokerage/slippage/DP. Daily Nifty σ ≈ **129 bps**, so a signal's edge must be a meaningful fraction of that. `[Estimate, Med]` [Zerodha charges](https://zerodha.com/charges/) · [NSE STT/levies](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)
15. **Honest verdict: GO on the pipeline, CONDITIONAL on the target.** Build a **LightGBM/CatBoost** model on **~30–80 decorrelated daily features**, with **volatility + a meta-labelled directional target**, evaluated with **purged CPCV and net-of-cost backtests**. Kill the directional bet if it can't beat the 52.9% base rate plus costs.

---

## 2. The idea, unpacked

### 2.1 What the product is
An ML system that consumes the repo's **NSE daily bhavcopy** (OHLC + volume + trades, 2000→2026) plus the downloaded **indices/macro series** (Nifty, India VIX, S&P 500, Nasdaq, USDINR, Brent, Gold, DXY) and outputs a forecast of the **next trading session** ("tomorrow") for an NSE instrument — as a **return**, a **direction probability**, and/or a **volatility estimate**. Training and testing happen on a separate machine; the immediate goal is a validated research pipeline, not live trading.

### 2.2 Job-to-be-done
- **Primary JTBD:** "From the data I already have, tell me what tomorrow's NSE move most likely is — up/down probability and expected magnitude/uncertainty — reliably enough that it is not just the base rate."
- **Secondary JTBD:** "Give me a leakage-safe, train/test harness I trust, because every tutorial I find reports impossible accuracy and I can't tell what's real."

### 2.3 Ideal customer profile (ICP)
1. **The builder themselves (self-directed quant/ML engineer)** — owns the model, trades own account, lowest legal friction. `[Inference, High]`
2. **Serious Indian retail/algo trader** — pays ₹500–₹5,000/month for tools; skeptical of signal-sellers. `[Fact, Med]`
3. **(Later) signal/API subscribers** — requires SEBI Research Analyst registration + NSE data licensing. Only after an audited track record. `[Inference, High]`

### 2.4 The "trend changes per week / uniform week-wise" question — resolved
You described the data as "mostly trend changes per week or so… uniform week wise." The measurement says:
- The **underlying data is daily**, one row per symbol per session, **6,625 sessions over ~26.76 years ≈ 247.6 sessions/year**, consistent with NSE's ~245–250 trading days/year. `[Fact, High — computed]` `[Fact, Med]` [BSE holiday list](https://www.bseindia.com/static/markets/marketinfo/listholi)
- Sessions per weekday are near-equal (Mon 1,327 / Tue 1,333 / Wed 1,332 / Thu 1,327 / Fri 1,306). `[Fact, High — computed]`
- Weeks: **5 sessions** in 1,075 ISO weeks, **4** in 284, **3** in 38 — so the data *looks* weekly-uniform, but there is no weekly aggregation in the file. `[Fact, High — computed]`
- **Implication for modeling:** do **not** force the problem to a weekly grid. Resample explicitly (daily primary; weekly as a separate, lower-frequency experiment). If you predict "tomorrow," you want the **daily** bar and a **1-session-ahead** target. The apparent weekly rhythm is just the market calendar.

### 2.5 Key assumptions (recorded, not yet validated)
- **A1:** "Predict tomorrow" = next **trading session** (1-session-ahead), not next calendar day. `[Assumption]`
- **A2:** Primary universe = **liquid `EQ` NSE names + Nifty 50 / Bank Nifty**, not illiquid/SME/BE/BZ series. `[Assumption]`
- **A3:** "Predict" = probabilistic direction **and/or** magnitude/volatility — not an exact price level. `[Assumption]`
- **A4:** Budget near-zero; research phase; training on a separate machine. `[Assumption]`
- **A5:** You accept the finding that **daily direction is near the noise floor** and that **volatility + meta-labeling** is the better first target. `[Assumption]`

### 2.6 What would falsify the idea
- No feature set beats an **EWMA/HAR/India-VIX volatility baseline** on out-of-sample realized volatility. → ML premise fails.
- Best **cost-aware, purged-CV, out-of-sample** directional model scores **≤ the 52.9% up-day base rate** (or nets < costs). → Direction bet fails.
- **Train/test shows >60% accuracy** on direction with proper purging → you have a bug (leakage), not a breakthrough. → Re-verify.
- Live/paper trading diverges from backtest by >10% trade match. → Execution invalidates it.

---

## 3. How "trends" actually behave {#3-how-trends-actually-behave}

All numbers in this section were computed on the repo's files on 2026-10-15 (read-only; no model trained).

### 3.1 Measured daily statistics

**Nifty 50** (`data/raw/indices/nifty50.parquet`, Yahoo-sourced, **2007-09-17 → 2026-10-07**, n = 4,674 log-returns):

| Statistic | Value | Read |
|---|---|---|
| Mean daily log-return | **+0.0346%** | small positive drift |
| Daily σ | **1.2926%** | ann. vol **20.52%** |
| Skew | **−0.278** | negative (crash) asymmetry |
| Excess kurtosis | **15.20** | very fat tails (normal = 0) |
| Up-day share | **52.89%** | the base rate |
| ACF(ret) lag 1–10 | **+0.037, −0.006, −0.025, …, +0.030** | ≈ random walk |
| ACF(\|ret\|) lag 1–10 | **+0.279, +0.311, +0.305, …, +0.277** | strong vol clustering |
| Variance ratio (2,5,10,20) | **1.037, 1.020, 0.999, 1.061** | ≈ random walk |
| Hurst (var-of-diff) | **0.513** | ≈ 0.5 = random walk |

`[Fact, High — computed]`

**150 most liquid `EQ` stocks** (pooled 523,066 daily log-returns, corporate-action gaps >±50% removed):

| Statistic | Value | Read |
|---|---|---|
| Mean / daily σ | **+0.0555% / 2.502%** | higher drift & vol than index |
| Pooled up-day share | **49.79%** | equal-weight base rate ≈ 50% |
| Pooled ACF(ret) lag 1 | **+0.0304** | tiny momentum |
| Per-stock ACF(ret) lag 1 | mean **+0.0325**, median **+0.0307**, **76.4% positive** | consistent weak momentum |
| ACF(\|ret\|) lag 1 | **+0.2617** | strong vol clustering |
| **Weekly** return ACF lag 1 | **+0.0027** | no weekly trend |
| Stock-days with \|log-return\| > 25% | **274** | corporate actions / extremes to clean |

`[Fact, High — computed]`

### 3.2 What this means for "trends"
1. **Raw daily returns have almost no linear trend.** Autocorrelation ~0.03, Hurst 0.513, VR ≈ 1.0. Any "trend" is at the noise floor.
2. **Volatility trends and clusters strongly.** |return| autocorrelation ~0.26–0.31 is ~9× the return autocorrelation. High-vol days follow high-vol days.
3. **There is a small positive short-horizon momentum** (lag-1 ≈ +0.03, positive for 76% of stocks). This is the classic "short-term continuation" — too small for single-stock directional *timing* after costs, but usable cross-sectionally and as a feature.
4. **Weekly is *less* predictable than daily here** (weekly ACF 0.003). Reinforces: model the daily horizon.
5. **Fat tails dominate.** A model must predict a distribution with kurtosis ~15, not a bell curve. Expect a handful of days to drive most error.

### 3.3 Calendar effects we measured
- Nifty day-of-week mean log-return: **Mon −0.007%, Tue +0.056%, Wed +0.102%, Thu −0.028%, Fri +0.050%**. Pooled liquid stocks: **Mon −0.001%, Tue +0.067%, Wed +0.137%, Thu +0.026%, Fri +0.049%**. `[Fact, High — computed]`
- **Wednesday is the strongest and Monday the weakest** in this NSE sample. This is a real but tiny and unstable effect; the literature on India is mixed (some find Wednesday, some find Monday) and it should be validated out-of-sample, not assumed. `[Fact, Med]` [Emerald 2023](https://www.emerald.com/insight/content/doi/10.1108/mf-01-2023-0010/full/pdf) · [IJFMR 2026](https://www.ijfmr.com/papers/2026/1/68354.pdf)

### 3.4 Cross-asset signal we measured
(2008-03-07 → 2026-10-06, n = 4,399 aligned sessions) `[Fact, High — computed]`

| Relationship | Correlation |
|---|---|
| Nifty ret(t) ↔ S&P 500 ret(t) (same-day) | **+0.303** |
| Nifty ret(t) ↔ **S&P 500 ret(t−1)** (overnight) | **+0.228** |
| Nifty ret(t) ↔ **Nasdaq ret(t−1)** | **+0.239** |
| Nifty ret(t) ↔ USDINR ret(t−1) | **−0.019** |
| **India VIX level(t) ↔ \|Nifty ret(t+1)\|** | **+0.463** |
| \|Nifty ret(t)\| ↔ \|Nifty ret(t+1)\| | **+0.268** |
| India VIX ret(t) ↔ **Nifty ret(t+1)** | **0.0** |

**This is the single most useful table for feature design:** overnight US equity is a genuine directional feature; India VIX is a genuine *volatility* feature but **not** a directional one; USDINR at daily lag is not useful.

---

## 4. Market dynamics: the theory behind the measurements

### 4.1 Efficient markets and the random walk
- Fama (1970) defined an efficient market as one where prices "fully reflect" available information; the random walk is the strict special case (iid returns). Weak-form tests use only past prices. `[Fact, High]` [Fama 1970](http://gesd.free.fr/fama1970.pdf)
- Fama (1991) reports the **predictable component of returns is a small fraction of daily/weekly variance** (growing to ~40% only at 2–10-year horizons), warns apparent short-horizon predictability may be "data-dredging and chance sample-specific conditions," and notes <1% of individual-stock short-horizon variance is explained by time-varying expected returns. `[Fact, High]` [Fama 1991](https://www.bu.edu/econ/files/2011/01/Fama2.pdf)
- **Matches your data:** VR ≈ 1.0, Hurst 0.513, ACF ≈ 0.03.

### 4.2 Autocorrelation, momentum vs reversal
- Lo & MacKinlay (1988) **reject the random walk for weekly** US returns (1962–1985), but the effect is strongest for **small/illiquid** portfolios (equal-weighted VR 1.30) and weak for the value-weighted index (VR ≈ 1.15); monthly increments are generally not rejected. `[Fact, High]` [NBER w2168](https://www.nber.org/system/files/working_papers/w2168/w2168.pdf)
- Fama (1991): daily individual-stock autocorrelations average ≈0 (negative ones are largely **bid-ask bounce**), and nonsynchronous trading spuriously inflates *portfolio* autocorrelation. `[Fact, High]` [Fama 1991](https://www.bu.edu/econ/files/2011/01/Fama2.pdf)
- Jegadeesh & Titman (1993): buying 3–12-month winners / selling losers earned ~**1.31%/month** (1.49% with a 1-week skip) — but this is a **months**-horizon effect that partially reverses after ~1 year. Lehmann (1990) documents **1-week** reversals (−0.35% to −0.55% for winners). `[Fact, High]` [Jegadeesh & Titman](https://bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf) · [NBER w2533](https://www.nber.org/papers/w2533)
- **Matches your data:** tiny positive lag-1 (momentum) at the daily horizon; ~0 weekly. Cross-sectional momentum is a *feature*, not a single-stock next-day signal.

### 4.3 Volatility clustering (why volatility is the tractable target)
- Engle (1982) introduced ARCH: mean-zero, serially uncorrelated returns whose conditional variance depends on the recent past. Bollerslev (1986) generalised to **GARCH(1,1)**, stationary iff α₁+β₁<1. `[Fact, High]` [Engle 1982](https://www.econometricsociety.org/publications/econometrica/1982/07/01/autoregressive-conditional-heteroscedasticity-estimates) · [Bollerslev 1986](https://doi.org/10.1016/0304-4076(86)90063-1)
- Cont (2001) lists **volatility clustering** as a core stylized fact; the autocorrelation of absolute returns decays slowly as a power law with exponent ~0.2–0.4 ("long memory"). `[Fact, High]` [Cont 2001](https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf)
- Corsi (2009) proposed **HAR-RV** (daily + weekly + monthly realized-vol components), which reproduces long memory parsimoniously and **outperforms short-memory models out-of-sample** at 1-day/1-week/2-week horizons. `[Fact, High]` [Corsi 2009](https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf)
- **Matches your data:** ACF(|r|) ≈ 0.28 persistent; India VIX level ↔ next-day |Nifty| corr 0.463.

### 4.4 Fat tails and non-normality
- Cont (2001): return distributions are heavy-tailed with a finite tail index ~2–5; kurtosis far above the Gaussian value (~16 for S&P 500 futures at 5-min); gain/loss asymmetry; aggregational Gaussianity. `[Fact, High]` [Cont 2001](https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf)
- **Matches your data:** Nifty excess kurtosis 15.20, skew −0.28.

### 4.5 Regime switching
- Hamilton (1989) modelled regime changes as a discrete-state Markov process. `[Fact, High]` [Hamilton 1989](https://users.ssc.wisc.edu/~behansen/718/Hamilton1989.pdf)
- Ahmad & Kamaiah (Nifty/Sensex daily, 1997–2010) identify **persistent bull/bear regimes**: bear ≈ −0.19%/day with high vol, bull ≈ +0.17%/day with low vol; persistence 0.95 (bear, ~20-day duration) and 0.97 (bull, ~39-day duration). `[Fact, High]` [MPRA 37174](https://mpra.ub.uni-muenchen.de/37174/1/MPRA_paper_37174.pdf)
- **Implication:** a model trained across regimes learns an average that fits none. Regime-conditional models or regime features (`is_high_vol`, trend-state dummies) matter.

### 4.6 Calendar / seasonality (India)
- Vipul (2005): Indian underlying prices are **marginally depressed one day before F&O expiration and strengthen the day after**, with abnormally high volume. `[Fact, High]` [J. Futures Markets](https://ideas.repec.org/a/wly/jfutmk/v25y2005i11p1045-1065.html)
- Day-of-week effects exist but are unstable (Wednesday-positive in one study; Monday-positive in another). `[Fact, Med]` [Afro-Asian J.](https://ideas.repec.org/a/ids/afasfa/v5y2015i2p99-112.html) · [Emerald 2006](https://www.emerald.com/ijoem/article/1/3/235/141602/Day-of-the-week-and-other-market-anomalies-in-the)
- **NSE expiry day changed from Thursday to Tuesday** for contracts from 1 Sep 2025 (SEBI's May-2025 single-weekly-expiry rule). Any "expiry-week" dummy must be **contract-aware**, not a fixed weekday. `[Fact, High]` [NSE circular FAOP68747](https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf)
- Market-wide circuit breakers trigger at **10%/15%/20%** on Nifty 50 or Sensex (whichever breaches first); these truncate returns and create halt sessions. `[Fact, High]` [NSE circuit breakers](https://www.nseindia.com/products-services/equity-market-circuit-breakers) · [SEBI 2001 circular](https://www.sebi.gov.in/legal/circulars/jun-2001/index-based-market-wide-circuit-breaker-in-compulsory-rolling-settlement_17986.html)

### 4.7 The honest counter-case (read this before believing any result)
- Welch & Goyal (2008): of 51 annual equity-premium predictors, **46 underperformed the historical mean** out-of-sample. `[Fact, High]` [Welch & Goyal](https://www.ivo-welch.org/research/journalcopy/goyal2008comprehensive.pdf)
- McLean & Pontiff (2016): across 97 published return predictors, average portfolio returns were **26% lower out-of-sample and 58% lower post-publication** (0.582% → 0.264%/month). `[Fact, High]` [McLean & Pontiff](https://gwern.net/doc/economics/2016-mclean.pdf)
- Hou, Xue & Zhang (2020): **65% of 452 anomalies fail |t| ≥ 1.96** and 82% fail |t| ≥ 2.78. `[Fact, High]` [NBER w23394](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf)
- Gu, Kelly & Xiu (2020): best monthly individual-stock R² = **0.40%**; a naive zero forecast beat full OLS. `[Fact, High]` [Gu-Kelly-Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)
- **Net:** the honest prior is that next-day *direction* is a low-probability, high-effort bet. Volatility and risk features are where the ML reliably adds value.

---

## 5. Feature engineering — the statistics toolbox {#5-feature-engineering}

The single most important preparatory fact from the companion dossier still holds: **bhavcopy prices are raw/unadjusted**, so corporate actions create fake returns (the repo already found **2,180 stock-days with a >2× or <0.5× overnight ratio** and **7,949 with a >25% move**). Fixing that mismatch is the highest-value "feature engineering" step, because every return/label is otherwise polluted. `[Fact, High — inspected]` [NSE corporate-action adjustments](https://www.nseindia.com/static/products-services/equity-derivatives-corporate-actions-adjustments) · [TradeEntry note: RELIANCE 2017 1:1 bonus shows prevclose 1645.4 vs close 818.1](https://docs.tradeentry.in/docs/developer/packages/nseeod/nseeod-corp-actions)

NSE's own adjustment factors: **bonus A:B → (A+B)/B**; **split/consolidation A:B → A/B**. Apply the factor to all prices *before* the ex-date; divide volume by the factor; leave turnover alone. `[Fact, High]` [NSE](https://www.nseindia.com/static/products-services/equity-derivatives-corporate-actions-adjustments)

### 5.1 Preprocessing & stationarity statistics
| Statistic / transform | What it answers | Notes / source |
|---|---|---|
| **Log returns** `ln(C_t/C_{t-1})` | makes the series additive and roughly stationary | log returns aggregate over time; simple returns aggregate across assets — don't mix. `[Fact, High]` [RiskMetrics TD](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a) |
| **ADF test** | null = unit root (non-stationary) | reject → stationary; `statsmodels.adfuller`. `[Fact, High]` [Dickey & Fuller 1979](https://doi.org/10.1080/01621459.1979.10482531) · [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) |
| **KPSS test** | null = stationary (opposite of ADF) | use **both** to disambiguate. `[Fact, High]` [KPSS 1992](https://doi.org/10.1016/0304-4076(92)90104-Y) |
| **Phillips–Perron** | unit root, nonparametric | available in `arch.unitroot`. `[Fact, High]` [Phillips & Perron 1988](https://doi.org/10.1093/biomet/75.2.335) |
| **Fractional differencing (FFD)** | stationarity *while preserving memory* | find min `d` s.t. ADF rejects; commonly d*≈0.2–0.6 for prices. `[Fact, High]` [López de Prado AFML Ch.5](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c05.xhtml) |
| **ACF / PACF** | serial correlation at each lag | use to decide momentum vs mean-reversion. `[Fact, High]` [statsmodels](https://www.statsmodels.org/stable/api.html) |
| **Ljung–Box** | joint test that autocorrelations ≤ h are zero | `[Fact, High]` [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html) |
| **Variance ratio** | VR<1 mean-reversion, >1 momentum | Lo & MacKinlay. `[Fact, High]` [RFS 1988](https://doi.org/10.1093/rfs/1.1.41) |
| **Hurst exponent** | H>0.5 persistent, <0.5 mean-reverting | biased/high-variance in small samples — don't over-read. `[Fact, Med]` [Mishra & Mishra 2020](https://journals.sagepub.com/doi/10.1177/0972150920966879) |

**Recommendation:** compute ADF+KPSS on each candidate series; use **fractional differencing** for price levels rather than integer differencing (integer d=1 destroys memory). `fracdiff` (sklearn-style transformer, ~339★, archived) or `mlfinlab`'s `frac_diff_ffd` implement it. `[Fact, Med]` [fracdiff](https://github.com/fracdiff/fracdiff) · [mlfinlab fracdiff](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/features/fracdiff.py)

### 5.2 Volatility features (the highest-signal family)
| Estimator | Definition | Efficiency vs close-to-close |
|---|---|---|
| Close-to-close | standard deviation of returns | 1.0 (baseline) |
| **Parkinson** | σ²=(ln(H/L))²/(4 ln 2) | ~5.2× |
| **Garman–Klass** | 0.5(ln H/L)² − (2ln2−1)(ln C/O)² | ~7.4× |
| **Rogers–Satchell** | ln(H/O)ln(H/C)+ln(L/O)ln(L/C) | ~6–8× |
| **Yang–Zhang** | combines overnight + open-close + RS | up to ~14× |

`[Fact, High]` [bookdown OHLC measures](https://bookdown.org/jarneric/financial_econometrics/5.1-ohlc-measures.html) · [MIT 18.642 L17](https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/mit18_642_f24_lec17_2.pdf)

Plus: **EWMA** (RiskMetrics λ≈0.94), **GARCH(1,1)** (needs `arch`), **HAR-RV** on daily/weekly/monthly RV. `[Fact, High]` [RiskMetrics](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a) · [Corsi 2009](https://doi.org/10.1093/jjfinec/nbp001)

**Grounding:** India VIX level ↔ next-day |Nifty| corr **0.463**, and today's |return| ↔ tomorrow's **0.268** — these are the strongest single-feature relationships in your data. `[Fact, High — computed]`

### 5.3 Momentum / technical indicators (with evidence)
| Indicator | Formula sketch | Evidence verdict |
|---|---|---|
| Moving-average crossover | fast vs slow SMA/EMA | Brock et al. (1897–1986) found predictive value; Bajgrowicz & Scaillet found it offset by costs. `[Fact, High]` [BLL 1992](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x) · [Bajgrowicz & Scaillet 2012](https://doi.org/10.1016/j.jfineco.2012.06.001) |
| RSI (14) | 100 − 100/(1+RS) | oscillator — most likely pure noise after costs. `[Opinion, Med]` [TTR](https://cran.r-project.org/web/packages/TTR/refman/TTR.html) |
| MACD (12,26,9) | EMA12 − EMA26, 9-EMA signal | momentum proxy; keep, but expect small IC. `[Fact, High]` [TTR](https://cran.r-project.org/web/packages/TTR/refman/TTR.html) |
| Bollinger Bands (20,2) | SMA ± 2σ + %B width | band **width** is a vol feature; fixed thresholds are noise. `[Fact, High]` [TTR](https://cran.r-project.org/web/packages/TTR/refman/TTR.html) |
| ATR (14) | Wilder-smoothed true range | good volatility/risk feature. `[Fact, High]` [TTR](https://cran.r-project.org/web/packages/TTR/refman/TTR.html) |
| Rate-of-change / time-series momentum | return over k days | strongest academic support. `[Fact, High]` [Moskowitz et al. 2012](https://doi.org/10.1016/j.jfineco.2011.11.003) |

**Recommendation:** include MA-distance, ROC, ATR, band-width; include RSI/stochastic but expect them to be selected away.

### 5.4 Volume / liquidity features
- **Amihud illiquidity** = mean(|r| / dollar volume) — price impact per rupee traded; predicts cross-sectional returns. `[Fact, High]` [Amihud 2002](https://doi.org/10.1016/S1386-4181(01)00024-6)
- **Turnover** (shares / outstanding) — liquidity measure; **OBV**, **VWAP deviation**, **volume z-score** = (vol − rolling mean)/rolling std.
- **Delivery %** — bhavcopy lacks it, but NSE's `sec_bhavdata_full` carries `DELIV_QTY`/`DELIV_PER`; add it if you want a genuinely differentiating feature. `[Fact, High]` [NSE security-wise archives](https://www.nseindia.com/report-detail/eq_security)
- All rolling windows must end at **t−1** to avoid look-ahead.

### 5.5 Cross-sectional / relative features (likely the real edge)
- **Rank / z-score a stock's momentum, value-proxy, liquidity across the universe each day.**
- **Relative strength vs Nifty** (stock return − index return over a window) and **rolling beta**.
- India momentum works cross-sectionally: winner-minus-loser strategies earn abnormal profits; **time-series momentum generally beats cross-sectional** after size/value adjustment. `[Fact, Med]` [Mohapatra & Misra 2020](https://www.sciencedirect.com/science/article/pii/S0970389617301647) · [Singh & Walia 2021](https://apcz.umk.pl/CJFA/article/view/CJFA.2020.018)
- Aim: **predict *relative* next-day performance**, then go long the top decile / short the bottom (market-neutral). This sidesteps the "always up" base rate entirely.

### 5.6 Cross-asset / macro features (India)
- **Overnight US equities are the headline feature:** S&P 500 lag-1 corr **+0.228**, Nasdaq lag-1 **+0.239** with next Nifty session. `[Fact, High — computed]`
- **India VIX** (level, change, term structure if available) — volatility, not direction. Use as a vol feature / regime variable. `[Fact, High]` [NSE India VIX methodology](https://nsearchives.nseindia.com/web/sites/default/files/inline-files/India_VIX_comp_meth.pdf)
- **FII/DII daily flows** are published free by NSE; evidence on flow→return is mixed (reverse causality; DIIs act counter-cyclically). Treat as experimental. `[Fact, Med]` [NSE FII/DII](https://www.nseindia.com/reports/fii-dii)
- Brent, gold, DXY, USDINR: weak/indirect at daily lag. Include but expect low importance. `[Inference, Low]`

### 5.7 Calendar / event features
- Day-of-week, month, turn-of-month, pre/post-holiday, **expiry week (contract-aware, post-2025 Tuesday expiry)**, results season, Budget day, Muhurat. Compute as dummies/flags and validate out-of-sample. `[Fact, Med]` [NSE FAOP68747](https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf)

### 5.8 Alternative data (optional, later)
- Options-implied (IV skew, put–call ratio, OI) predicts returns but **daily IC is near-zero** (|IC|<0.02); weekly is better. `[Estimate, Low]` [Boucher preprint](https://www.spboucher.ai/papers/wp7_uqo.pdf)
- News/sentiment gives large claimed Sharpes but leak-free tests collapse to ~54%. Treat with extreme skepticism. `[Fact, Med]` [CS224n report](https://web.stanford.edu/class/cs224n/final-reports/256968470.pdf)

### 5.9 Feature selection & importance
- **IC / Rank IC** = (rank) correlation of score with realized return; **ICIR = mean(IC)/std(IC)**. Qlib reports all four. `[Fact, High]` [Qlib](https://qlib.readthedocs.io/en/latest/component/report.html)
- **Mutual information** captures nonlinear dependence. `[Fact, High]` [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
- **Permutation importance** (on held-out, purged folds) beats impurity/gain importance, which is biased toward high-cardinality features. `[Fact, High]` [sklearn](https://scikit-learn.org/stable/modules/permutation_importance.html)
- **SHAP** for additive attribution (local accuracy, missingness, consistency). `[Fact, High]` [Lundberg & Lee 2017](https://arxiv.org/abs/1705.07874)
- **Cluster correlated features**, keep one representative — otherwise importance is double-counted. `[Fact, Med]` [sklearn](https://scikit-learn.org/stable/modules/permutation_importance.html)

### 5.10 Scaling & leakage
- Fit scalers on the **training fold only**; use **trailing/expanding** windows for z-scoring. Tree models don't need scaling. `[Fact, High]` [sklearn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- Classic traps: full-sample normalization; using same-day H/L/C to predict same-day C; target leakage; survivorship. `[Fact, High]` [sklearn](https://scikit-learn.org/stable/common_pitfalls.html)
- **NSE-specific trap:** restated `PREV_CLOSE` on ex-dates + the 2024 UDiFF date-format change are the two largest sources of spurious return outliers across 2000–2026. `[Inference, Med]`

### 5.11 Target / label engineering (this matters more than the model)
| Target | Pros | Cons |
|---|---|---|
| Next-day return (regression) | magnitude, IC-friendly | tiny SNR; report IC not R². `[Inference, Med]` |
| Binary direction | simple | scale-free, threshold-sensitive; base-rate-dominated. `[Fact, Med]` |
| Ternary (−1/0/+1) | adds "flat" class | static threshold ignores heteroskedasticity. `[Fact, Med]` [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) |
| **Volatility** (next-day RV / range) | **far higher SNR** (vol is persistent) | not directly a trade signal. `[Fact, Med]` |
| **Triple-barrier + meta-labeling** | upper/lower/time barriers scaled by vol; secondary model filters false positives | best-in-class for direction. `[Fact, High]` [mlfinlab](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py) · [Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/) |

**Recommendation:** build **two targets** — (a) next-day **volatility** (high SNR, validates the pipeline), and (b) a **meta-labelled** triple-barrier direction target (the actual bet).

### 5.12 How many features?
- Qlib's shipped benchmarks are **Alpha158** (158 hand-engineered) and **Alpha360** (raw prices/volume). Even the best achieve only **IC ≈ 0.05–0.11**. Marginal IC diminishes sharply. `[Fact, High]` [Qlib](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
- **Recommendation:** start with **~30–80 decorrelated features** (fractionally-differenced price, momentum/ROC, MA-distance, OHLC volatility estimators, ATR, vol-regime, volume/liquidity, cross-sectional rank, overnight-US, VIX, calendar). Add more only if purged CV says they help.

---

## 6. Which ML model gives the best next-day accuracy {#6-model-selection}

### 6.1 The verdict
**Tuned gradient-boosted trees (LightGBM first; CatBoost/XGBoost close) on tabular financial features.** Add a compact NN (MLP/GRU/TCN) only as an **ensemble complement**. Avoid: deep sequence models as the primary, LLM/foundation time-series models, and SVM/kNN as the main model.

### 6.2 The evidence (exact numbers)
**Microsoft Qlib benchmark — China A-shares, CSI300, mean over 20 seeds** `[Fact, High]` [Qlib README](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md):

| Model | Dataset | IC | Rank IC | Annualized Return |
|---|---|---:|---:|---:|
| **XGBoost** | Alpha158 | **0.0498** | 0.0505 | 0.0780 |
| **CatBoost** | Alpha158 | **0.0481** | 0.0454 | 0.0765 |
| **LightGBM** | Alpha158 | **0.0448** | 0.0469 | **0.0901** |
| TRA | Alpha158 | 0.0440 | 0.0540 | 0.0718 |
| Linear | Alpha158 | 0.0397 | 0.0472 | 0.0692 |
| MLP | Alpha158 | 0.0376 | 0.0429 | 0.0895 |
| **TFT** | Alpha158 (20 sel.) | **0.0358** | 0.0116 | 0.0847 |
| ALSTM | Alpha158 (20 sel.) | 0.0362 | 0.0463 | 0.0470 |
| GRU | Alpha158 (20 sel.) | 0.0315 | 0.0428 | 0.0344 |
| TCN | Alpha158 | 0.0279 | 0.0421 | 0.0262 |
| Transformer | Alpha158 | 0.0264 | 0.0407 | 0.0273 |
| **TabNet** | Alpha158 | **0.0204** | 0.0333 | 0.0227 |
| **TabNet** | Alpha360 | **0.0099** | 0.0290 | **−0.0369** |
| Transformer | Alpha360 | 0.0114 | 0.0327 | −0.0270 |
| LightGBM | Alpha360 | 0.0400 | 0.0536 | 0.0505 |

**Reading:** the tree models top the IC table; the deep models (TFT, TabNet, Transformer) are *not* better, and some are negative on Alpha360. **DoubleEnsemble (LGBM-based) reached IC 0.0521 / ann. 0.1158** — the best overall — which is a tree ensemble. `[Fact, High]`

**Why trees win on tabular data** `[Fact, High]`:
- Grinsztajn et al. (45 datasets, 20,000 GPU-hr search per learner): tree-based models remain SOTA on medium data (~10K samples); NNs are biased toward over-smooth solutions and are less robust to uninformative features. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
- Shwartz-Ziv & Armon (11 datasets): XGBoost beat deep models on 8/11; averaged relative error — ensemble 2.32%, **XGBoost 3.4%**, 1D-CNN 7.5%, TabNet 10.5%, NODE 14.2%. [arXiv:2106.03253](https://arxiv.org/abs/2106.03253)
- TabZilla (176 datasets): GBDTs do comparatively **better on larger, higher size-to-feature-ratio, heavy-tailed** data — exactly your regime. [arXiv:2305.02997](https://arxiv.org/abs/2305.02997)
- Gu, Kelly & Xiu: NN performance peaked at **3 hidden layers and declined**; trees and NNs both "unambiguously" improved prediction, but trees were strong. [Gu-Kelly-Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)

**Competition evidence — what actually won money** `[Fact, High]`:
- **Jane Street 2020 (1st):** supervised-autoencoder MLP blended with **XGBoost**; 5-fold 31-day purged group time-series split. [writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- **Ubiquant 2022 (1st):** **LGBM + TabNet** blend (5 folds each); 2nd/3rd/7th were LightGBM variants with **purged K-fold + embargo**. [1st](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy) · [2nd](https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/davide-stenner-2nd-place-solution-robust-cv-and-lg)
- **Optiver (realized-vol):** top solutions stacked **LGBM + NN**. [7th](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/writeups/michael-poluektov-7th-place-main-ideas)
- **Pattern:** GBDT is in essentially every winning financial-ML ensemble; the NN is a complement.

### 6.3 Model comparison table

| Model family | Next-day equity suitability | Verdict |
|---|---|---|
| **LightGBM** | best all-round on noisy tabular data; fast; handles missing/categorical | **Primary** `[Fact, High]` |
| **CatBoost** | strong on imbalance; ordered boosting; little tuning needed | **Strong 2nd / co-primary** `[Fact, High]` |
| **XGBoost** | tops Qlib IC (0.0498); mature | **Strong 3rd** `[Fact, High]` |
| Random Forest | good but usually below boosted; noisier | Baseline `[Fact, Med]` |
| Linear (ridge/lasso/elastic net) | weak but interpretable; OLS full-feature was negative | Baseline only `[Fact, High]` |
| MLP / GRU / TCN | can beat GBDT on Alpha360-style raw sequences; as **ensemble member** helps | Complement `[Fact, High]` |
| TFT / PatchTST / iTransformer / Informer | designed for forecasting; not better than GBDT here; need classification head | Optional `[Fact, Med]` |
| LSTM per-stock | popular in tutorials; routinely overfit; no clean edge | Avoid as primary `[Fact, Med]` |
| Zero-shot Chronos/TimesFM/Moirai | "perform poorly in zero-shot and fine-tuning"; only from-scratch finance pretraining helps | **Avoid** `[Fact, High]` [arXiv:2511.18578](https://arxiv.org/abs/2511.18578) |
| SVM / kNN / Gaussian process | SVM sometimes tops *mean* accuracy across 55 markets (51.6%) but nothing decisive | Not primary `[Fact, Med]` [IJDSA 2025](https://link.springer.com/article/10.1007/s41060-025-00854-4) |
| **GBDT + NN ensemble** | best documented combination | **Target architecture** `[Fact, High]` |

### 6.4 Class imbalance (direction models)
- Do **not** use SMOTE. Elor & Averbuch-Elor (73 datasets): balancing did not improve XGBoost/CatBoost; CatBoost *without* balancing was best; **tuning the decision threshold** matched balancing and was simpler. `[Fact, High]` [arXiv:2201.08528](https://arxiv.org/abs/2201.08528)
- SMOTE ignores temporal ordering. `[Fact, Med]` [T-SMOTE IJCAI 2022](https://www.ijcai.org/proceedings/2022/0334.pdf)
- **Use class weights or, preferably, threshold tuning + probability calibration.**

### 6.5 Hyperparameter tuning
- **Random search beats grid search** under a fixed budget. `[Fact, High]` [Bergstra & Bengio 2012](https://jmlr.org/papers/v13/bergstra12a.html)
- **Optuna TPE + a pruner** (MedianPruner/ASHA) is the recommended approach, optimizing a **purged-CV objective**. `[Fact, Med]` [Optuna](https://arxiv.org/abs/1907.10902)
- Guard against HPO overfitting: every extra trial inflates apparent gains; log the number of trials for the Deflated Sharpe adjustment. `[Fact, Med]` [Bailey et al. 2014](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)

---

## 7. The accuracy trap — why most "90% NSE models" are wrong {#7-the-accuracy-trap}

### 7.1 The base rate is the benchmark
Nifty up-day share = **52.89%** (and 43.0–58.8% by year). Pooled liquid stocks = **49.79%**. An "always up" rule therefore scores ~52.9% on the index. **Accuracy below ~53% (index) or ~50% (stocks) is worse than doing nothing.** `[Fact, High — computed]`

### 7.2 Accuracy is dominated by the majority class
- The accuracy paradox is formalized in Valverde-Albacete & Peláez-Moreno (2014); a majority classifier scores high when one class dominates. `[Fact, High]` [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0084217)
- Example: with 990 genuine / 10 fraud, an all-genuine classifier scores 99% and detects zero fraud. `[Fact, Med]` [SAS](https://support.sas.com/resources/papers/proceedings17/0942-2017.pdf)
- **Use balanced accuracy, MCC, AUC, Brier/log-loss, precision@k — not raw accuracy.** `[Fact, High]` [sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html)

### 7.3 The published NSE numbers are implausible
| Claimed result | Source | Assessment |
|---|---|---|
| Nifty XGBoost ~70% next-day accuracy (2015–2025) | TechRxiv 2025 | **Flag: likely leaky** — no purged/embargoed CV shown `[Fact, Low]` |
| CNN-Attention-GRU **98%** 1-day accuracy | CAGTRADE, Asia-Pacific Fin. Markets 2025 | **Flag: implausible** — leak-free daily equity is ~51–55% `[Fact, Low]` |
| h2o deep learning 74.3% Nifty, 88.5% IPC | E-7 country study 2025 | **Flag: 80/20 random split, long horizon; likely regime/leakage** `[Fact, Low]` |
| India VIX + candlestick next-day gap: 53.1% / 52.8% / **54.4%** | IJFMR 2025 (3,802 obs) | **Realistic** — matches the noise-floor evidence `[Fact, Med]` |
| Leak-free US daily direction: CatBoost 51.16%, finance-pretrained Chronos 51.74% | arXiv:2511.18578 (secondary summary) | **Realistic ceiling** `[Fact, Med]` |

`[Fact, High]` [TechRxiv](https://doi.org/10.36227/techrxiv.175281141.15130016/v1) · [CAGTRADE](https://ideas.repec.org/a/kap/apfinm/v32y2025i2d10.1007_s10690-024-09463-w.html) · [e7](https://ideas.repec.org/a/acc/malfin/v40y2025i123p92-111.html) · [IJFMR](https://doi.org/10.36948/ijfmr.2025.v07i04.53859)

### 7.4 The realistic targets (state these before you start)
| Target | Realistic OOS range |
|---|---|
| Next-day **volatility** (RV/range) | R² meaningfully positive; beats HAR/EWMA baseline `[Inference, High]` |
| Next-day **direction** (index) | **51–55%**, base rate 52.9% `[Estimate, Med]` |
| Cross-sectional **Rank IC** (daily) | **0.02–0.05**; >0.10 = bug `[Opinion, Med]` |
| **ICIR** | >0.5 is a quality bar `[Opinion, Med]` |
| Net-of-cost Sharpe | modest; needs to beat 0 meaningfully `[Inference, Med]` |

`[Fact, High]` [Qlib](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) · [Gu-Kelly-Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)

### 7.5 Why it's hard — the honest summary
- Gu, Kelly & Xiu: best monthly individual-stock R² = **0.40%**; full OLS = **−3.46%**. `[Fact, High]`
- Fama (1991): <1% of short-horizon individual-stock variance explained. `[Fact, High]`
- McLean & Pontiff: 58% of published edge decays post-publication. `[Fact, High]`
- SEBI: **93% of individual F&O traders lost money FY22–FY24**, aggregate **₹1.8 lakh crore**; only **7.2%** made a profit; **1%** earned >₹1 lakh after costs. `[Fact, High]` [SEBI PR 23-Sep-2024](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years%5F86906.html) · [Reuters](https://www.reuters.com/world/india/indias-retail-derivatives-traders-lost-18-trln-rupees-three-years-regulator-says-2024-09-23/)

---

## 8. Validation & testing protocol (you have the data; don't fool yourself) {#8-validation-protocol}

This is the section that decides whether your "nice testing" produces truth or fiction.

### 8.1 Leakage sources
- Standard k-fold assumes iid; overlapping forward labels + persistent features leak. `[Fact, High]` [Purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)
- Controlled simulation: with overlapping labels and an AR(1) φ=0.97 feature, a random forest reports CV **IC +0.31/+0.46/+0.56** (h=5/10/20) when **true skill is exactly zero**; AUC 0.623 claimed vs 0.504 true. `[Fact, Med]` [Purged CV paper](https://purged-cv.marketmaker.cc/paper.pdf)
- Naive shuffled k-fold returned R²≈0.83–0.91 on an unpredictable target; collapsed below predict-the-mean once purged. `[Fact, Med]` [github](https://github.com/eslazarev/purged-cross-validation)

### 8.2 The protocol (non-negotiable)
1. **Purged K-fold + embargo.** Purge training samples whose label horizon overlaps the test; embargo a buffer (1–2% for liquid markets) after each test fold. `[Fact, High]` [Purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)
2. **Combinatorial Purged CV (CPCV)** for multiple backtest paths. N=10 groups, k=3 test → 120 splits, 36 paths. `[Fact, Med]` [skfolio CPCV](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)
3. **Walk-forward** (anchored or rolling) for the final model — not random splits. Rolling deployed best in a controlled study (mean forward Sharpe 0.66 vs 0.62 anchored). `[Fact, Med]` [WFO paper](https://wfo.marketmaker.cc/paper.pdf)
4. **Deflated Sharpe Ratio (DSR) / Probabilistic Sharpe Ratio (PSR) / PBO.** DSR adjusts for multiple trials + skew + kurtosis + sample length. `[Fact, High]` [Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) · [PBO](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
5. **Minimum Backtest Length:** with only 5 years of data, **≤45 independent configurations** before OOS expected Sharpe ≈ 0. You have ~26 years — far better, but still count your trials. `[Fact, High]` [Bailey et al. 2014](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)
6. **Noise-controlled PSR:** naive PSR>0.95 fired in 60/200 datasets with no signal; DSR>0.95 fired in 0/200. `[Fact, Med]` [open-quant](https://open-quant.github.io/openquant/runbooks/cpcv-deflated-sharpe/)

### 8.3 Libraries to use
| Library | What | License / notes |
|---|---|---|
| `skfolio` | `CombinatorialPurgedCV` (purged_size, embargo_size) | BSD-3, ~2.5k★ `[Fact, High]` [repo](https://github.com/skfolio/skfolio) |
| `timeseriescv` | `PurgedWalkForwardCV`, `CombPurgedKFoldCV` | MIT `[Fact, High]` [repo](https://github.com/sam31415/timeseriescv/) |
| `mlfinlab` | fracdiff, labeling, purged CV | **not open source** (all-rights-reserved) `[Fact, High]` [repo](https://github.com/hudson-and-thames/mlfinlab/) |
| `vectorbt` | walk-forward + PBO/DSR robustness | open; ~8.4k★ `[Fact, High]` [repo](https://github.com/polakowo/vectorbt/) |
| `backtrader` / `zipline-reloaded` / `bt` | backtesting engines | open `[Fact, High]` |

**Warning:** verify any implementation — a documented `mlfinlab` issue shows its `PurgedKFold` can leave training events that overlap the test window. `[Fact, Med]` [mlfinlab #295](https://github.com/hudson-and-thames/mlfinlab/issues/295)

### 8.4 Metrics to report (not accuracy)
| Metric | Why |
|---|---|
| **AUC / ROC** | threshold-free ranking of direction. `[Fact, High]` |
| **Balanced accuracy / MCC** | imbalance-robust. `[Fact, High]` |
| **Brier / log-loss** | calibration + discrimination. `[Fact, High]` |
| **Rank IC / ICIR** | signal quality (quant standard). `[Fact, High]` |
| **precision@top-k** | what you'd actually trade. `[Fact, Med]` |
| **Net-of-cost Sharpe + turnover** | the only one that matters for money. `[Fact, Med]` |

`[Fact, High]` [sklearn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) · [Qlib report](https://qlib.readthedocs.io/en/latest/component/report.html)

### 8.5 Sanity flags
- If the best strategy's cost-adjusted Sharpe > 2× an equal-weight benchmark, **suspect a glitch**. `[Opinion, Med]` [mlfactor](http://www.mlfactor.com/backtest.html)
- If best Sharpe > 2× a random baseline after purging, suspect a bug. `[Opinion, Med]`
- If accuracy >60% on daily direction, suspect leakage. `[Inference, High]`

---

## 9. Costs — what a next-day signal must beat {#9-costs}

Round-trip statutory cost estimates (excluding per-order brokerage, DP, slippage), observed 2026-10-15:

| Instrument | Round-trip statutory cost | Main components |
|---|---|---|
| **Cash equity (delivery)** | **≈ 22 bps** | STT 0.1% buy + 0.1% sell; ETC ~0.6 bp; stamp 1.5 bp buy; SEBI 0.01 bp |
| **Cash equity (intraday)** | **≈ 3.4 bps** | STT 0.025% sell; ETC; stamp 0.3 bp |
| **Nifty/BankNifty futures** | **≈ 5.5 bps** | STT 0.05% sell (Apr-2026); ETC ~0.35 bp; stamp 0.2 bp |
| **Options** | **≈ 16 bps of premium** | STT 0.15% sell + 0.15% intrinsic on exercise; ETC 0.03553% |

`[Estimate, Med]` Components verified: [Zerodha charges](https://zerodha.com/charges/) · [NSE STT/levies](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)

Key facts:
- Equity **delivery** STT = **0.1% on both buy and sell** (20 bps round trip). `[Fact, High]` [Zerodha](https://zerodha.com/charges/)
- F&O futures STT = **0.05% on the sell side** (up from 0.0125% pre-Oct-2024). `[Fact, High]` [Zerodha](https://zerodha.com/charges/)
- Options STT = **0.15% on the sell side on premium**; exchange transaction charge NSE options = **0.03553% on premium**. `[Fact, High]` [Zerodha](https://zerodha.com/charges/)
- GST 18% on (brokerage + SEBI + transaction charges); SEBI ₹10/crore; stamp duty buy-side only. `[Fact, High]`
- NSE impact cost for the Nifty 50 basket is "0.1% or so"; Nifty-50 eligibility requires impact cost <0.50% for a ₹10-crore basket on ≥90% of observations. `[Fact, High]` [NSE impact cost](https://www.nseindia.com/static/products-services/indices-impact-cost) · [NSE FAQ](https://www.nseindia.com/static/products-services/indices-faqs)
- Illiquid SME names average impact cost ≈ **16.45%** — never model those as tradeable. `[Fact, Med]` [IJLTEMAS](https://www.ijltemas.in/DigitalLibrary/Vol.6Issue9/04-13.pdf)
- Cost drag ≈ turnover × cost; a 5.58× annual turnover at 5 bps cuts ~0.28%/yr and Sharpe by ~0.022. `[Estimate, Med]` [quantjourney](https://backtester.quantjourney.cloud/research/execution-costs)

**Implication:** delivery equity needs the signal to clear ~22 bps; intraday/futures ~3–6 bps. Since daily Nifty σ ≈ 129 bps, a direction model with AUC ~0.55 might net a few bps/trade — which is exactly why the base rate, threshold, and turnover discipline dominate outcomes.

---

## 10. Technical feasibility & build plan {#10-technical-feasibility}

### 10.1 Proposed architecture
```
bhavcopy.parquet (daily) ──┐
indices/*.parquet ─────────┤
                           ├─► [Corporate-action adjust] ─► [Feature builder]
security_master.parquet ───┘                                       │
                                                                   ▼
                                        ~30–80 features per (symbol, date):
                                        fracdiff price, ROC, MA-dist, ATR,
                                        GK/YZ vol, EWMA vol, vol-regime,
                                        volume z, Amihud, cross-sectional rank,
                                        S&P500_lag1, Nasdaq_lag1, IndiaVIX,
                                        day-of-week, expiry-flag
                                                                   │
                                                                   ▼
                       Targets: (a) next-day RV   (b) meta-labelled triple-barrier
                                                                   │
                                                                   ▼
                  LightGBM / CatBoost  (+ optional MLP/GRU in ensemble)
                                                                   │
                                                                   ▼
              Purged CPCV + embargo → walk-forward → DSR/PBO → net-of-cost backtest
```

### 10.2 Difficulty rating
| Component | Difficulty (1–10) | Notes |
|---|---|---|
| Load & clean bhavcopy (EQ filter, CA adjust) | **6** | CA adjustment is the hard part `[Inference, High]` |
| Feature builder (30–80 features) | **4** | standard pandas/ta-lib work |
| Targets (vol + triple-barrier + meta-label) | **6** | triple-barrier + purging is fiddly |
| Baseline models | **2** | LightGBM is easy |
| Leakage-safe validation (CPCV/embargo) | **7** | where most people fail |
| Net-of-cost backtest | **5** | needs turnover + cost model |
| Regime robustness / ensembles | **7** | the real research |

**MVP effort:** ~**2–4 person-weeks** for a clean volume/volatility model + leakage-safe harness; **+4–8 weeks** for a credible directional/meta-label study.

### 10.3 Dependencies & build-vs-buy
- **Build:** feature builder, CA adjustment, validation harness.
- **Buy/borrow:** LightGBM, CatBoost, XGBoost, Optuna, SHAP, `arch` (GARCH), `fracdiff`, `skfolio`/`timeseriescv`, `vectorbt`.
- **Missing from `.venv`** (from the companion dossier): sklearn, lightgbm, xgboost, catboost, optuna, shap, statsmodels, arch, fracdiff, sktime, torch. `[Fact, High — inspected]`

### 10.4 Data dependencies & quirks (NSE)
- **Bhavcopy prices are raw/unadjusted** — corporate actions must be applied before any return/label. `[Fact, High]` [NSE CA adjustments](https://www.nseindia.com/static/products-services/equity-derivatives-corporate-actions-adjustments)
- **Series matters:** `EQ` = normal; `BE`/`BZ` = trade-to-trade (compulsory delivery, no intraday); series switches (EQ→BE) create holes. Filter to `EQ`. `[Fact, Med]` [NSE legend of series](https://www.nseindia.com/market-data/legend-of-series)
- **Price bands:** 2/5/10/20% or no-band-with-derivatives; market-wide circuit breakers at 10/15/20% (Nifty or Sensex first). These **truncate returns** and create non-trading days. `[Fact, High]` [NSE price bands](https://www.nse.in/static/products-services/equity-market-price-bands) · [NSE circuit breakers](https://www.nseindia.com/products-services/equity-market-circuit-breakers)
- **Survivorship:** bhavcopy includes delisted names (good), but the HF minute set does not. Use bhavcopy as the source of truth. `[Fact, High — inspected]`
- **Expiry day changed** (Thu→Tue from Sep-2025) — dummy flags must be contract-aware. `[Fact, High]` [NSE FAOP68747](https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf)

---

## 11. Business & pricing (brief) {#11-business--pricing}

The full competitive/pricing analysis lives in [`nse-intraday-ml-forecasting.md`](./nse-intraday-ml-forecasting.md). Summary relevant here:
- Retail tooling prices cluster at **₹250–₹2,500/month**; serious algotraders pay **₹500–₹5,000/month**. `[Fact, Med]`
- Selling *signals* requires **SEBI Research Analyst registration**; black-box algo providers must register as RAs under the Feb-2025 framework. `[Fact, High]` [SEBI algo circular](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html)
- The category is polluted by unaudited high-accuracy claims; an **audited, timestamped track record** is the differentiator. `[Inference, High]`
- **For this project, treat it as self-directed research** — the lowest-legal-friction path.

---

## 12. Differentiation, scope & the sharpest wedge {#12-differentiation--scope}

### 12.1 Where we can win
1. **Credibility by validation.** Almost nobody publishes a *purged-CPCV, net-of-cost, deflated-Sharpe* NSE next-day result. Doing this honestly is itself novel and defensible.
2. **Volatility + risk-first.** Everyone chases direction; volatility is genuinely forecastable and useful (position sizing, risk).
3. **Cross-sectional, market-neutral framing** sidesteps the base-rate problem and is where documented India momentum exists.

### 12.2 The sharpest wedge (MVP)
> **"A leakage-safe next-day *risk + rank* engine on 26 years of NSE bhavcopy: predict which liquid names will be unusually volatile and which will relatively outperform/underperform, validated with CPCV and net-of-cost backtests."**

### 12.3 Smallest viable MVP
1. Clean `EQ` bhavcopy + corporate-action adjustment.
2. ~40 features + 2 targets (next-day RV; meta-labelled 1-day direction).
3. LightGBM baseline + CatBoost.
4. Purged CPCV + walk-forward + DSR/PBO.
5. A single honest scoreboard vs base rate and HAR/EWMA.

### 12.4 What NOT to build (yet)
- Deep LSTM/Transformer per stock. `[Inference, High]`
- Zero-shot foundation models. `[Fact, High]`
- Intraday execution / live trading. `[Inference, High]`
- Selling signals. `[Inference, High]`
- Tick-level data pipelines. `[Inference, Med]`

### 12.5 Potential moats
- A clean, point-in-time, CA-adjusted NSE dataset + reproducible harness (persistent asset).
- A documented, deflated out-of-sample track record (credibility).
- Cross-sectional feature library tuned to India.

---

## 13. Risks & unknowns {#13-risks--unknowns}

| Risk | Severity × Likelihood | Notes |
|---|---|---|
| **No tradeable directional edge** | High × High | The default outcome; vol is the fallback. `[Fact, High]` |
| **Leakage inflates results** | High × High | Purged CPCV non-negotiable. `[Fact, High]` |
| **Corporate-action corruption** | High × Med | 274 extreme stock-days found; fix before labels. `[Fact, High — computed]` |
| **Regime shift / non-stationarity** | High × High | 2000–2026 spans dot-com, GFC, COVID, 2024–26. `[Fact, High]` |
| **Costs eat the edge** | High × High | 5.5–22 bps vs 129 bps daily σ. `[Estimate, Med]` |
| **Survivorship / point-in-time** | Med × Med | bhavcopy includes delisted; use it. `[Fact, High]` |
| **Overfitting via HPO** | Med × High | Count trials; use DSR/PBO. `[Fact, High]` |
| **Data licensing/redistribution** | Med × Low (research) | NSE forbids redistribution. `[Fact, High]` |
| **Halted/limit days** | Med × Med | Truncated returns; flag them. `[Fact, Med]` |

---

## 14. Opportunities & novel additions {#14-opportunities--novel-additions}

1. **Volatility-targeted next-day risk model** — highest SNR, immediately useful for sizing.
2. **Cross-sectional long/short ranks** — market-neutral, base-rate-free.
3. **Meta-labelling** to improve a weak primary signal's precision. `[Fact, High]`
4. **Regime-conditional models** (bull/bear/high-vol) using India VIX + trend state. `[Fact, Med]`
5. **Delivery-% features** from `sec_bhavdata_full` (bhavcopy lacks them) — genuinely differentiating. `[Fact, High]`
6. **Overnight-gap model** using S&P 500/Nasdaq lag-1 (corr 0.23) — a clean, documented feature. `[Fact, High — computed]`
7. **Expiry-aware features** post the 2025 Tuesday-expiry change. `[Fact, High]`
8. **Publish the honest scoreboard** — a credibility asset in a mistrustful market. `[Inference, High]`

---

## 15. Recommended next steps {#15-recommended-next-steps}

Cheapest → most expensive:
1. **Fix prices first.** Build corporate-action adjustment; re-check the 274 extreme stock-days and the 2,180 abnormal overnight ratios. (days)
2. **Build a volatility baseline.** HAR/EWMA + India VIX → predict next-day |return|. Confirm R² > 0. (days)
3. **Build ~40 features + LightGBM** on next-day volatility with **purged CPCV**. (1 week)
4. **Add the meta-labelled directional target** and evaluate with AUC, balanced accuracy, Rank IC, **and net-of-cost Sharpe** vs the 52.9% base rate. (1–2 weeks)
5. **Add CatBoost/XGBoost + one NN ensemble**; run Optuna with a purged-CV objective; compute DSR/PBO. (1–2 weeks)
6. **Pre-register** the final protocol and evaluate on a held-out recent period (e.g. 2024–2026) you never touched. (days)
7. **Kill criterion:** if the directional overlay doesn't beat base rate + costs on the held-out period, ship the volatility/risk model instead.

---

## 16. Open questions for planning {#16-open-questions-for-planning}

1. **Universe:** Nifty 50 / Bank Nifty only, or the full liquid `EQ` cross-section? (Affects target and CV design.)
2. **Target priority:** volatility-first, or direction-first? (Recommend volatility.)
3. **Instrument for costs:** cash delivery (~22 bps) or futures (~5.5 bps)? (Changes what counts as edge.)
4. **Horizon:** strictly 1 session ahead, or also 5-day/1-week (where some literature shows more predictability)?
5. **Intraday data use:** incorporate the 2022–2026 1-minute HF data (session features) or daily-only for now?
6. **Corporate actions:** is there a reliable CA list in-repo, or must one be built/scraped?
7. **Point-in-time universe:** do we need historical index membership, or is a liquidity-screened universe acceptable?
8. **Compute:** training on the current machine or a separate GPU box?

---

## 17. Sources {#17-sources}

Access date for all URLs: **2026-10-15** unless noted. *Primary* = original paper/official page/repo; *Secondary* = summary/blog.

### Data facts (computed on this repo)
- `/home/samarth/thense/data/processed/bhavcopy/bhavcopy.parquet` — 10,672,269 rows, 8,897 symbols, 6,625 days, 2000-01-03→2026-10-07. *Primary — computed 2026-10-15.*
- `/home/samarth/thense/data/raw/indices/nifty50.parquet`, `india_vix.parquet`, `sp500.parquet`, `nasdaq.parquet`, `usdinr.parquet`, etc. *Primary — computed 2026-10-15.*

### Market dynamics & predictability
- Fama (1970), "Efficient Capital Markets" — http://gesd.free.fr/fama1970.pdf *(primary)*
- Fama (1991), "Efficient Capital Markets: II" — https://www.bu.edu/econ/files/2011/01/Fama2.pdf *(primary)*
- Lo & MacKinlay (1988), variance ratios — https://www.nber.org/system/files/working_papers/w2168/w2168.pdf *(primary)*
- Jegadeesh & Titman (1993) — https://bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf *(primary)*
- Lehmann (1990) — https://www.nber.org/papers/w2533 *(primary)*
- Cont (2001), stylized facts — https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf *(primary)*
- Engle (1982) — https://www.econometricsociety.org/publications/econometrica/1982/07/01/autoregressive-conditional-heteroscedasticity-estimates *(primary)*
- Bollerslev (1986) — https://doi.org/10.1016/0304-4076(86)90063-1 *(primary)*
- Corsi (2009), HAR-RV — https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf *(primary)*
- Hamilton (1989) — https://users.ssc.wisc.edu/~behansen/718/Hamilton1989.pdf *(primary)*
- Ahmad & Kamaiah, Markov-switching Nifty/Sensex — https://mpra.ub.uni-muenchen.de/37174/1/MPRA_paper_37174.pdf *(primary)*
- Vipul (2005), F&O expiry effect — https://ideas.repec.org/a/wly/jfutmk/v25y2005i11p1045-1065.html *(secondary)*

### Feature engineering & statistics
- Dickey & Fuller (1979) — https://doi.org/10.1080/01621459.1979.10482531 *(primary)*
- KPSS (1992) — https://doi.org/10.1016/0304-4076(92)90104-Y *(primary)*
- Phillips & Perron (1988) — https://doi.org/10.1093/biomet/75.2.335 *(primary)*
- statsmodels ADF — https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html *(primary)*
- López de Prado, AFML Ch.5 (fracdiff) — https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c05.xhtml *(primary)*
- `fracdiff` repo — https://github.com/fracdiff/fracdiff *(primary)*
- `mlfinlab` fracdiff/labeling — https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/features/fracdiff.py *(primary)*
- OHLC volatility estimators — https://bookdown.org/jarneric/financial_econometrics/5.1-ohlc-measures.html *(secondary)*
- MIT 18.642 Lecture 17 — https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/mit18_642_f24_lec17_2.pdf *(primary)*
- Amihud (2002) — https://doi.org/10.1016/S1386-4181(01)00024-6 *(primary)*
- Brock, Lakonishok & LeBaron (1992) — https://doi.org/10.1111/j.1540-6261.1992.tb04681.x *(primary)*
- Bajgrowicz & Scaillet (2012) — https://doi.org/10.1016/j.jfineco.2012.06.001 *(primary)*
- Moskowitz, Ooi & Pedersen (2012) — https://doi.org/10.1016/j.jfineco.2011.11.003 *(primary)*
- sklearn permutation importance — https://scikit-learn.org/stable/modules/permutation_importance.html *(primary)*
- Lundberg & Lee (2017), SHAP — https://arxiv.org/abs/1705.07874 *(primary)*
- sklearn common pitfalls — https://scikit-learn.org/stable/common_pitfalls.html *(primary)*
- mlfinlab labeling — https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py *(primary)*
- Hudson & Thames meta-labeling — https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/ *(primary)*

### Model selection
- Qlib benchmarks — https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md *(primary, verified)*
- Grinsztajn et al. (2022) — https://arxiv.org/abs/2207.08815 *(primary, verified)*
- Shwartz-Ziv & Armon (2021) — https://arxiv.org/abs/2106.03253 *(primary)*
- Gorishniy et al. (2021) — https://arxiv.org/abs/2106.11959 *(primary)*
- TabZilla — https://arxiv.org/abs/2305.02997 *(primary)*
- Gu, Kelly & Xiu (2020) — https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf *(primary)*
- Rahimikia et al. (2025), TSFMs in finance — https://arxiv.org/abs/2511.18578 *(primary, verified)*
- Chronos — https://arxiv.org/abs/2403.07815 · TimesFM — https://arxiv.org/abs/2310.10688 · Moirai — https://arxiv.org/abs/2402.02592 *(primary)*
- Jane Street 1st — https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s *(primary)*
- Ubiquant 1st — https://www.kaggle.com/competitions/ubiquant-market-prediction/writeups/k-i-y-1st-place-solution-our-betting-strategy *(primary)*
- Elor & Averbuch-Elor (2022), SMOTE — https://arxiv.org/abs/2201.08528 *(primary)*
- Bergstra & Bengio (2012) — https://jmlr.org/papers/v13/bergstra12a.html *(primary)*
- Optuna — https://arxiv.org/abs/1907.10902 *(primary)*

### Validation & evaluation
- Purged CV (Wikipedia) — https://en.wikipedia.org/wiki/Purged_cross-validation *(secondary)*
- Purged CV controlled study — https://purged-cv.marketmaker.cc/paper.pdf *(primary)*
- Walk-forward controlled study — https://wfo.marketmaker.cc/paper.pdf *(primary)*
- Bailey & López de Prado, Deflated Sharpe — https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf *(primary)*
- Bailey et al. (2014), Pseudo-Mathematics — https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf *(primary)*
- PBO — https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf *(primary)*
- López de Prado (2018), 10 reasons ML funds fail — https://doi.org/10.3905/jpm.2018.44.6.120 *(primary)*
- skfolio CPCV — https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html *(primary)*
- `timeseriescv` — https://github.com/sam31415/timeseriescv/ *(primary)*
- vectorbt — https://github.com/polakowo/vectorbt/ *(primary)*
- sklearn metrics — https://scikit-learn.org/stable/modules/model_evaluation.html *(primary)*
- Qlib report/IC — https://qlib.readthedocs.io/en/latest/component/report.html *(primary)*

### Costs & NSE structure
- Zerodha charges — https://zerodha.com/charges/ *(primary, verified)*
- NSE STT & levies — https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies *(primary)*
- NSE impact cost — https://www.nseindia.com/static/products-services/indices-impact-cost *(primary)*
- NSE price bands — https://www.nse.in/static/products-services/equity-market-price-bands *(primary)*
- NSE circuit breakers — https://www.nseindia.com/products-services/equity-market-circuit-breakers *(primary)*
- NSE corporate-action adjustments — https://www.nseindia.com/static/products-services/equity-derivatives-corporate-actions-adjustments *(primary, verified)*
- NSE legend of series — https://www.nseindia.com/market-data/legend-of-series *(primary)*
- NSE FII/DII — https://www.nseindia.com/reports/fii-dii *(primary)*
- NSE India VIX methodology — https://nsearchives.nseindia.com/web/sites/default/files/inline-files/India_VIX_comp_meth.pdf *(primary)*
- NSE expiry shift circular FAOP68747 — https://nsearchives.nseindia.com/content/circulars/FAOP68747.pdf *(primary)*
- SEBI algo-trading circular (Feb 2025) — https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html *(primary)*
- SEBI F&O loss study (Sep 2024) — https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years%5F86906.html *(primary, verified)*
- NSE historical data guide — https://www.nsehistoricaldata.co.in/guides/corporate-actions-and-unadjusted-prices *(secondary)*

### India-specific predictability
- Hiremath & Kumari (2014), adaptive market — https://link.springer.com/article/10.1186/2193-1801-3-428 *(primary)*
- Mohapatra & Misra (2020), India momentum — https://www.sciencedirect.com/science/article/pii/S0970389617301647 *(primary)*
- Narayan et al. (2015), India predictability — https://www.sciencedirect.com/science/article/abs/pii/S0378426615001156 *(primary)*
- NSE working papers — https://www.nseindia.com/static/research/working-papers *(primary)*
- TechRxiv Nifty XGBoost (~70%) — https://doi.org/10.36227/techrxiv.175281141.15130016/v1 *(primary — flagged likely leaky)*
- CAGTRADE (~98%) — https://ideas.repec.org/a/kap/apfinm/v32y2025i2d10.1007_s10690-024-09463-w.html *(secondary — flagged implausible)*
- Nifty gap forecasting (VIX+candles, ~53–54%) — https://doi.org/10.36948/ijfmr.2025.v07i04.53859 *(primary)*
- E-7 deep learning (74.3% Nifty) — https://ideas.repec.org/a/acc/malfin/v40y2025i123p92-111.html *(secondary — flagged leaky)*
- Fama/Goyal/Welch/McLean-Pontiff/Hou-Xue-Zhang cited above.

---

*End of dossier. Hand this file to plan mode next.*
