---
title: NSE Intraday ML Forecasting — Market Data, Competitor & Technical Research
date: 2026-10-07
idea: Build an ML model that predicts NSE (India) market moves at 5-minute, hourly, and daily horizons, starting from the cheapest reliable data acquisition possible and training on a separate machine.
verdict: CONDITIONAL-GO — data is cheap and abundant (free stack can reach ~1B rows); 5-min directional alpha after NSE costs is a low-probability, high-effort bet. Proceed only as a staged, volatility-first, cost-aware research program with an explicit kill criterion.
confidence: High on data availability & costs; Med on feasibility of a live edge; Low on any claim of high intraday directional accuracy.
---

# NSE Intraday ML Forecasting — Research Dossier

> **How to read this:** Every non-obvious claim is tagged `[Fact|Opinion|Inference|Estimate, High|Med|Low]` and carries a source link. Prices, repo stars and dataset stats were observed on **2026-10-07** unless stated. `[unverified]` = could not confirm against a primary source. This is research, not legal, investment, or tax advice.

## Table of contents
1. [TL;DR](#1-tldr)
2. [The idea, unpacked](#2-the-idea-unpacked)
3. [Market landscape](#3-market-landscape)
4. [Competitors & alternatives](#4-competitors--alternatives)
5. [Voice of the customer](#5-voice-of-the-customer)
6. [Open source & prior art](#6-open-source--prior-art)
7. [Technical feasibility & build plan](#7-technical-feasibility--build-plan)
8. [Business & pricing](#8-business--pricing)
9. [Differentiation & scope](#9-differentiation--scope)
10. [Risks & unknowns](#10-risks--unknowns)
11. [Opportunities & novel additions](#11-opportunities--novel-additions)
12. [Recommended next steps](#12-recommended-next-steps)
13. [Open questions for planning](#13-open-questions-for-planning)
14. [Sources](#14-sources)

---

## 1. TL;DR

1. **The user's primary ask — maximum reliable NSE data for free/minimal cost — is solved and cheap.** A free stack can deliver **~715M+ all-market 1-minute rows (2022–2026)**, **~259M options rows**, and **~20M daily rows back to the 1990s**, at **₹0**. `[Fact, High]`
2. **Best free bulk-history play:** the Hugging Face dataset `xxparthparekhxx/indian-stock-market-minute-data` — **2,500+ NSE stocks & indices, ~715M rows, 1-minute 2022–2026 + daily 2000–2026, MIT license, ~10.5 GB** — sourced from Upstox. Verified live. `[Fact, High]` [HF](https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data)
3. **Best free ongoing intraday source:** **Upstox API** — 1-minute bars from **Jan 2022**, daily from **Jan 2000**, **free**, ~50 req/s. Agent One SmartAPI, Fyers and ICICI Breeze are also free. `[Fact, High]`
4. **Best minimal-cost paid intraday source:** **DhanHQ Data API at ₹499 + GST/month** — 1/5/15/30/60-min bars **last 5 years**, daily from inception, and **20-level (even 200-level) depth**. `[Fact, High]`
5. **Zerodha Kite Connect is now just ₹500/month** (down from the old ₹2,000 + ₹2,000 add-on), includes live WebSocket + historical candles, minute bars from ~Feb 2015. `[Fact, High]` [Zerodha](https://zerodha.com/products/api/)
6. **Longest free *daily* history** comes from NSE bhavcopy archives and Kaggle "NSE India Stock Data 1990–2021", not from any broker API. Free *intraday* practically begins ~2015 (Kite/Kaggle) and reliably ~2022 (Upstox/HF). `[Fact, Med]`
7. **Do not scrape `nseindia.com` for a product** — NSE's Terms of Use expressly prohibit "systematic or automated data collection activities (including scraping, data mining, data extraction and data harvesting)". Broker APIs and Yahoo also forbid redistribution. Internal research is the low-risk zone; selling data/signals flips on licensing + SEBI rules. `[Fact, High]`
8. **The realistic ceiling on 5-min directional accuracy for a liquid NSE index is ~51–55%. Anything ≥60% sustained on index intraday should be presumed data leakage until proven with purged CV and a naive baseline.** Published leakage-free deep models sit at ~50%. `[Inference, High]`
9. **Volatility is the tractable target, not direction.** Next-period volatility clusters and is genuinely predictable (GARCH/HAR/India VIX); next-period *direction* has an R² often <1–2%. Model volatility first. `[Fact, High]`
10. **NSE transaction costs are brutal at 5-min scale.** Cash-equity round trip ≈ **10.6 bps**; index-futures ≈ **5.9 bps**. A typical 5-minute Nifty move has σ ≈ **11–12 bps**, so friction ≈ one full standard deviation of the thing you are predicting. `[Estimate, Med]`
11. **SEBI has already been here:** 93% of individual F&O traders lost money FY22–FY24 (₹1.81 lakh crore aggregate); FY26 still 87.7% lost, and **97% of FPI/prop profits came from algo entities** — retail is the liquidity, not the alpha. `[Fact, High]` [SEBI PR 37/2024](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html)
12. **The retail India algo market is crowded at the plumbing layer (Tradetron, AlgoTest, AlgoBulls, Streak, Sensibull, Quantsapp) but nearly empty at the *audited-forecasting* layer.** Nobody in India publishes an independently verified, timestamped NSE prediction track record. That is the credibility gap — and the differentiator. `[Inference, High]`
13. **Regulatory squeeze is a moat, not just a cost:** SEBI's Feb-2025 algo framework (full effect 1 Apr 2026) mandates static IPs, order tagging, broker-principal/algorithm-provider-agent structure, and **RA registration for black-box algos sold to others**. `[Fact, High]` [SEBI](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html)
14. **Build on the shoulders of `microsoft/qlib` (49.2k★, MIT) + `LightGBM` for tabular + `purgedcv` for leakage-safe validation + `Nautilus Trader` for execution realism** — not on the many 0–14★ "Nifty LSTM" toy repos. `[Opinion, High]`
15. **Suggested verdict: a staged conditional-go.** Spend 3–6 person-weeks on a free-data Nifty daily/hourly *volatility* model with strict purged-CV and cost-aware backtest. Kill it if you can't beat a HAR/EWMA volatility baseline or if the directional overlay nets <10 bps/trade. Only then spend 8–16 weeks on a 5-min system.

---

## 2. The idea, unpacked

### 2.1 What the product is
An ML system that consumes NSE price/volume (and optionally options, cross-asset and flow) data and outputs forecasts of NSE instruments over **multiple horizons: 5 minutes, 1 hour, and 1 day** — "for as long time as possible" (i.e., maximize history depth for training). Trained on a separate machine. The immediate goal is a validated research pipeline and model, not a live trading product.

### 2.2 Job-to-be-done
- **Primary JTBD:** "Tell me, from data, whether the next 5 min / hour / day of a Nifty/instrument move is up, down, or (better) how large and how uncertain it will be — reliably enough to act on after costs."
- **Secondary JTBD:** "Give me a clean, survivorship-bias-free, corporate-action-adjusted NSE dataset and a backtesting harness I can trust, because I can't assemble one today."

### 2.3 Ideal customer profile (ICP)
Three plausible ICPs, ranked by fit with the evidence:
1. **The builder themselves (self-directed quant/ML engineer)** — owns the model, trades own account. Lowest legal friction. `[Inference, High]`
2. **Serious retail/algo trader (India)** — pays ₹500–₹5,000/month for tools; complains loudly about data quality and backtest/live divergence; skeptical of signal-sellers. `[Fact, Med]`
3. **(Later, high-friction) signal/API subscribers** — would require SEBI Research Analyst registration and NSE data licensing. Only pursue after an audited track record exists. `[Inference, High]`

### 2.4 Key assumptions (recorded, not yet validated)
- **A1:** The user targets Nifty 50 / Bank Nifty / liquid F&O, not illiquid midcaps. `[Assumption]`
- **A2:** "Predict" = produce a probabilistic directional and/or volatility signal per horizon, not an exact price. `[Assumption]`
- **A3:** Budget is genuinely near-zero (free/minimal), and there is tolerance for a research phase with no immediate live trading. `[Assumption]`
- **A4:** Geography is India; instrument is NSE; the model will be trained on a separate (likely GPU or beefy CPU) machine. `[Assumption]`
- **A5:** The user is open to the finding that 5-min *direction* is a poor bet and that volatility/meta-labeling is the better first target. `[Assumption]`

### 2.5 What would falsify the idea
- You cannot beat an **EWMA/HAR volatility baseline** on out-of-sample realized volatility with any model. → The ML premise fails.
- Your best **cost-aware, purged-CV, out-of-sample directional** strategy yields **<10 bps/trade** net on Nifty futures. → Not worth building.
- A **paper-trading** period (1–3 months) shows live-vs-backtest trade matches below ~90% or slippage materially worse than modeled. → Execution invalidates it.
- Free data shows **material gaps, unadjusted corporate actions, or timestamp leakage** that you cannot repair. → Data premise fails.

---

## 3. Market landscape

### 3.1 Context and "why now"
- **Retail participation exploded.** India has ~12+ crore retail market participants while SEBI-registered Research Analysts + Investment Advisers together number **under ~2,600** — a massive advice/tooling gap. `[Fact, Med]` [Value Research](https://www.valueresearchonline.com/stories/566248/sebi-registered-research-analysts-investment-advisers/)
- **The money is flowing to algos.** SEBI found that in FY24, **97% of FPI profits and 96% of proprietary-trader profits came from algorithmic entities**; retail losses concentrate in options (92% of FY26 losses; 97% of traders are option buyers). `[Fact, High]` [SEBI Aug-2026 study](https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1787236407005.pdf)
- **"Why now" tailwinds for an ML NSE forecaster:**
  - Free broker APIs (Upstox, Dhan, Angel One, Fyers) made high-resolution intraday history cheap for the first time; **free bulk datasets now exceed 700M rows**.
  - Open frameworks (`qlib`, `LightGBM`, `river`, `darts`, `sktime`, `Nautilus`) are mature and permissively licensed.
  - LLMs/foundation time-series models lower feature-engineering and code cost.
- **"Why now" headwinds:**
  - SEBI's **retail algo framework** (circular 4 Feb 2025; full effect 1 Apr 2026) imposes registration, static IPs, order tagging and black-box RA obligations. `[Fact, High]`
  - **Budget 2026 raised F&O STT** (futures to 0.05% sell; options to 0.15% premium), worsening the cost hurdle. `[Fact, Med]`
  - The AI-stock-prediction category is polluted by unaudited, marketing-driven claims, breeding retail distrust. `[Fact, Med]`

### 3.2 Market size (method and sources)
- **Total Indian retail brokerage is large but the "prediction tooling" slice is speculative.** smallcase alone raised **$126M** and had **₹113.7 crore FY25 revenue**; Zerodha has **~8 million clients**. `[Fact, Med]`
- **TAM for India retail algo/analytics tooling** `[Estimate, Low]`: ~1–2 crore active retail traders × realistic ₹500–₹3,000/month tool willingness = **₹6,000–₹36,000 crore/year theoretical**, but actual willingness-to-pay concentrates on **₹250–₹2,500/month** (see §5, §8). Treat any big number here as an upper bound, not a forecast.
- **SAM for "audited ML NSE forecasting"** `[Estimate, Low]`: the subset who will pay for *forecasts* rather than data/alerts is currently near-zero because no credible product exists; the adjacent "analytics + alerts" market is already served. `[Inference, Med]`
- **Method note:** no credible third-party sizing of the "NSE prediction" niche exists. The honest read is that the *data/tooling* market is real and growing, while the *forecast-as-a-product* market is unproven. `[Inference, High]`

### 3.3 Segments
| Segment | What they want | Served today? |
|---|---|---|
| DIY quant / ML engineer (the user) | Clean data + backtest harness | Partially (free data exists; harness is DIY) |
| Serious retail algo trader | No-code + reliable backtest/live | Yes (Tradetron, AlgoTest, AlgoBulls) |
| Options trader | Greeks/analytics/strategy | Yes (Sensibull, Quantsapp) |
| Signal subscriber | "Tell me what to buy" | Yes but distrusted/unaudited |
| Institutional/quant fund | Licensed tick data + research | Yes (Bloomberg, TrueData, GlobalDatafeeds, Databento) |

### 3.4 Tailwinds vs headwinds (summary)
| Tailwinds | Headwinds |
|---|---|
| Free 700M+ row datasets, free broker APIs `[Fact, High]` | SEBI RA/IA + algo registration for any distribution `[Fact, High]` |
| Open ML/backtest frameworks `[Fact, High]` | NSE data licensing forbids redistribution `[Fact, High]` |
| Growing retail demand + distrust of incumbents `[Fact, Med]` | Costs (STT up in 2026) eat marginal edge `[Fact, Med]` |
| LLM/foundation models reduce build cost `[Inference, Med]` | Evidence that retail intraday ML mostly loses `[Fact, High]` |

---

## 4. Competitors & alternatives

**Pricing observed 2026-10-07 unless noted.** Many rows were fetched live; JS/login-gated pages are marked `[unverified]`.

### 4.1 Direct, indirect, status-quo and big-tech adjacency

| Name | What it is | Target | Pricing (INR/USD) | ML/prediction? | Traction | Strength | Weakness | Link |
|---|---|---|---|---|---|---|---|---|
| **Tradetron** | No-code algo builder + marketplace + MCP AI connector | India/each retail | Free; ₹250 → ₹7,500/mo tiers | AI *builds* strategies, does **not** forecast | Claims 405k+ traders, ~60k algos | Multi-broker, marketplace, AI builder | Black-box marketplace; backtest cost/support complaints | [tradetron.tech](https://tradetron.tech/) |
| **AlgoTest** | Options backtest/forward-test/live | Indian F&O retail | ₹499 → ₹14,999/mo; 25 free backtests/wk | "AI builder", no forecast | 39M+ trades, 30k+ traders `[Fact, Med]` | Cheap, transparent credits | F&O-only, credit accounting | [algotest.in](https://algotest.in/pricing) |
| **AlgoBulls** | Enterprise+retail algo infra | Retail, brokers, RA/RIA | ₹99/7-day; retail ~₹1,180+/mo `[unverified, Low]` | Python predictive-algo support, no forecast | Claims 1M+ orders `[Fact, Low]` | SEBI-aligned, ISO 27001 | Opaque pricing | [algobulls.com](https://algobulls.com/pricing) |
| **Streak (Zerodha)** | No-code strategy/scanner | Zerodha users | Free/unlimited for Zerodha users (2026) | None | 2M+ users `[Fact, Low]` | Free, tight Kite integration | Alerts only, no auto-trade; broker lock-in | [streak.tech](https://www.streak.tech/) |
| **Quantsapp** | Options analytics suite (107 tools) | Indian options traders | Free; PRO ₹1,667/mo (yearly) | "Quant models", not return forecast | 17 lakh+ traders `[Fact, Med]` | Deep F&O analytics, execution | Expensive; "signals lag the move" reviews | [quantsapp.com](https://www.quantsapp.com/pricing) |
| **Sensibull** | Options strategy/analytics | Indian options retail | Free; Pro list ₹1,300/mo, promo ₹640 | None | "India's largest options platform" `[Fact, Med]` | Zerodha-backed UX | No prediction/execution | [sensibull.com](https://sensibull.com/) |
| **StockEdge** | Analytics/screener/scans | Indian investors | Premium ₹399/mo; Pro ₹1,499/mo | Pattern detection, not forecast | SEBI RA INH300007493 `[Fact, Med]` | 500+ scans | EOD, no execution | [stockedge.com](https://web.stockedge.com/) |
| **Trendlyne** | Screener + analytics + MCP AI connector | India+US | Free; GURUQ ₹310/mo; PRO ₹1,500/mo | Scores/backtest, no forecast | Widely used | Breadth, AI data connectors | EOD, no execution ML | [trendlyne.com](https://trendlyne.com/subscription/plans/) |
| **QuantConnect** | Cloud quant platform (LEAN), ML research | Global quants | Free + paid tiers (dynamic) `[unverified]` | Yes, full ML | LEAN 21.9k★ `[Fact, High]` | All asset classes, own IP | Steep, C#-native, price opacity | [quantconnect.com](https://www.quantconnect.com/pricing) |
| **Numerai** | ML prediction tournament | Global data scientists | Free; stake NMR | Core = ML ensemble prediction | $1.045B fund AUM `[Fact, Med]` | Auditable outcomes, real fund | **No NSE/India**; token risk | [numer.ai](https://numer.ai/) |
| **Kavout** | ML stock ranking (Kai Score) | Global investors | Free; Pro $16/mo; Premium $39/mo | Yes (ranking) | Added NSE/BSE `[Fact, Med]` | Multi-market ML ranking | Research-only, no execution | [kavout.com](https://www.kavout.com/pricing-plans) |
| **Tickeron** | AI trading robots/signals | US retail | ~$5–$250/mo | Yes (self-reported 184–268%/yr) | Trustpilot 3.4/5 `[Fact, Med]` | Many bots | Unaudited, US-centric | [tickeron.com](https://tickeron.com/pricing/) |
| **Danelfin** | AI Stock Score 1–10 | US/EU retail | Free; Plus $29/mo; Pro $79/mo; Elite $179/mo | Yes (+21% claimed alpha) | `[Fact, Med]` | Explainable, API | **No NSE coverage** | [danelfin.com](https://danelfin.com/pricing) |
| **Univest** | AI screener + SEBI RA calls | Indian retail | From ₹6/day; premium ~₹25,999/yr | "AI" + human analyst; claims 86% accuracy | RA INH000013776 `[Fact, Med]` | Regulated accountability | Self-reported accuracy | [univest.in](https://univest.in/) |
| **DalalAI** | "AI stock intelligence" India | Indian investors | Free+paid `[unverified]` | Yes, claims verified predictions | **Not SEBI-registered; scoreboard 0%/0** `[Fact, High]` | Transparent (empty) scoreboard | No proof, unregulated | [dalalai.com](https://dalalai.com/) |
| **I Know First** | AI forecasts incl. NSE indices | Global/India | Paid `[unverified]` | Yes, self-reported returns | Long-running `[Fact, Med]` | India packages | Cherry-picked returns | [iknowfirst.com](https://iknowfirst.com/) |
| **TradingView** | Charting + Pine backtest | Global mass retail | Free; Essential ₹995/mo → Ultimate ₹17,333/mo | No predictive model | 100M traders `[Fact, Med]` | Ubiquity, webhooks | Not an NSE algo broker | [tradingview.com](https://www.tradingview.com/pricing/) |
| **Zerodha / Dhan / Upstox / Angel One / Fyers** | Brokers with free/low-cost APIs | Indian retail/devs | ₹20/order; Kite ₹500/mo; Dhan data ₹499/mo | None | Zerodha ~8M clients `[Fact, Med]` | Free APIs, trust | Execution only; you supply the alpha | [zerodha.com](https://zerodha.com/) |
| **Status quo: spreadsheets / TradingView eyeballing / Telegram signals / doing nothing** | Manual analysis, copied calls | Most retail | ₹0–₹2L courses | No | — | Free/familiar | Not systematic, not auditable | — |

### 4.2 The 5 that matter most

**1. Tradetron — the closest thing to a distribution channel, and not a competitor to a forecaster.**
Tradetron is a no-code multi-broker algo builder plus a strategy marketplace; it already ships an AI layer that turns English into strategy code via MCP connectors, but **it does not predict returns**. `[Fact, High]` [Tradetron](https://tradetron.tech/). A genuine forecasting model would be *complementary*: Tradetron supplies execution and distribution; the model supplies signal. Weakness: black-box marketplace strategies now face SEBI RA obligations, and users report backtest/support problems. `[Opinion, Med]`

**2. Quantsapp — the deepest India-specific analytics incumbent, but not a predictor.**
Quantsapp has 107 tools and 17+ lakh traders, transparent pricing (PRO ₹1,667/mo billed yearly, PRO+ ₹3,333/mo), and options-flow/greeks analytics — but **no price-direction forecast**. `[Fact, High]` [Quantsapp pricing](https://www.quantsapp.com/pricing). The recurring review complaint that "indicators update only after the move has already happened" is precisely the gap a real predictor addresses. `[Fact, Med]`

**3. Danelfin / Tickeron — proof the West already sells "AI prediction," and proof India is open.**
Danelfin sells a transparent 1–10 AI score with published pricing and claims top-decile alpha, but **covers only North America/Europe — no NSE**. `[Fact, Med]` [Danelfin](https://danelfin.com/pricing). Kavout has added NSE/BSE ranking but is research-only. The exact product described here — ML *NSE intraday* prediction — is not served by the leading Western vendors. `[Inference, High]`

**4. Univest / DalalAI / I Know First — who claims "AI NSE prediction" today, and the audit problem.**
Univest (SEBI RA INH000013776) advertises "86% accuracy on closed recommendations" — self-reported. `[Fact, Med]`. DalalAI markets verified AI predictions but its own scoreboard showed **0% across 0 predictions** and it is **not SEBI-registered**. `[Fact, High]` [DalalAI](https://dalalai.com/). **Nobody in India publishes an independently audited, timestamped NSE forecast record.** That is the single biggest credibility gap and therefore the biggest positioning opportunity. `[Inference, High]`

**5. The regulatory wall reshapes the category and favours the compliant.**
SEBI's Feb-2025 algo circular makes brokers the principal and algo providers agents, mandates exchange-issued algo IDs, static IPs, OAuth/2FA and empanelment, and forces **black-box algo sellers to register as Research Analysts**. `[Fact, High]` [SEBI](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html). This is both a barrier (compliance cost) and a moat (unregistered signal-sellers are being squeezed out). `[Inference, High]`

### 4.3 Big-tech / platform adjacency
- **Brokers are commoditising data access** (free APIs), removing the old ₹4,000/month moat. `[Fact, High]`
- **TradingView** owns retail charting but explicitly is not an NSE execution venue. `[Fact, Med]`
- **NSE itself sells the authoritative feed** (L1/L2/L3, tick, 1-min/5-min snapshots, order & trade data) — the licensing gate for any productised data. `[Fact, High]` [NSE market data](https://www.nseindia.com/static/market-data/real-time-data-subscription)

---

## 5. Voice of the customer

> **Method caveat:** Reddit's `.json` endpoints returned 403 during this research; quotes below were recovered verbatim from search-indexed copies of the live threads. X/Twitter and YouTube comment threads could not be reliably retrieved. Treat all quotes as illustrative, not statistically representative; the URLs were not all re-opened individually. `[Inference, Med]`

### 5.1 r/IndiaAlgoTrading (the densest India data/algo community)
> "I have a couple of Intraday Strategies and I'm looking for vendors who can provide Indian Stock Market Historical Market Data..." — u/Shrinii, [thread](https://www.reddit.com/r/IndiaAlgoTrading/comments/1jg8wfy/looking_for_vendors_who_can_give_me_historical/)

> "For free data better go with brokers, except Zerodha all brokers provide data free of cost." — comment, [thread](https://www.reddit.com/r/IndiaAlgoTrading/comments/1jsdi9l/algotrading_data/)

> "so you want to earn some bucks from this model? but looking for free data. this is impossible to find." — comment, same thread

> "As an ML engineer, my instinct was to throw complex deep learning models (like LSTMs) at price data. It failed... deep models just memorize historical noise (overfit) and blow up live." — u/WinterSpecial7970, [thread](https://www.reddit.com/r/IndiaAlgoTrading/comments/1unug36/)

> "Zero trades matched between live and backtest. Not 'few mismatches'. Literally 0." — u/Neel_Sam on Upstox V3, [thread](https://www.reddit.com/r/IndiaAlgoTrading/comments/1qe99mp/)

> "NSE actively blocks IP ranges belonging to major cloud providers... classic cat-and-mouse game." — comment, [thread](https://www.reddit.com/r/IndiaAlgoTrading/comments/1uzbfch/serious_help/)

### 5.2 r/IndianStreetBets, r/NSEbets (retail side)
> "You can't really predict the market using regression in any profitable way." — u/RobotsMakingDubstep, [thread](https://www.reddit.com/r/IndianStreetBets/comments/17ix9k5/)

> "There are 100 indicators out there. Still 90% lose money. Why?" — u/mr-ajar, [thread](https://www.reddit.com/r/IndianStreetBets/comments/1rwdmdc/)

> "No adjusted data for splits/dividends. This is the biggest problem. My algorithms fails because the data is unprocessed." — u/SpeedyCafe, [thread](https://www.reddit.com/r/IndianStreetBets/comments/ope6m8/)

> "my broker's websocket consistently drops right around 9:15-9:16... my first 5 min candle is incomplete." — u/Nightcrawler_2000, [thread](https://www.reddit.com/r/NSEbets/comments/1shm7nl/)

### 5.3 r/algotrading, r/quant
> "yahoo finance is very unreliable in terms of providing data for Indian stocks and often fails to do its job." — u/CommunityDifferent34, [thread](https://www.reddit.com/r/algotrading/comments/1noil62/)

> "There is almost no information in daily returns... The model can barely do better from random guessing." — comment, [thread](https://www.reddit.com/r/algotrading/comments/1r2gxn3/)

> "Shit in - shit out." — r/quant on data quality, [thread](https://www.reddit.com/r/quant/comments/1fjo8on/)

> "I want data for each trade executed in NSE... with nano or macro second timestamp precision... I cannot afford to set up a co-location server." — u/Maverick4096, [r/quantindia](https://www.reddit.com/r/quantindia/comments/1svmf34/api_for_data/)

### 5.4 ValuePickr & Zerodha TradingQnA
> "The unavailability of data is perhaps the biggest problem for us in India." — basumallick, [ValuePickr](https://forum.valuepickr.com/t/quant-investing/35451)

> "One of the biggest challenges in quants systems these days is people are using too much on ML... unless there is an underlying logic to that pattern, it usually fails as it ends up being a random pattern." — ValuePickr, same thread

> "TradingView backtests on 5 min charts give you roughly 17 days of data — that is not enough to validate anything meaningful." — [TradingQnA](https://tradingqna.com/t/full-algo-automation-in-python-for-indian-markets/192235)

> "Historical data is now free (with base Kite Connect subscription)." — Matti (Zerodha), [Kite forum](https://kite.trade/forum/discussion/14806/)

### 5.5 Hacker News
> "I wouldn't trust any backtracking test with these models. Try doing a real-time test over 8 months and see what happens." — [HN comment](https://news.ycombinator.com/item?id=46154491)

> "Noise dominates everything you will find in statistical arbitrage. R^2 of 1% are something to write home about." — [HN comment](https://news.ycombinator.com/item?id=14943605)

> "the target series should be log-normal returns, not raw asset price... You could trivially get better forecasts by naively repeating the last-observed price." — [HN comment](https://news.ycombinator.com/item?id=18415336)

### 5.6 Synthesis

**Top pains (ranked):**
1. **Data quality/adjustment** — unadjusted splits/bonuses, survivorship bias, restated fundamentals, feed O/H/L/C mismatches. `[Fact, High]`
2. **Cost of clean data** — broker APIs cheap but incomplete; authorised vendors quote-based; NSE direct expensive. `[Fact, High]`
3. **API/infrastructure friction** — NSE blocks cloud IPs, SEBI static-IP mandate, daily token re-login, 9:15 websocket drops, rate limits not as promised. `[Fact, High]`
4. **Overfitting / backtest-live divergence** — the single most-discussed ML theme. `[Opinion, High]`
5. **Transaction costs kill edge** — STT/slippage repeatedly cited. `[Opinion, High]`

**Desired outcomes:** a single clean, survivorship-bias-free, corporate-action-adjusted NSE/BSE dataset (1-min/1-sec, equities + F&O); backtest↔live parity; cheap access that works from cloud; a fully automated system that "works for me." `[Inference, High]`

**Objections:** "free quality data doesn't exist"; ML price prediction seen as mostly overfitting/scam; distrust of platform backtests and signal-sellers after SEBI enforcement actions. `[Opinion, High]`

**Current workarounds:** free broker APIs + self-built databases; bhavcopy + corporate-action files self-adjusted; local tick storage; residential proxies; TradingView + webhooks (called fragile); paid courses/Telegram signals. `[Fact, High]`

**Willingness to pay:** strong for **clean historical data** and tools (₹500–₹3,000/mo accepted; "$25+/mo" for global data); heavy spend on courses/advisory (₹23k–₹2L) but trust collapsing. **No credible evidence of premium willingness to pay for raw signals.** `[Inference, Med]`

**Exact vocabulary:** "the data problem in India," "adjusted data," "survivorship," "look-ahead," "overfit," "memorize noise," "blow up live," "backtest vs live," "slippage," "STT," "static IP," "Akamai wall," "cat-and-mouse game," "websocket drops at 9:15," "tick-by-tick / TBT," "point-in-time vs restated," "black-box," "finfluencer," "scam." `[Fact, High]`

---

## 6. Open source & prior art

Star counts / last activity pulled live via the GitHub API on **2026-10-07**.

### 6.1 Data-acquisition libraries
| Repo | What it does | Stars | Last activity | License | Reuse | Link |
|---|---|---|---|---|---|---|
| `ranaroussi/yfinance` | Yahoo downloader (`^NSEI`, `RELIANCE.NS`); 1m–3mo intraday, daily to ~2000 | **25,454** | 2026-10-05 | Apache-2.0 | High (free intraday, no NSE scraping) | [GitHub](https://github.com/ranaroussi/yfinance) |
| `jugaad-py/jugaad-data` | NSE/BSE live + EOD/bhavcopy/index | **584** | 2026-10-06 | YOLO (public domain) | High (safest license) | [GitHub](https://github.com/jugaad-py/jugaad-data) |
| `swapniljariwala/nsepy` | Old NSE scraper | **807** | 2023-12-24 | MIT-style | **Low — dead** | [GitHub](https://github.com/swapniljariwala/nsepy) |
| `aeron7/nsepython` | Unofficial NSE wrapper + option chain | 368 | 2026-10-07 | GPL-3.0 | Med (GPL taint) | [GitHub](https://github.com/aeron7/nsepython) |
| `BennyThadikaran/NseIndiaApi` | Fast NSE API, option chain, EOD | 160 | 2026-10-05 | GPL-3.0 | Med (GPL) | [GitHub](https://github.com/BennyThadikaran/NseIndiaApi) |
| `marketcalls/openalgo` | Broker-agnostic algo bridge, 36 plugins, "Historify" local store | **2,794** | 2026-10-07 | AGPL-3.0 | Med (execution; AGPL copyleft) | [GitHub](https://github.com/marketcalls/openalgo) |
| `marketcalls/openchart` | NSE/NFO intraday + EOD from NSE charting | — | 2026 | — | Med | [GitHub](https://github.com/marketcalls/openchart) |

### 6.2 ML-for-trading and backtesting frameworks
| Repo | What it does | Stars | Last activity | License | Reuse | Link |
|---|---|---|---|---|---|---|
| `microsoft/qlib` | AI quant platform: data + model zoo + backtest (Alpha158/360) | **49,200** | 2026-10-05 | MIT | **High — best skeleton** | [GitHub](https://github.com/microsoft/qlib) |
| `AI4Finance-Foundation/FinRL` | Financial RL agents | **16,575** | 2026-10-06 | MIT | Med (mostly daily) | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| `polakowo/vectorbt` | Vectorized research/backtest | **9,290** | 2026-09-26 | Apache-2.0 + notice | High | [GitHub](https://github.com/polakowo/vectorbt) |
| `huseinzol05/Stock-Prediction-Models` | 100+ forecasting/trading notebooks | 9,500 | 2023-04-16 | Apache-2.0 | Med (dated) | [GitHub](https://github.com/huseinzol05/Stock-Prediction-Models) |
| `stefan-jansen/zipline-reloaded` | Event-driven backtester | 1,959 | 2026-01-06 | Apache-2.0 | Med | [GitHub](https://github.com/stefan-jansen/zipline-reloaded) |
| `nautechsystems/nautilus_trader` | Production Rust engine, L2 fill models | **29,681** | 2026-10-07 | LGPL-3.0 | Med-High (execution realism) | [GitHub](https://github.com/nautechsystems/nautilus_trader) |
| `QuantConnect/Lean` | Full algo engine | 21,896 | 2026-10-06 | Apache-2.0 | Med | [GitHub](https://github.com/QuantConnect/Lean) |
| `AI4Finance-Foundation/FinRL-Meta` | Datasets + envs | 1,946 | 2026-10-06 | MIT | Med | [GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta) |
| `hudson-and-thames/mlfinlab` | Lopez de Prado financial ML | 4,935 | 2023-10-02 | Commercial-restricted | Low-Med | [GitHub](https://github.com/hudson-and-thames/mlfinlab) |

### 6.3 Time-series, validation, indicators, evaluation
| Repo | What it does | Stars | Last activity | License | Reuse | Link |
|---|---|---|---|---|---|---|
| `sktime/sktime` | Unified TS-ML + expanding-window splitters | 10,059 | 2026-10-04 | BSD-3 | High (leakage-safe CV) | [GitHub](https://github.com/sktime/sktime) |
| `unit8co/darts` | Forecasting baselines + backtest | 9,536 | 2026-10-07 | Apache-2.0 | High | [GitHub](https://github.com/unit8co/darts) |
| `sktime/pytorch-forecasting` | TFT, N-BEATS | 4,999 | 2026-10-04 | MIT | Med-High | [GitHub](https://github.com/sktime/pytorch-forecasting) |
| `online-ml/river` | Incremental/online learning | 6,118 | 2026-10-07 | BSD-3 | High (streaming drift) | [GitHub](https://github.com/online-ml/river) |
| `TA-Lib/ta-lib-python` | C TA-Lib wrapper | 12,275 | 2026-09-21 | BSD-2 | High | [GitHub](https://github.com/TA-Lib/ta-lib-python) |
| `xgboosted/pandas-ta-classic` | 250+ indicators (maintained fork) | 449 | 2026-10-02 | MIT | High | [GitHub](https://github.com/xgboosted/pandas-ta-classic) |
| `bukosabino/ta` | Pure-Python indicators | 5,234 | 2026-03-18 | MIT | High | [GitHub](https://github.com/bukosabino/ta) |
| `bashtage/arch` | ARCH/GARCH volatility | 1,588 | 2026-10-07 | NCSA/BSD | High | [GitHub](https://github.com/bashtage/arch) |
| `statsmodels/statsmodels` | Econometrics/ARIMA/GARCH | 11,677 | 2026-10-06 | BSD-3 | High | [GitHub](https://github.com/statsmodels/statsmodels) |
| `ranaroussi/quantstats` | Tearsheets/Sharpe/drawdown | 7,686 | 2026-09-27 | Apache-2.0 | High | [GitHub](https://github.com/ranaroussi/quantstats) |
| `eslazarev/purged-cross-validation` | Purged CV, embargo, CPCV, DSR/PBO | — | — | — | **High (essential)** | [GitHub](https://github.com/eslazarev/purged-cross-validation) |

### 6.4 Free datasets (the heart of the data-sourcing answer)
| Dataset | Coverage | Rows | Cost | License | Link |
|---|---|---|---|---|---|
| **`xxparthparekhxx/indian-stock-market-minute-data`** | 2,500+ NSE stocks/indices; **1-min 2022–2026 + daily 2000–2026**; UTC timestamps; from Upstox | **~715M** (10.5 GB compressed / ~125 GB uncompressed) | Free | **MIT** | [HF](https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data) |
| **`rissin/nse-options-intraday`** | NIFTY/BANKNIFTY/SENSEX options; daily EOD 2001→present (NSE F&O bhavcopy) + Upstox 1-min 2024→present | **258.9M** (2.51 GB) | Free | "other" | [HF](https://huggingface.co/datasets/rissin/nse-options-intraday) |
| `debashis74017/nifty-50-minute-data` | Nifty 50 1/3/5/15/60-min + daily, 2015–2026, weekly-updated | ~4.19 GB | Free | Kaggle terms | [Kaggle](https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data) |
| `debaonline4u/NSE-Data` | Nifty50 + Next50, 1/3/5/10/15/30/60-min + daily | ~6.5 GB | Free | GitHub | [GitHub](https://github.com/debaonline4u/NSE-Data) |
| `stoicstatic` NSE 1990–2021 | 1,700+ stocks + 50+ indices, daily | 3,189 CSV / 173 MB | Free | Kaggle terms | [Kaggle](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020) |
| `paramamithra` NSE stocks | All listed stocks, daily OHLCV, updated daily | — | Free | Kaggle terms | [Kaggle](https://www.kaggle.com/datasets/paramamithra/historical-data-of-stocks-listed-on-nse) |
| `tejhq/indian-markets` | Indian markets | 25.4M | Free | HF | [HF](https://huggingface.co/datasets/tejhq/indian-markets) |
| `AMP4010/Historical_Nifty_50_Constituent_Weights_20Y` | 20y Nifty constituent weights (anti-survivorship) | — | Free | HF | [HF](https://huggingface.co/datasets/AMP4010/Historical_Nifty_50_Constituent_Weights_20Y) |
| `tickertruthorg/nse-india-security-master` | Instrument/symbol metadata | — | Free | HF | [HF](https://huggingface.co/datasets/tickertruthorg/nse-india-security-master) |

### 6.5 What we must build ourselves
- **A resilient NSE data ingestion + corporate-action adjustment + point-in-time store.** No maintained, licensed, permissively-licensed NSE intraday library exists; the scrapers are GPL and break silently. `[Inference, High]`
- **Triple-barrier / meta-labeling** — no permissively-licensed maintained implementation (`mlfinlab` abandoned 2023 + commercially restricted). Reimplement. `[Fact, High]`
- **An India-specific transaction-cost model** — no backtester ships NSE STT/stamp/exchange/GST/SEBI/spread. Build it. `[Inference, High]`
- **Execution/OMS layer** with SEBI algo tagging, static-IP handling, rate limits. SDKs exist (Kite, Upstox-Python, DhanHQ-py); the risk layer is ours. `[Opinion, Med]`
- **Point-in-time options chain history** — the expensive, hard part; free sources are snapshots. `[Estimate, Med]`

---

## 7. Technical feasibility & build plan

### 7.1 Target formulation — pick the right target first
| Target | Formulation | Verdict |
|---|---|---|
| Next-bar return regression | `r_{t+h}=log(P_{t+h}/P_t)` | Useful, but MSE is dominated by volatility |
| Direction classification | `1[r_{t+h}>0]` | Near 50/50 at 5-min; accuracy is a weak metric |
| Triple-barrier / meta-labeling | label by first barrier hit; secondary model for whether to act | **High ROI** — decouples direction from sizing `[Fact, High]` |
| Quantile forecast | predict 5/50/95th percentile | Good for risk/sizing |
| **Volatility regression** | predict realized variance/vol | **Start here — genuinely predictable** `[Fact, High]` |

**Why volatility first:** volatility clusters (GARCH/EWMA/HAR persistence), so future realized vol is forecastable with high R²; next-bar *direction* has an R² typically <1–2% and below transaction costs. `[Inference, High]`. India VIX is computed from Nifty option bid-ask using CBOE methodology; the implied-minus-realized premium is itself a model-friendly signal. `[Fact, High]` [NSE India VIX white paper](https://nsearchives.nseindia.com/web/sites/default/files/inline-files/white_paper_IndiaVIX.pdf)

### 7.2 Feature families
| Family | Features | Data cost |
|---|---|---|
| Return lags | log-returns lags 1..k, rolling sums | Free |
| Realized vol/variance | RV 5/15/60 bars; Parkinson, Garman-Klass; HAR-RV | Free — **strongest baseline** |
| Trend/momentum | EMA(9/21/50), MACD, RSI(14), ADX, Bollinger z | Free |
| VWAP deviation | close vs session VWAP | Free |
| Volume/turnover | volume z-score, OBV, turnover (₹) | Free |
| Volatility surface | India VIX level/change, ATM IV, skew, IV−RV | VIX free; chain paid |
| Overnight/global | GIFT Nifty, S&P 500, Nasdaq, DXY | Mostly free/delayed |
| FX/commodities | USDINR, Brent | Free/delayed |
| Flows | FII/DII net cash + OI | Free but **EOD only** |
| Calendar | time-of-day, day-of-week, **expiry = Tuesday**, days-to-expiry | Free |
| Order-flow imbalance | signed volume, OFI, queue imbalance | **Paid L2** — best alpha, worst cost |
| News sentiment | timestamped headlines, FinBERT | Paid for clean timestamps |

> **NSE expiry fact to encode:** NSE moved Nifty weekly expiry from Thursday to **Tuesday effective 28 Aug 2025**; only one weekly benchmark index option per exchange survives (NSE=Nifty, BSE=Sensex). Bank Nifty/FinNifty weekly options were discontinued Nov 2024. `[Fact, High]` [CNBC-TV18](https://www.cnbctv18.com/market/nse-to-shift-expiry-of-derivatives-to-tuesdays-starting-august-28-19625489.htm)

### 7.3 Model zoo
| Class | Models | Role |
|---|---|---|
| Linear | Logistic, Ridge, ElasticNet | Sanity floor; hard to beat at 5-min |
| **Tree ensembles** | **LightGBM, XGBoost, CatBoost** | **Default workhorse on tabular features** |
| Sequence DL | LSTM/GRU, TCN | Marginal over trees unless raw sequences |
| Transformer | TFT, PatchTST, N-BEATS/N-HiTS | Multi-horizon quantiles |
| Foundation TS | Chronos, TimesFM, Moirai | Ensemble members / cold start |
| Meta-model | Secondary classifier on primary signal | Bet selection/sizing — high ROI |
| Ensembling | GBDT + CatBoost + linear stack | Variance reduction beats exotic models |

LightGBM's GOSS+EFB are the standard for tabular speed/accuracy: `[Fact, High]` [LightGBM paper](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf). TFT (`[Fact, High]` [arXiv](https://arxiv.org/pdf/1912.09363v1)) and PatchTST (`[Fact, High]` [arXiv](https://arxiv.org/abs/2211.14730)) are the main deep forecaster candidates.

### 7.4 Validation done right (non-negotiable)
1. **No shuffling, ever.** Time order is sacred.
2. **Purged K-Fold + embargo** (remove label-overlap rows; buffer serial correlation). `[Fact, High]` [Purged CV](https://en.wikipedia.org/wiki/Purged_cross-validation)
3. **Combinatorial Purged CV (CPCV)** → distribution of Sharpe / probability of backtest overfitting.
4. **Deflated Sharpe Ratio (DSR) / PSR** with disclosed trial count. `[Fact, High]` [Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
5. **Naive baselines always:** always-up, random-walk, buy-and-hold, EWMA/HAR vol.
6. **All-in cost model** per segment (below).
7. **Independent paper-trade OOS** before any live capital.

> **Warning from the literature:** with only N=10 configurations tried, expected in-sample Sharpe is ~1.57 even when true OOS Sharpe is 0. `[Fact, High]` [Bailey et al.](https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf)

### 7.5 NSE cost model (round-trip, observed 2026-10-07)
| Component | Equity intraday | Index futures | Options |
|---|---|---|---|
| Brokerage (Zerodha) | 0.03% or ₹20/order | 0.03% or ₹20/order | ₹20/order |
| STT | 0.025% sell | 0.05% sell (from Apr 2026) | 0.15% sell premium |
| Exchange txn (NSE) | ~0.00297% | ~0.00183% | ~0.03553% of premium |
| Stamp duty (buy) | 0.003% | 0.002% | 0.003% |
| SEBI fee | ₹10/crore | ₹10/crore | ₹10/crore |
| GST | 18% on (brokerage+fees) | same | same |

Sources: `[Fact, High]` [Zerodha charges](https://zerodha.com/charges/), `[Fact, High]` [NSE STT](https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax). Third-party estimate: cash-equity all-in round trip ≈ **10.63 bps**, of which ~78% is statutory, identical across brokers. `[Estimate, Med]` [AlphaBench](https://www.alphabench.in/blog/intraday-trading-costs-nse-tax)

**Worked examples `[Estimate, Med]`:**
- Equity: buy 1,000 @ ₹100, sell @ ₹100.50 → gross ₹500, costs ≈ ₹82.8 → net ≈ ₹417 → breakeven ≈ **8.3 bps**.
- Nifty futures, 1 lot (75) notional ~₹18.75L round trip → costs ≈ ₹1,108 ≈ **5.9 bps**.
- Nifty ATM option premium ₹150→₹160, lot 75 → gross ₹750, costs ≈ ₹75.3 ≈ **0.67% of premium**.

**5-min signal-to-noise:** Nifty annualized vol ~15–18% → daily σ≈1% → 5-min σ≈1%/√75 ≈ **11–12 bps**. Friction ≈ one full 5-min standard deviation. `[Estimate, Med]`

### 7.6 Reference architecture
```
 EXTERNAL SOURCES                          GLOBAL / OVERNIGHT ALIGNER
 NSE bhavcopy (EOD) | 1-min bars |          GIFT Nifty, SPX, NDX,
 India VIX | FII/DII (EOD) |                USDINR, Brent  (TZ-aware)
 option chain (paid) | L2 ticks (paid)
            |                                      |
            v                                      v
     RAW LAKE (Parquet/S3, symbol/date) --> PIT FEATURE STORE
            |                                (event_ts + knowledge_ts;
            |                                 Feast / DuckDB + Postgres)
            |                                      |
            +------------------+-------------------+
                               v
        RESEARCH / TRAINING PLANE  (separate machine)
        purged K-fold + embargo, CPCV, DSR/PBO, MLflow
        LightGBM/CatBoost  |  TFT/PatchTST  |  ensembling
                               |
                    versioned model artifact
                               v
   INFERENCE            SIGNAL / RISK            EXECUTION
   batch EOD / 5-min -> meta-label filter,   -> Nautilus or broker API
   model server         vol-scaled sizing,       (Strategy ID, static IP,
   (CPU; GPU if DL)     kill-switch               order tagging)
                               |
                               v
   MONITORING / DRIFT (PSI/KS, rolling IC, live-vs-backtest) 
             + PAPER TRADING -> LIVE (reconciliation, slippage, P&L)
```

### 7.7 Training hardware & cost
| Workload | CPU/RAM | GPU | Cost `[Estimate, Low]` |
|---|---|---|---|
| Tabular GBDT (10M×300) | 16–32 cores / 64–128 GB | none | ~$1–5/run |
| Deep (PatchTST/TFT), 10M seq | 16 cores / 64 GB | 1×16–24 GB (L4/A10G) | ~$20–200/experiment |
| Full research program (1–2 mo) | — | spot GPU + CPU | **~$300–1,500** |
| Live inference (tabular, 5-min) | 2–4 cores / 4–8 GB | none | ~$10–40/mo |

A single workstation with one 24 GB GPU + 128 GB RAM handles the entire tabular program. `[Opinion, Med]`

### 7.8 Difficulty & MVP effort
| Milestone | Difficulty (1–10) | Effort `[Estimate]` | Realistic outcome |
|---|---|---|---|
| EOD research prototype (free data, vol + direction, LightGBM, purged walk-forward) | **3** | **3–6 person-weeks** | Validated research result; low probability of live edge |
| 5-min model (free data, PIT store, GBDT + TFT, CPCV, event-driven cost backtest, paper trade) | **6** | **8–16 person-weeks** | Vol edge plausible; directional edge marginal |
| Production intraday system (redundancy, monitoring, algo registration, risk/kill-switch) | **9** | **6–12 person-months** | Ops/cost dominate, not ML |

### 7.9 The genuinely hard parts (ranked)
1. **Non-stationarity / regime change** — models decay fast.
2. **Low signal-to-noise** — at 5-min edge is often below noise and costs.
3. **Overfitting** — many features, short history; CPCV/DSR mandatory.
4. **Execution costs & slippage** — 6–12 bps kills marginal edges.
5. **Data quality / point-in-time** — survivorship, restatements, timestamp leakage.
6. **Look-ahead** — subtle (full-sample normalization, future VWAP, option OI snapshot timing).
7. **Capacity / liquidity** — edge vanishes with size.
8. **Latency / data cost** — L2/clean chain expensive; colocation not retail-accessible.
9. **Survivorship** — especially single-stock cross-section.

---

## 8. Business & pricing

### 8.1 Monetization options (if ever productised)
| Model | Feasibility | Notes |
|---|---|---|
| **Own-account trading** (the primary plan) | Highest legal ease | No RA/IA; still subject to SEBI algo/static-IP/order-tagging rules `[Fact, High]` |
| Sell **clean data API** | Medium | Requires NSE/vendor license + redistribution rights `[Fact, High]` |
| Sell **signals / predictions** | Low without RA registration | "Research services for consideration" → RA registration; black-box → explicitly RA `[Fact, High]` |
| Sell **model-as-a-service / MCP** | Medium | Still signals if it names securities/expected moves |
| Sell **backtesting/tooling** | Medium-High | Crowded, but less regulated than advice |
| **Audited track record as a content/credibility asset** | High | Differentiator; nobody in India has it `[Inference, High]` |

### 8.2 Competitor price points (INR/month unless noted)
| Product | Entry | Mid | Premium |
|---|---|---|---|
| Tradetron | Free | ₹1,000–₹2,083 | ₹4,167–₹7,500 |
| AlgoTest | ₹499 | ₹2,499 | ₹5,999–₹14,999 |
| Quantsapp | Free | ₹1,667 | ₹4,000–₹8,000 |
| StockEdge | ₹399 | ₹1,499 | ₹2,499 |
| Trendlyne | Free | ₹310–₹1,500 | ₹1,500 |
| Sensibull | Free | ~₹1,300 list | — |
| TradingView | Free | ₹995–₹2,095 | ₹4,195–₹17,333 |
| Danelfin (US) | Free | $29 | $79–$179 |
| Data-only (Dhan) | — | ₹499 | — |
| Kite Connect | Free (no data) | ₹500 | — |

### 8.3 Unit economics & distribution
- **Data cost is near-zero** for the free stack; the dominant cost is **engineering time** and **GPU compute (~$300–1,500 for a full research program)**. `[Estimate, Med]`
- **If selling signals:** must price above ~₹500–₹2,000/month to cover SEBI compliance + data licensing + support, against incumbents already at that band. `[Inference, Med]`
- **Distribution channels:** Reddit/YouTube/X communities, TradingQnA/ValuePickr, broker ecosystems (Zerodha/Streak, Dhan algo marketplace, Tradetron marketplace), open-source "build in public." `[Inference, Med]`
- **Partnership angle:** a validated NSE forecaster could plug into Tradetron/Dhan algo marketplaces as a signal provider — but that immediately triggers RA registration. `[Inference, High]`

---

## 9. Differentiation & scope

### 9.1 Where we can win
1. **Audited, timestamped, cost-aware track record.** The one thing nobody in India publishes. `[Inference, High]`
2. **Volatility-first + meta-labeling**, not naive price regression — an honest, defensible framing. `[Opinion, High]`
3. **Leakage-free validation as a feature** (purged CV, DSR, PBO) — turns "trust me" into "check the numbers." `[Opinion, High]`
4. **India-specific cost/execution realism** baked in. `[Opinion, High]`
5. **Compliance-ready architecture** (RA-ready, algo-registration-ready) before competitors are forced to retrofit. `[Inference, Med]`

### 9.2 The sharpest wedge
**Do not build "predict Nifty at 5-min with deep learning."** Build a **Nifty/Bank-Nifty intraday *volatility* and *risk* forecaster with a meta-labeled directional overlay, validated with purged CV + deflated Sharpe and a full NSE cost model, on free data.** This is the narrowest thing that (a) can demonstrably work, (b) cannot be faked by leakage, and (c) lays the groundwork for everything else. `[Opinion, High]`

### 9.3 Smallest viable MVP
- **Universe:** Nifty 50 index (spot + futures continuous).
- **Horizons:** 1-day and 1-hour.
- **Data:** free — HF minute dataset + NSE bhavcopy + India VIX + GIFT Nifty/overnight global, yield-free.
- **Model:** HAR/EWMA baseline vs LightGBM vs one TFT/PatchTST.
- **Validation:** purged walk-forward + CPCV + DSR; naive baselines; full cost model.
- **Deliverable:** an honest report: "here is OOS volatility R² and directional edge net of costs, with kill criteria." `[Opinion, High]`

### 9.4 What NOT to build (yet)
- Sub-minute/HFT or order-book alpha (needs licensed L2 + colocation). `[Opinion, High]`
- Single-stock illiquid cross-section (survivorship, impact). `[Opinion, High]`
- A paid signal product before an audited record + RA registration. `[Inference, High]`
- Deep learning as the first model (start with GBDT + linear baselines). `[Opinion, High]`
- Exact-price prediction (a category of error). `[Opinion, High]`

### 9.5 Potential moats
- **Proprietary, point-in-time, corporate-action-clean NSE dataset** assembled painstakingly from free sources. `[Inference, Med]`
- **An audited public track record** (credibility compounds; hard to fake). `[Inference, High]`
- **Compliance/registration head start** as the regulatory wall rises. `[Inference, Med]`

---

## 10. Risks & unknowns

| # | Risk | Type | Severity | Likelihood | Notes |
|---|---|---|---|---|---|
| 1 | **No exploitable directional edge after costs** | Market | Very High | High | Base rate from SEBI + literature `[Fact, High]` |
| 2 | **Overfitting / backtest-live divergence** | Technical | Very High | High | Universal practitioner complaint `[Opinion, High]` |
| 3 | **Data quality / corporate actions / survivorship** | Technical | High | High | Free data is unadjusted; needs pipeline `[Fact, High]` |
| 4 | **Regulatory: RA/IA + algo rules if distributed** | Legal | High | High (if selling) | SEBI Feb-2025 circular `[Fact, High]` |
| 5 | **NSE/broker data licensing (redistribution)** | Legal | High | Med | No redistribution in broker terms `[Fact, High]` |
| 6 | **Non-stationarity / regime change** | Market | High | High | Models decay `[Opinion, High]` |
| 7 | **Execution slippage / latency worse than modeled** | Technical | High | Med-High | Websocket drops, 9:15 gaps `[Fact, Med]` |
| 8 | **Cost inflation (STT up 2026)** | Market | Med | High | Erodes edge `[Fact, Med]` |
| 9 | **Platform dependency (broker API changes)** | Platform | Med | Med | Kite history depth changed before `[Fact, Med]` |
| 10 | **Scraping blocks / IP bans / cloud-IP bans** | Technical | Med | High | Akamai wall `[Fact, High]` |
| 11 | **Talent/time: full pipeline is 6–12 months** | Execution | Med | Med | Ops dominates `[Estimate, Med]` |
| 12 | **DPDP/privacy if holding user PII** | Legal | Low-Med | Med | Only if productised `[Fact, Med]` |

**Counter-case (must be internalised):** 93% (FY22–24) and 87.7% (FY26) of retail derivative traders lose money; 25% of gross P&L went to transaction costs; FPI/prop profits are 96–97% algo-generated. Any claim that ML lets a retail participant predict NSE intraday at high accuracy must first beat this population-level evidence. `[Fact, High]` [SEBI](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html)

---

## 11. Opportunities & novel additions

- **Audited "Prediction Scoreboard" as a product/community asset.** Publish timestamped OOS predictions and outcomes. Nobody does this; it directly attacks the credibility gap. `[Inference, High]`
- **Volatility/risk API for options sellers** — forecast realized vol vs India VIX to size premium-selling; a more defensible edge than direction. `[Inference, Med]`
- **Open-source the data pipeline** (point-in-time NSE store) as distribution/credibility; monetise the model, not the data. `[Inference, Med]`
- **Meta-labeling overlay as a standalone "should I take this trade?" filter** that plugs into existing platforms (Tradetron/Dhan marketplaces). `[Inference, Med]`
- **Regime detector** (VIX bands, HMM) as a safety switch that gates any trading. `[Inference, Med]`
- **Cross-asset overnight GIFT Nifty / US lead-lag feature** for open-gap context (evidence: GIFT Nifty leads the index, information share ~62%). `[Fact, Med]`
- **Options chain / PCR / OI features** (free snapshots) once the core works. `[Inference, Med]`
- **NSE-specific cost & slippage simulator** released as a library — high utility, low competition. `[Inference, Med]`

---

## 12. Recommended next steps

### The cheapest, fastest sequence
1. **Week 0 (days):** Download `xxparthparekhxx/indian-stock-market-minute-data` and `rissin/nse-options-intraday`; ingest NSE bhavcopy + India VIX. Build a DuckDB/Parquet store. `[Opinion, High]`
2. **Weeks 1–3 — FREE, KILL-CRITERION EXPERIMENT:** Nifty daily/1-hour, **volatility only**, HAR/EWMA vs LightGBM, purged walk-forward + embargo, DSR. **Kill if you can't beat HAR OOS.** `[Opinion, High]`
3. **Weeks 4–6:** Add meta-labeled directional overlay; event-driven cost-aware backtest with the full NSE cost model; paper-trade logic. **Kill if net edge <10 bps/trade.** `[Opinion, High]`
4. **Weeks 7–16 (only if 3 passes):** Move to 5-min; add PIT feature store; test TFT/PatchTST vs LightGBM; paper trade 1–3 months. `[Opinion, Med]`
5. **Month 5+ (only if paper trading survives):** Productionise; handle static IP, order tagging, monitoring, kill-switch; consider RA registration only if distributing. `[Opinion, High]`

### Cheapest validation experiments
- **Landing-page test** for "audited NSE intraday forecast" to gauge demand — zero code.
- **Concierge test:** manually produce 30 days of timestamped forecasts and publish outcomes.
- **Data audit:** quantify gaps/adjustments in the free HF dataset vs NSE bhavcopy for Nifty and 20 liquid stocks — this alone determines whether the free data premise holds.

---

## 13. Open questions for planning

1. **Instrument:** Nifty 50 index futures, Bank Nifty, or a liquid-stock cross-section? (Changes data, cost, and feature set.)
2. **Target:** direction, volatility, or quantile? (Strong recommendation: volatility first.)
3. **Horizon priority:** is 5-min truly required, or is 1-hour/daily acceptable? (5-min roughly triples effort and halves expected edge.)
4. **Trading or research-only?** If trading, which broker (drives API, cost, and algo-compliance path)?
5. **Own-account only, or eventual distribution?** Determines whether SEBI RA/IA registration and NSE data licensing are in scope.
6. **Hardware:** what GPU/RAM does the training machine have? Determines whether deep models are worth attempting.
7. **Risk capital & loss tolerance** for any paper/live phase.
8. **Data licensing posture:** comfortable running purely on Upstox/HF-derived data internally, or must we license NSE/authorised-vendor data from day one?
9. **Willingness to publish an audited track record** (the key differentiator).
10. **Time budget** — is this a 1-month research sprint or a 6–12 month build?

---

## 14. Sources

**Access date for all URLs: 2026-10-07.** P = primary, S = secondary.

### Data sources — official NSE
- (P) NSE all reports / bhavcopy — https://www.nseindia.com/all-reports
- (P) NSE historical index data — https://www.nseindia.com/reports-indices-historical-index-data
- (P) NSE real-time market data subscription — https://www.nseindia.com/static/market-data/real-time-data-subscription
- (P) NSE EOD & historical data subscription — https://www.nseindia.com/static/market-data/eod-historical-data-subscription
- (P) NSE data usage & sharing policy — https://www.nseindia.com/static/market-data/nse-data-policy
- (P) NSE Terms of Use (scraping prohibition) — https://www.nseindia.com/static/nse-terms-of-use
- (P) NSE India VIX white paper — https://nsearchives.nseindia.com/web/sites/default/files/inline-files/white_paper_IndiaVIX.pdf

### Data sources — broker APIs
- (P) Zerodha Kite Connect pricing & products — https://zerodha.com/products/api/
- (P) Zerodha charges (STT/costs) — https://zerodha.com/charges/
- (P) Upstox V3 historical candle API — https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/
- (P) Upstox rate limits — https://upstox.com/developer/api-documentation/rate-limiting/
- (P) DhanHQ trading/data APIs — https://dhanhq.co/trading-apis
- (P) DhanHQ intraday historical (5 yr) — https://docs.dhanhq.co/api/v2/historical-data/get-intraday-historical
- (S) Dhan data API ₹499 + GST — https://dhan.co/support/platforms/dhanhq-api/how-does-the-dhanhq-data-api-subscription-work
- (P) Angel One SmartAPI — https://smartapi.angelone.in/
- (P) Fyers API — https://myapi.fyers.in/
- (P) ICICI Direct Breeze API — https://www.icicidirect.com/futures-and-options/api/breeze
- (S) Broker market-data comparison — https://stocksdeveloper.in/documentation/supported-brokers/api-comparison/market-data

### Data sources — vendors & aggregators
- (P) TrueData market data APIs — https://www.truedata.in/market-data-apis
- (P) GlobalDatafeeds APIs — https://globaldatafeeds.in/apis
- (P) EODData membership — https://www.eoddata.com/products/default.aspx
- (P) EODHD pricing — https://eodhd.com/pricing
- (P) Alpha Vantage premium — https://www.alphavantage.co/premium/
- (P) Twelve Data pricing — https://twelvedata.com/pricing
- (P) Databento venues (NSE covered) — https://databento.com/venues
- (P) TickData NSE — https://www.tickdata.com/equity-data/national-stock-exchange-of-india
- (P) GoCharting NSE market data — https://gocharting.com/trading/market-data/nse

### Free datasets
- (P) HF `xxparthparekhxx/indian-stock-market-minute-data` — https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data
- (P) HF `rissin/nse-options-intraday` — https://huggingface.co/datasets/rissin/nse-options-intraday
- (P) Kaggle Nifty 50 minute 2015–2026 — https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data
- (P) Kaggle NSE 1990–2021 — https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020
- (P) GitHub `debaonline4u/NSE-Data` — https://github.com/debaonline4u/NSE-Data

### Open source
- (P) microsoft/qlib — https://github.com/microsoft/qlib
- (P) ranaroussi/yfinance — https://github.com/ranaroussi/yfinance
- (P) jugaad-py/jugaad-data — https://github.com/jugaad-py/jugaad-data
- (P) swapniljariwala/nsepy (dead) — https://github.com/swapniljariwala/nsepy
- (P) marketcalls/openalgo — https://github.com/marketcalls/openalgo
- (P) AI4Finance-Foundation/FinRL — https://github.com/AI4Finance-Foundation/FinRL
- (P) polakowo/vectorbt — https://github.com/polakowo/vectorbt
- (P) nautechsystems/nautilus_trader — https://github.com/nautechsystems/nautilus_trader
- (P) sktime/sktime — https://github.com/sktime/sktime
- (P) unit8co/darts — https://github.com/unit8co/darts
- (P) online-ml/river — https://github.com/online-ml/river
- (P) bashtage/arch — https://github.com/bashtage/arch
- (P) eslazarev/purged-cross-validation — https://github.com/eslazarev/purged-cross-validation
- (P) huseinzol05/Stock-Prediction-Models — https://github.com/huseinzol05/Stock-Prediction-Models

### Evidence & academic
- (P) Gu, Kelly & Xiu, RFS 2020 — https://academic.oup.com/rfs/article/33/5/2223/5879773
- (P) Gao, Han, Li & Zhou, "Market intraday momentum," JFE 2018 — https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351 ; SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866
- (P) Heston, Korajczyk & Sadka, JF 2010 — https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2010.01573.x
- (P) Cont, Kukanov & Stoikov, price impact of order book events — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822
- (P) Zhang, Zohren & Roberts, deep OFI, Math. Finance 2023 — https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413
- (P) Bailey & López de Prado, Deflated Sharpe Ratio — https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- (P) Bailey et al., pseudo-mathematics of backtests — https://www.davidhbailey.com/dhbpapers/backtest-pseudo.pdf
- (P) Ke, Kelly & Xiu, text-based returns, NBER w26186 — https://www.nber.org/system/files/working_papers/w26186/w26186.pdf
- (P) CNN stock prediction reproducibility audit — https://github.com/matla91/cnn-stock-prediction-reproducibility-study
- (P) Radfar 2025, false-positive price prediction — https://www.nature.com/articles/s41599-025-04761-8
- (S) Nifty intraday regime study (FDR-corrected) — http://bazaarbaazi.com/research/nifty-regime/
- (S) GIFT Nifty lead-lag — https://www.mdpi.com/1911-8074/18/9/527

### Regulation, costs & legal
- (P) SEBI algo-trading circular, 4 Feb 2025 — https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html
- (P) SEBI 93% retail F&O loss study, PR 37/2024 — https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html
- (P) SEBI FY26 F&O study — https://www.sebi.gov.in/sebi_data/attachdocs/aug-2026/1787236407005.pdf
- (P) NSE STT — https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax
- (S) Intraday trading costs analysis — https://www.alphabench.in/blog/intraday-trading-costs-nse-tax
- (P) Kite Connect terms (no redistribution) — https://kite.trade/terms/
- (P) SEBI Research Analysts (Third Amendment) Regs 2024 — https://taxguru.in/sebi/sebi-research-analysts-third-amendment-regulations-2024.html

### Competitors
- (P) Tradetron pricing — https://tradetron.tech/pages/pricing
- (P) AlgoTest pricing — https://algotest.in/pricing
- (P) Quantsapp pricing — https://www.quantsapp.com/pricing
- (P) Trendlyne plans — https://trendlyne.com/subscription/plans/
- (P) StockEdge — https://web.stockedge.com/
- (P) QuantConnect pricing — https://www.quantconnect.com/pricing
- (P) Danelfin pricing — https://danelfin.com/pricing
- (P) Kavout pricing — https://www.kavout.com/pricing-plans
- (P) Tickeron pricing — https://tickeron.com/pricing/
- (P) TradingView pricing — https://www.tradingview.com/pricing/
- (P) Univest — https://univest.in/
- (P) DalalAI — https://dalalai.com/
- (P) Sensibull — https://sensibull.com/

### Voice of customer
- (S) r/IndiaAlgoTrading threads — https://www.reddit.com/r/IndiaAlgoTrading/
- (S) r/IndianStreetBets threads — https://www.reddit.com/r/IndianStreetBets/
- (S) r/algotrading threads — https://www.reddit.com/r/algotrading/
- (S) r/quantindia thread — https://www.reddit.com/r/quantindia/comments/1svmf34/api_for_data/
- (S) Hacker News threads — https://news.ycombinator.com/item?id=46154491 ; https://news.ycombinator.com/item?id=14943605
- (S) ValuePickr quant investing — https://forum.valuepickr.com/t/quant-investing/35451
- (S) Zerodha TradingQnA — https://tradingqna.com/t/full-algo-automation-in-python-for-indian-markets/192235

---

*Report compiled by Ideate. Findings are research, not investment or legal advice. Verify SEBI circulars, NSE tariffs and data-vendor terms against primary sources before acting. Hand this file to plan mode next.*
