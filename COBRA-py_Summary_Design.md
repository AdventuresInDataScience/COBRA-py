# Rule-Based Direct Policy Optimisation (RBDPO)
## Summary & Design Document — v1.0

*A Framework for Interpretable Automated Trading Strategy Discovery*

---

> **MVP Scope Notice.** This document describes Version 1.0 of RBDPO, which operates exclusively on single-instrument OHLCV time-series data. Two planned extensions are scoped but not yet implemented: **(1)** incorporation of external macroeconomic and alternative time-series data as additional rule inputs (e.g. `policy_rate > 2%`, `yield_curve_spread < 0`); and **(2)** a portfolio-level optimisation loop enabling joint strategy discovery across multiple instruments with shared risk constraints. A placeholder section for the portfolio extension is included at the end of this document. These extensions will be specified in a subsequent design document once the MVP is validated.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Is Being Built](#2-what-is-being-built)
3. [Motivation and Value](#3-motivation-and-value)
4. [System Architecture and Design Choices](#4-system-architecture-and-design-choices)
5. [Rule Archetypes and the Policy Language](#5-rule-archetypes-and-the-policy-language)
6. [Indicator Universe](#6-indicator-universe)
7. [Stop-Loss and Take-Profit Design](#7-stop-loss-and-take-profit-design)
8. [Optimiser Selection and Rationale](#8-optimiser-selection-and-rationale)
9. [Indicator Precomputation Strategy](#9-indicator-precomputation-strategy)
10. [Backtesting Engine and Evaluation](#10-backtesting-engine-and-evaluation)
11. [Objective Function Design](#11-objective-function-design)
12. [Novelty and Comparison to Related Work](#12-novelty-and-comparison-to-related-work)
13. [Potential Limitations and Caveats](#13-potential-limitations-and-caveats)
14. [Future Extensions](#14-future-extensions)
15. [References](#15-references)

---

## 1. Executive Summary

Rule-Based Direct Policy Optimisation (RBDPO) is a framework for automatically discovering interpretable algorithmic trading strategies. Rather than learning latent representations through neural networks, RBDPO searches directly over a structured, human-readable policy space: a bounded set of binary indicator-based rules combined with parametric stop-loss and take-profit logic. Each candidate policy is evaluated via full offline backtesting, and the best policy is found using black-box, gradient-free global optimisers capable of handling the resulting mixed-integer, conditional, non-differentiable search problem.

The result is a complete, transparent trading strategy whose every condition is human-interpretable, auditable, and directly deployable — without requiring any post-hoc explanation or model inference infrastructure. This makes RBDPO well-suited to live execution, regulatory scrutiny, and practitioner oversight, while retaining the automation benefits typically associated with machine learning.

The user specifies a named optimisation objective (e.g. `sharpe`, `calmar`, `sortino`, `max_return`) at run time. The framework is designed so that the objective is a first-class configuration parameter — different users with different risk preferences can drive the search toward qualitatively different strategies from the same policy space.

---

## 2. What Is Being Built

RBDPO is a Python software package that, given historical OHLCV price data and a user-specified objective, automatically discovers optimal rule-based trading strategies. The system operates in three sequential stages:

**Stage 1 — Indicator Precomputation.** A comprehensive library of technical indicators is computed across all relevant parameter combinations and held in memory as NumPy arrays. This is a one-time cost per dataset. During optimisation, evaluating any indicator value is a simple array lookup — no computation occurs inside the hot path.

**Stage 2 — Policy Search.** A black-box optimiser iteratively proposes candidate strategy configurations. Each configuration encodes: which entry rules are active (binary switches), which rule archetypes and indicators are used, what parameters those indicators take, and the full specification of the stop-loss and take-profit blocks. All optimisers support a user-configurable **random seed** for reproducibility. Different seeds can be used to obtain diverse solutions from the same run configuration, which is useful for ensemble building.

**Stage 3 — Evaluation.** Each proposed strategy is backtested using vectorbt. The resulting performance metrics are aggregated into a scalar objective score — determined by the user's chosen objective — and returned to the optimiser. The optimiser iterates toward better solutions within a configured evaluation budget.

The final output is the best-found strategy expressed as a fully human-readable rule set, alongside its complete backtest performance profile.

---

## 3. Motivation and Value

### 3.1 The Problem with Conventional Approaches

Most modern algorithmic trading research applies deep reinforcement learning (DQN, PPO, SAC; Mnih et al. 2015; Schulman et al. 2017) or end-to-end neural architectures directly to market data (Jiang et al. 2017; Liang et al. 2018). While these can achieve empirically strong results in controlled settings, they suffer from several critical problems in practice:

- **Opacity.** Neural policies are black boxes. Understanding why a trade was entered or exited is impossible, creating problems for risk management, debugging, and regulatory compliance (Adadi & Berrada 2018; Rudin 2019).
- **Fragility.** Learned representations can overfit to the statistical regime of training data and fail under regime change, a well-documented problem in financial ML (Prado 2018).
- **Sample complexity.** Policy gradient methods require many environment interactions and careful reward shaping (Sutton & Barto 2018). Offline RL requires additional distributional constraints (Levine et al. 2020).
- **Production burden.** Running a neural policy live requires GPU inference, model serving infrastructure, and ongoing version management.
- **Verification difficulty.** It is very hard to convince a risk committee or regulator that a neural strategy is doing the right thing for the right reason (Rudin 2019).

### 3.2 The Problem with Manual Rule Design

The alternative — manually designing rule-based strategies — is the dominant practitioner approach. However:

- The search space of possible rule combinations is vast, and human intuition tends to explore only a small, familiar region (Murphy 1999).
- Optimising rule parameters by hand or via grid search is computationally inefficient and subject to curve-fitting (Bailey et al. 2017; Harvey et al. 2016).
- There is no systematic way to explore the full conditional structure of the strategy space.

### 3.3 What RBDPO Offers

RBDPO sits between these two poles. It retains the interpretability and deployment simplicity of rule-based strategies while applying the automated search power of modern black-box optimisation. The complexity is moved from the model to the search process.

| Benefit | Description |
|---|---|
| Full interpretability | Every rule is a named indicator comparison. Any trader can read, understand, and verify the strategy. |
| User-defined objective | The optimisation target (Sharpe, Calmar, Sortino, etc.) is a first-class parameter, not a hardcoded constant. |
| Production simplicity | Deployment requires only evaluating a boolean expression over precomputed indicators — no model weights or inference pipeline. |
| Regulatory suitability | Explainable decisions align with regulatory requirements (e.g. MiFID II, SR 11-7). Rule-based strategies are trivially auditable. |
| Regime transparency | Since rules are explicit, a practitioner can reason directly about when conditions are likely to hold and when they may break down. |
| Reproducibility | All optimiser runs accept a random seed, enabling exact reproduction and deliberate exploration of diverse solutions. |
| Low deployment overhead | The entire strategy can be described in a configuration file of a few lines. |
| Practitioner alignment | The framework produces strategies that look like what a systematic trader would write — because it searches the same space. |

---

## 4. System Architecture and Design Choices

### 4.1 Policy Representation

A policy π is defined as:

```
π = {
    entry_rules:  [R₁ ∧ R₂ ∧ ... ∧ Rₙ],   n ≤ N_max
    exit_rules:   [R'₁ ∧ R'₂ ∧ ... ∧ R'ₘ], m ≤ M_max
    sl_block:     { sl_type, sl_params }
    tp_block:     { tp_type, tp_params }
    position_size: method
}
```

Each rule Rᵢ is binary (evaluates to true or false at each timestep). Entry signals fire when all active entry rules are simultaneously true. Exit signals fire when any active exit rule is true, or when the stop-loss/take-profit block triggers.

**Rationale for AND (conjunctive) entry logic.** AND logic directly reflects the practitioner concept of "confluence" — a trade is only taken when multiple independent conditions align simultaneously. This naturally filters out low-confidence setups. OR logic would generate far more frequent, lower-quality signals. More complex Boolean combinations (e.g. `(A AND B) OR C`) are excluded to keep the policy space tractable and the strategy self-explanatory. The AND-only constraint is consistent with the conjunctive rule representation used throughout the GP-trading literature (Allen & Karjalainen 1999; Neely et al. 1997).

**Rationale for bounded rule count.** Limiting each block to N_max rules (default: 3) prevents the policy from degenerating into an overfit conjunction of many individually weak conditions. Regularisation via the objective function (penalising unnecessary active rules) complements this hard bound.

### 4.2 Binary Rule Representation

Every rule is a binary signal: at each bar it is either true or false. This:

- Makes the search space discrete and finite (for any fixed parameter discretisation), which is amenable to combinatorial optimisation.
- Keeps indicator evaluation uniform — any signal is reduced to a boolean array before entering the policy engine.
- Supports vectorised evaluation: combining N rules is N parallel array AND operations, which is extremely fast.

### 4.3 Offline Evaluation

The entire optimisation is offline. The optimiser proposes complete strategies which are backtested end-to-end on historical data — it never interacts with a live or simulated environment step-by-step. This is "direct policy optimisation" in the sense that the full policy is evaluated holistically.

**Rationale.** Online RL requires careful environment design, reward shaping, and is susceptible to distributional shift. Offline evaluation allows the use of proper financial performance metrics as objectives rather than surrogate rewards, consistent with the offline RL framing of Levine et al. (2020) but without distributional constraint machinery, since the policy class itself is simple and bounded.

### 4.4 User-Specified Objective

The objective function is a first-class configuration parameter. Supported named objectives:

| Key | Description |
|---|---|
| `sharpe` | Maximise Sharpe ratio (annualised return / annualised volatility) |
| `calmar` | Maximise Calmar ratio (annualised return / max drawdown) |
| `sortino` | Maximise Sortino ratio (penalises downside volatility only) |
| `max_return` | Maximise total return |
| `composite` | Weighted combination: w₁·Sharpe + w₂·Calmar + w₃·Return − w₄·MaxDD (weights user-configurable) |

### 4.5 Random Seeds and Reproducibility

All supported optimisers expose a `seed` parameter. The same seed with the same configuration guarantees an identical sequence of candidate evaluations. Different seeds with the same configuration intentionally explore different regions of the search space, enabling **solution diversity**: running multiple seeds produces a portfolio of uncorrelated strategies with different risk/return profiles. The seed is recorded in every output artefact.

- **DEHB:** `seed` passed to the `DEHB` constructor.
- **Nevergrad:** set via `optimizer.parametrization.random_state = np.random.RandomState(seed)`.

### 4.6 Language and Technology Stack

The system is implemented entirely in Python. Key libraries:

- **vectorbt** — vectorised backtesting and indicator precomputation.
- **pandas-ta** — pure-Python technical indicator library covering all required indicators, with no system-level C dependencies. Preferred over TA-Lib, which requires a compiled C library that is difficult to install in reproducible environments.
- **Nevergrad / DEHB** — black-box optimisation.
- **ConfigSpace** — conditional hyperparameter space definition.
- **joblib** — parallel indicator precomputation.
- **uv** — fast, reproducible Python environment and dependency management.

**Rationale for pandas-ta over TA-Lib.** TA-Lib requires `libta-lib` to be compiled and installed at the system level, which varies across operating systems and breaks reproducible CI/CD and containerised environments. `pandas-ta` is a pure-Python implementation covering all indicators required by RBDPO and installs with a single `uv add pandas-ta`. The minor speed difference is irrelevant given that indicators are precomputed once.

**Rationale for staying in Python.** An earlier design considered a Julia (FastBack.jl) backend. This was rejected because FastBack.jl is a low-level framework with no built-in indicators, the Julia ecosystem lacks candlestick pattern detection, and the speed advantage over vectorbt's vectorised NumPy operations is marginal when indicators are precomputed.

---

## 5. Rule Archetypes and the Policy Language

Rules are not hardcoded instances (e.g. "price > SMA(200)"). Instead, the system defines a set of **archetypes** — parameterised templates — and the optimiser selects both the archetype and its parameters. This is analogous to grammar-guided genetic programming (Whigham 1995; Dempsey et al. 1998), but uses black-box search rather than evolutionary operators.

### 5.1 Formal Rule Grammar

```
RULE       := COMPARISON | CROSSOVER | BAND_TEST | PATTERN | STAT_TEST | DERIVATIVE

COMPARISON := VALUE OPERATOR VALUE
CROSSOVER  := VALUE crosses_{above|below} VALUE
BAND_TEST  := VALUE {above|below|inside} BAND(period, multiplier)
PATTERN    := EXTREMUM | CONSECUTIVE | SEQUENCE
STAT_TEST  := ZSCORE(indicator, period) OPERATOR threshold
DERIVATIVE := SLOPE|CHANGE(indicator, period) OPERATOR threshold

VALUE      := price | indicator(PARAMS) | VALUE[lag]
OPERATOR   := { >, < }
PARAMS     := period [, additional_params...]
```

### 5.2 Archetype Catalogue

**Comparison Rules** — Test the relative magnitude of two values at the current bar:
- `price > indicator(period)` — e.g. price above a moving average
- `indicator1(p1) > indicator2(p2)` — e.g. fast MA above slow MA
- `indicator(period) > threshold` — e.g. RSI above 50
- `indicator(period) > indicator(period)[lag]` — current value above its own past value

**Crossover Rules** — Test whether a transition occurred at the current bar (edge detection):
- `indicator1(p1) crosses_above indicator2(p2)`
- `price crosses_above indicator(period)`
- `indicator(period) crosses_above threshold`

**Band/Channel Rules** — Test position relative to a computed band:
- `price above band_upper(period, multiplier)` — e.g. above Bollinger upper band
- `price below band_lower(period, multiplier)`
- `indicator(period) inside band(period, multiplier)`

**Pattern Rules** — Test structural conditions over a lookback window:
- `highest(price, period) == price` — price at N-bar high (Donchian-style breakout)
- `lowest(price, period) == price` — price at N-bar low
- `consecutive(condition, count)` — condition true for N consecutive bars
- `price[0] > price[1] > price[2]` — directional sequence (higher closes)

**Derivative Rules** — Test the rate of change or direction of an indicator:
- `change(indicator(period), lag) > threshold` — positive momentum in indicator
- `slope(indicator, period) > 0` — rising indicator
- `acceleration(indicator, period) > 0` — rate of change is itself increasing

**Statistical Rules** — Test relative position in a distribution:
- `zscore(indicator, lookback) > threshold` — indicator significantly above its mean
- `indicator(period) > percentile(indicator, lookback, level)` — percentile rank test

### 5.3 MVP Scope: OHLCV Time-Series Only

In this MVP, all rules derive solely from OHLCV time-series data. Rules requiring external data sources (macroeconomic series, policy rates, yield spreads, sentiment, options flow) are explicitly excluded from v1.0. The planned v2.0 extension will introduce an external time-series data layer. See Section 14.1.

---

## 6. Indicator Universe

The indicator universe is curated to cover the most empirically useful, commonly applied indicators while remaining computationally tractable.

### 6.1 Moving Averages and Trend
- **SMA(period)** — Simple Moving Average
- **EMA(period)** — Exponential Moving Average
- **WMA(period)** — Weighted Moving Average

### 6.2 Oscillators and Momentum
- **RSI(period)** — Relative Strength Index (Wilder 1978)
- **MACD(fast, slow, signal)** — MACD line, signal line, histogram (Appel 1979)
- **Stochastic(k, d, smooth)** — %K and %D (Lane 1984)
- **ROC(period)** — Rate of Change
- **CCI(period)** — Commodity Channel Index (Lambert 1980)

### 6.3 Volatility and Bands
- **Bollinger Bands(period, std, ma_type)** — upper, middle, lower (Bollinger 2001). The band selector (upper/middle/lower) is a rule-level choice, not a precomputed parameter — all three are stored together per combination.
- **ATR(period)** — Average True Range (Wilder 1978). Used both as a signal and as a scaling factor in SL/TP blocks.
- **Keltner Channel(ema_period, atr_period, multiplier)** — upper, lower (Keltner 1960)
- **Donchian Channel(period)** — upper, lower (Donchian 1960)

### 6.4 Trend Strength
- **ADX(period)** — Average Directional Index (Wilder 1978)
- **Parabolic SAR(step, max_step)** — (Wilder 1978)

### 6.5 Volume (when available)
- **OBV** — On-Balance Volume (Granville 1963)
- **VWAP** — Volume Weighted Average Price

### 6.6 Parameter Space per Indicator

| Indicator | Parameters | Typical Range |
|---|---|---|
| SMA / EMA | period | 5 – 200 |
| RSI | period | 7 – 21 |
| MACD | fast, slow, signal | (8–15), (20–30), (7–12) |
| Bollinger | period, std_multiple, ma_type | (10–50), (1.0–3.0), {SMA, EMA} |
| Keltner | ema_period, atr_period, multiplier | (10–30), (10–30), (1.0–3.0) |
| ATR | period | 7 – 21 |
| ADX | period | 10 – 30 |
| Stochastic | k_period, d_period, smooth | (5–21), (3–14), (3–7) |
| Parabolic SAR | step, max_step | (0.01–0.05), (0.1–0.3) |
| Donchian | period | 10 – 100 |

This conditional structure — where the chosen indicator determines which parameters exist — is a key reason standard grid-search or fixed-vector optimisers are poorly suited to this problem.

---

## 7. Stop-Loss and Take-Profit Design

Risk management is a first-class component of the policy. The optimiser selects both the *type* of stop and its *parameters*.

### 7.1 Stop-Loss Archetypes

| Type | Formula | Parameters |
|---|---|---|
| Fixed percentage | entry × (1 − pct) | pct ∈ [0.005, 0.10] |
| ATR multiple | entry − atr_mult × ATR(period) | atr_mult ∈ [1.0, 4.0], period ∈ [7, 21] |
| Structural: swing low | lowest(low, lookback) | lookback ∈ [5, 50] |
| Structural: Bollinger lower | BB_lower(period, std) | same as indicator params |
| Trailing ATR | rolling_max(high) − trail_mult × ATR(period) | trail_mult, period |

**Rationale for ATR-based stops.** ATR-based stops adapt to current volatility, preventing stops that are too tight in volatile conditions or too wide in quiet conditions. This family of stops is well-supported empirically (Kaufman 2013; Pardo 2008).

**Rationale for structural stops.** Swing-low and Bollinger-based stops place the stop at a structurally meaningful level, reducing the risk of stops in arbitrary noise zones.

### 7.2 Take-Profit Archetypes

| Type | Formula | Parameters |
|---|---|---|
| Fixed percentage | entry × (1 + pct) | pct ∈ [0.005, 0.20] |
| ATR multiple | entry + atr_mult × ATR(period) | atr_mult ∈ [1.0, 6.0], period ∈ [7, 21] |
| Risk-reward ratio | entry + rr × (entry − stop_loss) | rr ∈ [1.0, 5.0] |
| Structural: swing high | highest(high, lookback) | lookback ∈ [5, 100] |
| Structural: Bollinger upper | BB_upper(period, std) | same as indicator params |

**Rationale for risk-reward TP.** Expressing the target as a multiple of risk is standard practice (Schwager 1993) and ensures minimum expected value per trade.

---

## 8. Optimiser Selection and Rationale

The policy search problem is **mixed-integer, conditional, black-box, and non-differentiable**. Standard gradient-based methods cannot be applied. Grid search is exponentially expensive.

### 8.1 DEHB — *Primary Recommendation*

DEHB (Awad et al. 2021) combines Differential Evolution (Storn & Price 1997) with multi-fidelity scheduling via Hyperband (Li et al. 2017):

- **Multi-fidelity:** poor configurations are evaluated on shorter data windows first and eliminated early, reserving full backtests for promising candidates.
- **Differential Evolution:** handles mixed discrete/continuous spaces naturally.
- Uses ConfigSpace natively for conditional spaces.
- **Random seed:** passed directly to the `DEHB` constructor.

**Rationale.** DEHB was developed for NAS (Awad et al. 2021) and demonstrates state-of-the-art performance on conditional hyperparameter spaces structurally identical to RBDPO's search space.

### 8.2 Nevergrad (NGOpt) — *Prototyping / Diversity*

Nevergrad (Rapin & Teytaud 2018) is a general-purpose black-box platform with mixed-type parameter support:

- Easy to configure and iterate on during development.
- **Random seed:** set via `optimizer.parametrization.random_state = np.random.RandomState(seed)`.
- Well-suited to ensemble building: running with different seeds produces diverse strategies.

### 8.3 LA-MCTS — *Optional / Hierarchical*

LA-MCTS (Wang et al. 2020) treats the search as a tree problem where each level corresponds to a hierarchical decision. Binary rule switches map naturally onto tree pruning.

- **Random seed:** controlled via the NumPy random state before initialisation.
- Most principled choice for RBDPO's hierarchical policy structure, but requires more integration work than DEHB.

### 8.4 Why Not Standard Bayesian Optimisation?

GP-based BO (Snoek et al. 2012) does not scale to high-dimensional mixed-integer conditional spaces. SMAC3 (Hutter et al. 2011) with random forests is more viable but less efficient than DEHB on NAS-class problems. CMA-ES (Hansen & Ostermeier 2001) and PSO (Kennedy & Eberhart 1995) are designed for continuous spaces and are less natural for highly discrete conditional structures.

---

## 9. Indicator Precomputation Strategy

### 9.1 Core Principle

All indicator arrays are computed once before optimisation begins and held in memory. During each objective function call, retrieving an indicator is an array column lookup (nanoseconds), compared to milliseconds for recomputation.

### 9.2 vectorbt's Precomputation Model

vectorbt's indicator runner produces a 2D NumPy array of shape `(T, K)` where T is the number of bars and K is the number of parameter combinations. Column selection uses a MultiIndex:

```python
bb_upper = INDICATORS['bb'].upper[(20, 2.0, 'sma')]  # pure NumPy column slice
```

### 9.3 Feasibility of Full Precomputation

Full precomputation is feasible with 96 GB RAM. Indicative counts:

- **SMA/EMA:** ~40 arrays per type
- **RSI:** ~15 arrays
- **Bollinger Bands:** ~800 parameter sets → 2,400 arrays (upper/middle/lower stored together)
- **MACD:** ~530 valid combinations
- **Stochastic:** ~1,020 combinations
- **Keltner Channel:** ~2,000 combinations
- **Parabolic SAR, Donchian, ATR, ADX, CCI, ROC:** ~100 arrays combined

Total: ~20,000–100,000 arrays. At 2,500 bars × float32, this occupies ~1–2 GB RAM. Parallelised precomputation completes in 1–3 minutes — an acceptable one-time cost.

### 9.4 Lazy Caching for Out-of-Grid Parameters

For combinations outside the precomputed grid, a lazy `lru_cache`-wrapped fallback computes on demand. The ConfigSpace definition is aligned to the precomputed grid, keeping cache misses rare.

### 9.5 Memory Architecture

```
INDICATORS = {
    'sma':  { period         → np.ndarray(T,) },
    'ema':  { period         → np.ndarray(T,) },
    'rsi':  { period         → np.ndarray(T,) },
    'bb':   { (p, std, type) → BBObject(upper, middle, lower) },
    'macd': { (f, s, sig)    → MACDObject(macd, signal, hist) },
    'atr':  { period         → np.ndarray(T,) },
    ...
}
```

All values are NumPy arrays or lightweight namedtuples of NumPy arrays. No Python-level iteration occurs inside the backtest hot path.

---

## 10. Backtesting Engine and Evaluation

### 10.1 vectorbt as the Backtest Engine

vectorbt (Stańczyk 2021) is chosen because:

- **Vectorised execution.** Portfolio simulation uses NumPy/Numba kernels. A single strategy on 2,500 bars evaluates in ~1–10 ms.
- **Native multi-indicator support.** Precomputed arrays feed directly in without conversion.
- **Built-in stop/take-profit handling.** Native `STOPLOSS` and `TAKEPROFIT` generators handle exit logic efficiently.
- **Rich performance metrics.** Sharpe, Calmar, Sortino, max drawdown, trade-level statistics all available natively.
- **Single-position enforcement.** `max_size=1` prevents position stacking.

### 10.2 Backtest Configuration

- **Data:** Full OHLCV time series (no look-ahead; strictly in-sample)
- **Capital:** Fixed initial cash (configurable; default £10,000)
- **Position sizing:** Fixed fraction or full-cash
- **Max open positions:** 1 (MVP baseline)
- **Costs:** Configurable flat fee and slippage (default: 10bps per trade)
- **Frequency:** Driven by input data granularity

### 10.3 Multi-Fidelity Evaluation (DEHB)

When using DEHB, the `fidelity` parameter maps to the fraction of historical data used. Configurations are promoted from 10% → 25% → 50% → 100% of data as they demonstrate promise, substantially reducing total compute cost.

---

## 11. Objective Function Design

The objective function is user-specified and configurable. The optimiser minimises the returned value (objectives are negated internally).

### 11.1 Named Objectives

| Key | Formula | Notes |
|---|---|---|
| `sharpe` | −Sharpe(π) | Risk-adjusted return; primary default |
| `calmar` | −Calmar(π) | Return per unit of max drawdown |
| `sortino` | −Sortino(π) | Penalises downside volatility only |
| `max_return` | −TotalReturn(π) | Pure return maximisation; higher overfit risk |
| `composite` | −(w₁·S + w₂·C + w₃·R − w₄·MDD) | Weighted blend; weights user-configurable |

### 11.2 Regularisation

A complexity penalty is applied regardless of the chosen objective:

```
J_reg(π) = J(π) + λ · num_active_rules(π)
```

This penalises strategies with more active rules, favouring simpler policies with equivalent performance.

### 11.3 Handling Invalid Strategies

Strategies producing fewer than `min_trades` (default: 10) trades, or for which metrics are NaN, are assigned a large penalty score (999.0).

---

## 12. Novelty and Comparison to Related Work

### 12.1 Genetic Programming for Trading

The earliest systematic use of GP for rule discovery in financial markets was Allen & Karjalainen (1999), who evolved buy/sell rules for the S&P 500 using binary tree representations, finding evidence of predictability before transaction costs but not after. Neely et al. (1997) applied GP to foreign exchange, finding rules that outperformed simple technical benchmarks on several currencies. Becker & Seshadri (2003) extended this to Indian equity markets, reporting strong in-sample but weaker out-of-sample results. Brabazon & O'Neill (2006) provided a comprehensive treatment of GP in computational finance. Chen et al. (2009) introduced a multi-objective GP framework for simultaneous return and risk optimisation.

RBDPO shares the philosophy of searching a discrete rule space but differs fundamentally:

- **No evolutionary operators.** RBDPO uses modern black-box optimisers rather than genetic crossover and mutation. Evolutionary operators applied to tree representations can produce arbitrary, unparseable expressions. RBDPO's policy grammar enforces interpretability by construction.
- **Structured, bounded policy class.** GP trees can grow arbitrarily complex and are prone to code bloat (Koza 1992; Langdon & Poli 1998). RBDPO's fixed-slot structure with N_max rules per block prevents this by design.
- **Conditional parameter handling.** Traditional GP does not manage indicator-specific conditional parameter spaces. RBDPO uses ConfigSpace (Lindauer et al. 2019) for principled conditional encoding.
- **Multi-fidelity optimisation.** DEHB's multi-fidelity mechanism has no equivalent in the GP-trading literature. It is a qualitatively different approach to the same problem.

### 12.2 Hyperparameter Optimisation Frameworks

HPO frameworks — SMAC3 (Hutter et al. 2011), Hyperopt (Bergstra et al. 2013), Optuna (Akiba et al. 2019), DEHB (Awad et al. 2021) — were developed to tune ML pipelines over conditional configuration spaces. RBDPO applies the same algorithmic machinery, but the "model" is a trading rule set and the "training" is a backtest.

This framing is novel in the trading literature. Prior work applies Optuna or Hyperopt to optimise *parameters* of a fixed rule (e.g. the RSI threshold), not to jointly search over rule *structure* and parameters simultaneously. To the best of our knowledge, no prior work applies HPO infrastructure — specifically conditional ConfigSpace definitions combined with multi-fidelity black-box optimisers — to the problem of rule-based trading strategy discovery.

### 12.3 Direct Policy Search and Black-Box RL

Direct policy search (Schmidhuber 2013; Deisenroth et al. 2013) optimises parameterised policies without backpropagation. Evolution Strategies (Salimans et al. 2017; Such et al. 2017) demonstrated that simple black-box search can match deep RL on continuous control tasks. RBDPO is a form of direct policy search over a discrete policy space, which sidesteps the continuous action-space assumptions of ES-based methods.

Compared to offline RL approaches (Fujimoto et al. 2019; Kumar et al. 2020; Levine et al. 2020), RBDPO does not require a learned value function, distributional constraint, or behaviour policy assumption. The policy class is simple enough that direct search is tractable without these mechanisms.

### 12.4 Neural Architecture Search (NAS)

The conditional, hierarchical search space of RBDPO (rule type → indicator → parameters) is structurally analogous to NAS search spaces (Zoph & Le 2017; Liu et al. 2019; Real et al. 2019). NAS methods including BOHB (Falkner et al. 2018), DARTS (Liu et al. 2019), and DEHB (Awad et al. 2021) address the challenge of efficiently searching conditional graph-structured configuration spaces. The mapping is direct: NAS cell type → rule archetype; operation parameters → indicator parameters; cell activation → rule active/inactive binary switch.

### 12.5 Technical Analysis Parameter Optimisation

Brock et al. (1992) found that simple MA crossover and trading range break rules had predictive power for the DJIA over 1897–1986. Sullivan et al. (1999) subsequently demonstrated that data-snooping bias substantially inflated this apparent performance and proposed bootstrap-based corrections. Faber (2007) showed that a simple 10-month MA timing rule improved risk-adjusted returns in long-run asset class data. Hsu et al. (2016) confirmed technical rule profitability using a stepwise test controlling for data-snooping.

RBDPO generalises this line of work by jointly optimising rule *selection*, *combination*, and *parametrisation* simultaneously. The walk-forward protocol and complexity penalty are designed specifically to mitigate the data-snooping bias documented by Sullivan et al. (1999) and Bailey et al. (2017).

### 12.6 Grammar-Guided Symbolic Search

Whigham (1995) and Dempsey et al. (1998) introduced grammar-guided GP, where a formal grammar constrains the expressions that can be evolved. RBDPO's rule archetype catalogue is a finite grammar in the same spirit. The key difference is that RBDPO uses a deliberately restricted, practitioner-aligned grammar (not an arbitrary formal language) and replaces evolutionary operators with modern black-box optimisers.

Kaboudan (2000) and Dempster et al. (2001) applied symbolic regression via GP to financial prediction. While these approaches discover nonlinear mathematical relationships in price data, the expressions they produce are arbitrary formulas rather than structured trading rules. RBDPO produces strategies in the grammar of practitioner language — indicator comparisons, crossovers, threshold tests — which are natively interpretable without expert translation.

### 12.7 Summary of Novelty

To the best of our knowledge, RBDPO is the first framework to combine all of the following in a single system:

1. A formal, practitioner-aligned rule grammar with indicator-conditional parameter spaces
2. Multi-fidelity black-box optimisation (DEHB) applied to the joint problem of rule selection, combination, and parametrisation
3. A user-specified, first-class objective function enabling diverse risk preferences
4. Full reproducibility via seed control and deterministic backtest evaluation
5. A precomputed indicator cache eliminating recomputation during the optimiser hot path

---

## 13. Potential Limitations and Caveats

### 13.1 Overfitting and Backtest Bias

The most significant risk is overfitting to the in-sample period (Harvey et al. 2016; Bailey et al. 2017; Prado 2018). Mitigations include:

- Walk-forward validation as the primary evaluation protocol
- Regularisation via the complexity penalty term
- Statistical significance testing (deflated Sharpe ratio — Bailey & Prado 2014; Monte Carlo permutation tests — Sullivan et al. 1999)
- Limiting the evaluation budget to prevent exhaustive search

### 13.2 Combinatorial Search Space

Despite curation, the combined space of rule archetypes × indicators × parameters × SL/TP types is very large. No optimiser guarantees finding the global optimum. The system finds a *good* strategy within the evaluation budget, not necessarily the *best* possible one.

### 13.3 Non-Stationarity

Financial time series are non-stationary (Lo 2004). A strategy optimised on one regime may fail in another. RBDPO does not explicitly model regime, though the walk-forward protocol partially mitigates this.

### 13.4 Precomputed Indicator Discretisation

Precomputed indicator grids are necessarily discretised. The optimiser can only access values on the grid. Very fine-grained parameter sensitivity (e.g. RSI(13.7) vs RSI(14)) is not captured. In practice this is unlikely to matter — indicators are generally insensitive to small parameter perturbations within a reasonable range (Faber 2007).

### 13.5 Transaction Costs

The default configuration uses simple flat-fee cost models. Strategies discovered under low-cost assumptions may be unprofitable after realistic slippage and market impact. Realistic cost modelling should be applied before any live deployment (Kissell 2013).

### 13.6 Single-Instrument MVP Scope

Version 1.0 targets long-only strategies on a single instrument. Long/short, multi-asset, and portfolio-level extensions are planned (see Section 14.2).

---

## 14. Future Extensions

### 14.1 External Time-Series Data (Planned: v2.0)

The MVP restricts rule inputs to OHLCV-derived indicators. The v2.0 extension will introduce an **external time-series data layer**, allowing any regularly sampled time series to participate in the rule grammar alongside price-derived indicators. Examples:

- **Macroeconomic filters:** `policy_rate > 2.0`, `cpi_yoy > 3.0`, `unemployment_rate < 4.5`
- **Fixed income signals:** `yield_curve_10y2y < 0` (inverted yield curve), `credit_spread > 300bps`
- **Cross-asset signals:** `vix_index > 20`, `dxy_index_trend == "rising"`
- **User-supplied series:** any time series aligned to the price bar frequency

The architecture extension will add: a new `EXTERNAL_COMPARISON` archetype to the rule grammar; an external series registry alongside the indicator cache; and additional conditional slots in the ConfigSpace definition. The policy representation, backtest engine, and optimiser integration remain unchanged — external rules remain binary and participate in the same AND conjunction logic.

Rule archetypes for external series will follow the same pattern as price rules, for example:
- `external_series(name) > threshold`
- `external_series(name) crosses_above threshold`
- `external_series(name) > external_series(name)[lag]`

### 14.2 Portfolio-Level Optimisation Loop (Planned: v3.0)

> **[PLACEHOLDER — To Be Specified]**
>
> This section will describe the planned portfolio-level extension to RBDPO. This is architecturally more complex than single-instrument optimisation and will be fully specified once the MVP has been validated in production.
>
> The high-level intent is:
> - Run RBDPO on each instrument in a portfolio universe to produce per-instrument candidate strategies
> - Introduce a portfolio-level objective layer that evaluates the joint performance of the combined strategy set (e.g. portfolio Sharpe, diversification ratio, correlation-adjusted return)
> - Implement an outer optimisation loop at the portfolio level that selects which per-instrument strategies to activate and at what sizing, subject to risk constraints (e.g. maximum individual instrument weight, target portfolio volatility)
> - Ensure the single-instrument backtesting infrastructure is reused without modification as the inner evaluation loop
>
> Full design will be documented in a subsequent specification once the MVP codebase is validated.

---

## 15. References

Adadi, A. & Berrada, M. (2018). Peeking inside the black-box: A survey on explainable artificial intelligence. *IEEE Access*, 6, 52138–52160.

Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623–2631.

Allen, F. & Karjalainen, R. (1999). Using genetic algorithms to find technical trading rules. *Journal of Financial Economics*, 51(2), 245–271.

Appel, G. (1979). *The Moving Average Convergence-Divergence Trading Method*. Signalert Corporation.

Awad, N., Mallik, N., Hutter, F. & Bergman, E. (2021). DEHB: Evolutionary hyperband for scalable, robust and efficient hyperparameter optimization. *IJCAI 2021*, 2147–2153.

Bailey, D.H. & Prado, M.L. de (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.

Bailey, D.H., Borwein, J., Prado, M.L. de & Zhu, Q.J. (2017). Pseudo-mathematics and financial charlatanism. *Notices of the American Mathematical Society*, 61(5), 458–471.

Becker, L.A. & Seshadri, M. (2003). GP-evolved technical trading rules can outperform buy and hold. *Proc. 5th International Conference on Computational Intelligence and Natural Computing*.

Bergstra, J., Yamins, D. & Cox, D. (2013). Hyperopt: A Python library for optimizing the hyperparameters of machine learning algorithms. *Proc. 12th Python in Science Conference*, 13–20.

Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.

Brabazon, A. & O'Neill, M. (2006). *Biologically Inspired Algorithms for Financial Modelling*. Springer.

Brock, W., Lakonishok, J. & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731–1764.

Chen, S.-H., Kuo, T.-W. & Hoi, K.-M. (2009). Genetic programming and financial trading: How much about "what we know". In *Handbook of Financial Engineering*, 99–154. Springer.

Deisenroth, M.P., Neumann, G. & Peters, J. (2013). A survey on policy search for robotics. *Foundations and Trends in Robotics*, 2(1–2), 1–142.

Dempsey, I., O'Neill, M. & Brabazon, A. (1998). Investigations into the application of grammatical evolution to the discovery of technical trading rules. *Proc. Artificial Neural Networks in Finance and Economics Conference*.

Dempster, M.A.H., Payne, T.W., Romahi, Y. & Thompson, G.W.P. (2001). Computational learning techniques for intraday FX trading using popular technical indicators. *IEEE Transactions on Neural Networks*, 12(4), 744–754.

Donchian, R. (1960). Trend-following methods in commodity price analysis. *Commodity Yearbook*. Commodity Research Bureau.

Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69–79.

Falkner, S., Klein, A. & Hutter, F. (2018). BOHB: Robust and efficient hyperparameter optimization at scale. *ICML 2018*, 1437–1446.

Fujimoto, S., Meger, D. & Precup, D. (2019). Off-policy deep reinforcement learning without exploration. *ICML 2019*, 2052–2062.

Granville, J. (1963). *Granville's New Key to Stock Market Profits*. Prentice-Hall.

Hansen, N. & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.

Harvey, C.R., Liu, Y. & Zhu, H. (2016). ...and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.

Hsu, P.H., Hsu, Y.C. & Kuan, C.M. (2016). Testing the predictability of technical analysis using a new stepwise test without data snooping bias. *Journal of Empirical Finance*, 17(3), 471–484.

Hutter, F., Hoos, H.H. & Leyton-Brown, K. (2011). Sequential model-based optimization for general algorithm configuration. *LION 2011*, 507–523.

Jiang, Z., Xu, D. & Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio management problem. *arXiv:1706.10059*.

Kaboudan, M.A. (2000). Genetic programming prediction of stock prices. *Computational Economics*, 16(3), 207–236.

Kaufman, P.J. (2013). *Trading Systems and Methods* (5th ed.). Wiley.

Keltner, C.W. (1960). *How to Make Money in Commodities*. Keltner Statistical Service.

Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. *Proc. IEEE ICNN*, 4, 1942–1948.

Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*. Academic Press.

Koza, J.R. (1992). *Genetic Programming*. MIT Press.

Kumar, A., Zhou, A., Tucker, G. & Levine, S. (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS 2020*, 33, 1179–1191.

Lambert, D. (1980). Commodity channel index: Tool for trading cyclic trends. *Commodities Magazine*.

Lane, G. (1984). Lane's stochastics. *Technical Analysis of Stocks and Commodities*, 2(3).

Langdon, W.B. & Poli, R. (1998). Fitness causes bloat. In *Soft Computing in Engineering Design and Manufacturing*, 13–22. Springer.

Levine, S., Kumar, A., Tucker, G. & Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv:2005.01643*.

Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. & Talwalkar, A. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization. *JMLR*, 18(185), 1–52.

Liang, Z., Chen, H., Zhu, J., Jiang, K. & Li, Y. (2018). Adversarial deep reinforcement learning in portfolio management. *arXiv:1808.09940*.

Lindauer, M., et al. (2019). SMAC3: A versatile Bayesian optimization package for hyperparameter optimization. *JMLR*, 23(54), 1–9.

Liu, H., Simonyan, K. & Yang, Y. (2019). DARTS: Differentiable architecture search. *ICLR 2019*.

Lo, A.W. (2004). The adaptive markets hypothesis. *Journal of Portfolio Management*, 30(5), 15–29.

Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

Murphy, J.J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.

Neely, C., Weller, P. & Dittmar, R. (1997). Is technical analysis in the foreign exchange market profitable? A genetic programming approach. *Journal of Financial and Quantitative Analysis*, 32(4), 405–426.

Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley.

Prado, M.L. de (2018). *Advances in Financial Machine Learning*. Wiley.

Rapin, J. & Teytaud, O. (2018). Nevergrad — A gradient-free optimization platform. https://github.com/facebookresearch/nevergrad.

Real, E., Aggarwal, A., Huang, Y. & Le, Q.V. (2019). Regularized evolution for image classifier architecture search. *AAAI 2019*, 4780–4789.

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

Salimans, T., Ho, J., Chen, X., Sidor, S. & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. *arXiv:1703.03864*.

Schmidhuber, J. (2013). Deep learning in neural networks: An overview. *Neural Networks*, 61, 85–117.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

Schwager, J.D. (1993). *The New Market Wizards*. HarperBusiness.

Snoek, J., Larochelle, H. & Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. *NeurIPS 2012*, 2951–2959.

Stańczyk, J. (2021). vectorbt. https://github.com/polakowo/vectorbt.

Storn, R. & Price, K. (1997). Differential evolution. *Journal of Global Optimization*, 11(4), 341–359.

Such, F.P., et al. (2017). Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. *arXiv:1712.06567*.

Sullivan, R., Timmermann, A. & White, H. (1999). Data-snooping, technical trading rule performance, and the bootstrap. *Journal of Finance*, 54(5), 1647–1691.

Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Wang, L., Fonseca, R. & Tian, Y. (2020). Learning search space partition for black-box optimization using Monte Carlo tree search. *NeurIPS 2020*, 19511–19522.

Whigham, P.A. (1995). Grammatically-based genetic programming. *Proc. Workshop on Genetic Programming*, 33–41.

Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

Zoph, B. & Le, Q.V. (2017). Neural architecture search with reinforcement learning. *ICLR 2017*.

---

*End of Summary & Design Document — v1.0*
