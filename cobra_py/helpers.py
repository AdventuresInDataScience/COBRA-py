from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
import yaml

from cobra_py.data.loader import load_ohlcv
from cobra_py.data.preprocessor import preprocess
from cobra_py.backtest.engine import run_backtest
from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config
from cobra_py.reporting.report import generate_report
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.nevergrad_runner import run_nevergrad
from cobra_py.search.space import build_config_space
from cobra_py.search.tpe_runner import run_tpe
from cobra_py.validation.walk_forward import walk_forward_validate


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "default.yaml"
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _extract_ohlcv_from_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(_normalize_name(p) for p in col if str(p).strip()) for col in data.columns]
    else:
        data.columns = [_normalize_name(c) for c in data.columns]

    def pick_column(aliases: list[str]) -> str | None:
        cols = list(data.columns)
        for alias in aliases:
            if alias in cols:
                return alias
        for alias in aliases:
            prefix = f"{alias}_"
            for c in cols:
                if c.startswith(prefix):
                    return c
        return None

    mapping = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        # Prefer raw close over adjusted close so OHLC consistency checks remain valid.
        "close": ["close", "c", "adj_close", "adjusted_close", "close_adj"],
        "volume": ["volume", "vol", "v"],
    }

    selected_cols: dict[str, str] = {}
    missing: list[str] = []
    for target, aliases in mapping.items():
        col = pick_column(aliases)
        if col is None:
            missing.append(target)
        else:
            selected_cols[target] = col

    if missing:
        raise ValueError(f"Could not map yfinance columns for {missing}; available columns: {list(data.columns)}")

    out = pd.DataFrame({k: data[v] for k, v in selected_cols.items()}, index=data.index)
    out.index.name = "datetime"
    return out


def fetch_yfinance_ohlcv(symbol: str = "SPY", start: str = "2018-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("yfinance is required for fetch_yfinance_ohlcv. Install with `pip install yfinance`.") from exc

    raw = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for symbol '{symbol}'")

    ohlcv = _extract_ohlcv_from_yfinance(raw)
    return load_ohlcv(ohlcv, min_bars=1)


def make_objective_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": cfg["objective"].get("name", "sharpe"),
        "composite_weights": cfg["objective"].get("composite_weights", [0.5, 0.3, 0.1, 0.1]),
        "max_drawdown_cap": float(cfg["objective"].get("max_drawdown_cap", 0.20)),
        "complexity_penalty": cfg["objective"].get("complexity_penalty", 0.02),
        "min_trades": cfg["objective"].get("min_trades", 10),
        "n_entry_rules": int(cfg["policy"].get("n_entry_rules", 3)),
        "n_exit_rules": int(cfg["policy"].get("n_exit_rules", 1)),
    }


def get_active_registry(cfg: dict[str, Any]):
    ind_cfg = cfg.get("indicators", {})
    return build_registry_from_config(
        DEFAULT_REGISTRY,
        include=ind_cfg.get("include"),
        exclude=ind_cfg.get("exclude"),
        param_ranges=ind_cfg.get("param_ranges"),
    )


def list_available_optimisers() -> list[str]:
    return ["dehb", "nevergrad", "tpe"]


def list_available_objectives() -> list[str]:
    return ["sharpe", "calmar", "sortino", "ulcer", "max_return", "max_return_dd_cap", "composite"]


def _run_with_optimiser(name: str, **kwargs):
    key = str(name).strip().lower()
    if key == "dehb":
        return run_dehb(**kwargs)
    if key == "nevergrad":
        ng_algo = kwargs.pop("nevergrad_algorithm", "NGOpt")
        return run_nevergrad(**kwargs, optimiser_name=str(ng_algo))
    if key == "tpe":
        return run_tpe(**kwargs)
    raise ValueError(f"Unknown optimiser '{name}'. Available: {list_available_optimisers()}")


def run_optimiser(
    source: str | Path | pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    run_walk_forward: bool = False,
    evaluate_oos: bool = False,
) -> dict[str, Any]:
    cfg = deepcopy(config) if config is not None else load_config(config_path)
    cfg = deep_update(cfg, overrides or {})
    if output_path is not None:
        cfg.setdefault("output", {})["path"] = str(output_path)

    data = load_ohlcv(
        source,
        freq=cfg["data"].get("freq"),
        min_bars=int(cfg["data"].get("min_bars", 500)),
    )
    train, test = preprocess(data, cfg["data"])

    registry = get_active_registry(cfg)
    cache = precompute_all(train, registry, n_jobs=int(cfg["indicators"].get("n_jobs", -1)))
    cs = build_config_space(
        n_entry_rules=int(cfg["policy"].get("n_entry_rules", 3)),
        n_exit_rules=int(cfg["policy"].get("n_exit_rules", 1)),
        registry=registry,
        seed=int(cfg["optimiser"].get("seed", 42)),
        backtest_config=cfg["backtest"],
    )

    result = _run_with_optimiser(
        cfg["optimiser"].get("name", "dehb"),
        cache=cache,
        data=train,
        config_space=cs,
        obj_config=make_objective_config(cfg),
        backtest_config=cfg["backtest"],
        budget=int(cfg["optimiser"].get("budget", 200)),
        seed=int(cfg["optimiser"].get("seed", 42)),
        dehb_backend=cfg["optimiser"].get("dehb_backend", "auto"),
        min_fidelity=float(cfg["optimiser"].get("min_fidelity", 0.2)),
        max_fidelity=float(cfg["optimiser"].get("max_fidelity", 1.0)),
        n_workers=int(cfg["optimiser"].get("n_workers", 1)),
        nevergrad_algorithm=cfg["optimiser"].get("nevergrad_algorithm", "NGOpt"),
    )

    wf_result = None
    oos_metrics = None
    if evaluate_oos and len(test) > 0:
        cache_test = precompute_all(test, registry, n_jobs=int(cfg["indicators"].get("n_jobs", -1)))
        oos_metrics = run_backtest(result.best_policy, cache_test, test, cfg["backtest"])

    if run_walk_forward and bool(cfg["validation"].get("walk_forward", True)):
        def optimise_fold(train_df: pd.DataFrame, full_cfg: dict[str, Any]):
            fold_cache = precompute_all(train_df, registry, n_jobs=1)
            fold_cs = build_config_space(
                n_entry_rules=int(full_cfg["policy"].get("n_entry_rules", 3)),
                n_exit_rules=int(full_cfg["policy"].get("n_exit_rules", 1)),
                registry=registry,
                seed=int(cfg["optimiser"].get("seed", 42)),
                backtest_config=cfg["backtest"],
            )
            return _run_with_optimiser(
                cfg["optimiser"].get("name", "dehb"),
                cache=fold_cache,
                data=train_df,
                config_space=fold_cs,
                obj_config=make_objective_config(cfg),
                backtest_config=cfg["backtest"],
                budget=max(20, int(cfg["optimiser"].get("budget", 200)) // 5),
                seed=int(cfg["optimiser"].get("seed", 42)),
                dehb_backend=cfg["optimiser"].get("dehb_backend", "auto"),
                min_fidelity=float(cfg["optimiser"].get("min_fidelity", 0.2)),
                max_fidelity=float(cfg["optimiser"].get("max_fidelity", 1.0)),
                n_workers=int(cfg["optimiser"].get("n_workers", 1)),
                nevergrad_algorithm=cfg["optimiser"].get("nevergrad_algorithm", "NGOpt"),
            )

        wf_result = walk_forward_validate(
            data=test,
            optimise_fn=optimise_fold,
            config=cfg,
            n_splits=int(cfg["validation"].get("n_splits", 3)),
            train_pct=float(cfg["validation"].get("train_pct", 0.7)),
            registry=registry,
        )

    payload = generate_report(result, wf_result, cfg["output"].get("path", "./results/"))
    return {
        "config": cfg,
        "train": train,
        "test": test,
        "result": result,
        "report": payload,
        "oos_metrics": oos_metrics,
        "walk_forward": wf_result,
    }


def summarise_reports(named_reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, payload in named_reports.items():
        summary = payload.get("summary", {})
        best_metrics = payload.get("best_metrics", {})
        rows.append(
            {
                "run": name,
                "optimiser": summary.get("optimiser_name"),
                "objective": summary.get("objective"),
                "best_metric_name": summary.get("best_metric_name"),
                "best_metric_value": summary.get("best_metric_value"),
                "best_score": summary.get("best_score"),
                "evals": summary.get("n_evaluations"),
                "total_return": best_metrics.get("total_return"),
                "sharpe_ratio": best_metrics.get("sharpe_ratio"),
                "sortino_ratio": best_metrics.get("sortino_ratio"),
                "calmar_ratio": best_metrics.get("calmar_ratio"),
                "ulcer_index": best_metrics.get("ulcer_index"),
                "max_drawdown": best_metrics.get("max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def plot_equity_curves(
    named_reports: dict[str, dict[str, Any]],
    normalize: bool = True,
    title: str = "Strategy Equity Curves",
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    x_index: dict[str, Any] | None = None,
    show_range_slider: bool = True,
):
    backend_key = str(backend).strip().lower()
    if backend_key not in {"matplotlib", "plotly"}:
        raise ValueError("backend must be either 'matplotlib' or 'plotly'")

    if backend_key == "matplotlib":
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("matplotlib is required for plot_equity_curves with backend='matplotlib'. Install with `pip install matplotlib`.") from exc

        fig, ax = plt.subplots(figsize=(10, 5))
        for name, payload in named_reports.items():
            eq = np.asarray(payload.get("best_metrics", {}).get("equity_curve", []), dtype=float)
            if len(eq) < 2:
                continue
            series = eq / max(eq[0], 1e-12) if normalize else eq

            x_vals = None
            if x_index and name in x_index:
                candidate = np.asarray(x_index[name])
                if len(candidate) == len(series):
                    x_vals = pd.to_datetime(candidate, errors="coerce")
                    if np.any(pd.notna(x_vals)):
                        ax.plot(x_vals, series, linewidth=1.8, label=name)
                    else:
                        ax.plot(series, linewidth=1.8, label=name)
                else:
                    ax.plot(series, linewidth=1.8, label=name)
            else:
                ax.plot(series, linewidth=1.8, label=name)

        ax.set_title(title)
        ax.set_xlabel("Date" if x_index else "Bars")
        ax.set_ylabel("Normalized equity" if normalize else "Equity")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=140)
        return fig, ax

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("plotly is required for plot_equity_curves with backend='plotly'. Install with `pip install plotly`.") from exc

    fig = go.Figure()
    for name, payload in named_reports.items():
        eq = np.asarray(payload.get("best_metrics", {}).get("equity_curve", []), dtype=float)
        if len(eq) < 2:
            continue
        series = eq / max(eq[0], 1e-12) if normalize else eq

        x_vals: Any = np.arange(len(series))
        if x_index and name in x_index:
            candidate = np.asarray(x_index[name])
            if len(candidate) == len(series):
                parsed = pd.to_datetime(candidate, errors="coerce")
                if np.any(pd.notna(parsed)):
                    x_vals = parsed

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=series,
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date" if x_index else "Bars",
        yaxis_title="Normalized equity" if normalize else "Equity",
        hovermode="x unified",
        legend_title_text="Run",
    )
    fig.update_xaxes(rangeslider_visible=bool(show_range_slider))

    if save_path is not None:
        output = Path(save_path)
        if output.suffix.lower() in {".html", ".htm"}:
            fig.write_html(str(output), include_plotlyjs="cdn")
        else:
            try:
                fig.write_image(str(output))
            except ValueError as exc:  # pragma: no cover - requires kaleido
                raise ValueError("Saving Plotly static images requires kaleido. Install with `pip install kaleido`, or save as .html.") from exc

    return fig
