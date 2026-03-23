from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import importlib.metadata as im
import pandas as pd
import yfinance as yf
import yaml

from cobra_py.data.loader import load_ohlcv
from cobra_py.data.preprocessor import preprocess
from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY
from cobra_py.reporting.report import generate_report
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.space import build_config_space


# %% 1. Create examples workspace and environment paths
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "configs" / "default.yaml").exists():
            return p
    raise RuntimeError("Could not find project root containing pyproject.toml and configs/default.yaml")


try:
    project_root = find_project_root(Path(__file__).resolve())
except NameError:  # pragma: no cover - interactive fallback
    project_root = find_project_root(Path.cwd())

examples_dir = project_root / "examples"
examples_dir.mkdir(parents=True, exist_ok=True)

print("Project root:", project_root)
print("Examples dir:", examples_dir)


# %% 2. Install package in editable mode and add yfinance
# Run once in a fresh env from project root:
#   uv pip install -e .
#   uv pip install yfinance ipython jupyter matplotlib


# %% 3. Validate imports and runtime paths
print("cobra-py version:", im.version("cobra-py"))
print("yfinance version:", yf.__version__)
print("cobra_py module path:", Path(__import__("cobra_py").__file__).resolve())


def deep_update(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_spy_ohlcv(start: str = "2018-01-01", end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    raw = yf.download("SPY", start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No SPY data returned from yfinance. Check internet access and ticker symbol.")
    raw = raw.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    raw.index.name = "datetime"
    return raw[["open", "high", "low", "close", "volume"]]


# %% 4. Load base configuration from file
base_cfg = load_config(project_root / "configs" / "default.yaml")
print({
    "objective": base_cfg["objective"]["name"],
    "budget": base_cfg["optimiser"]["budget"],
    "seed": base_cfg["optimiser"]["seed"],
})


# %% 5. Define in-script configuration overrides
alt_overrides = {
    "optimiser": {"budget": 20, "seed": 123},
    "objective": {"name": "calmar", "min_trades": 1},
    "policy": {"n_entry_rules": 2, "n_exit_rules": 1},
    "indicators": {"n_jobs": 1},
}
cfg_alt = deep_update(base_cfg, alt_overrides)
print({"base_objective": base_cfg["objective"]["name"], "alt_objective": cfg_alt["objective"]["name"]})


# %% 6. Download SPY OHLC data with yfinance
spy_ohlcv = fetch_spy_ohlcv(start="2018-01-01", interval="1d")
print("Rows:", len(spy_ohlcv))
print("Has missing:", bool(spy_ohlcv.isna().any().any()))
print("Monotonic index:", bool(spy_ohlcv.index.is_monotonic_increasing))
print(spy_ohlcv.tail(3))


def run_experiment(base_cfg: dict, cfg_overrides: dict, output_dir: Path) -> dict:
    cfg = deep_update(base_cfg, cfg_overrides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output"]["path"] = str(output_dir)

    data = load_ohlcv(spy_ohlcv, freq=cfg["data"].get("freq"), min_bars=int(cfg["data"].get("min_bars", 500)))
    train, _test = preprocess(data, cfg["data"])

    cache = precompute_all(train, DEFAULT_REGISTRY, n_jobs=int(cfg["indicators"].get("n_jobs", 1)))

    cs = build_config_space(
        n_entry_rules=int(cfg["policy"].get("n_entry_rules", 3)),
        n_exit_rules=int(cfg["policy"].get("n_exit_rules", 1)),
        registry=DEFAULT_REGISTRY,
        seed=int(cfg["optimiser"].get("seed", 42)),
    )

    obj_cfg = {
        "objective": cfg["objective"].get("name", "sharpe"),
        "composite_weights": cfg["objective"].get("composite_weights", [0.5, 0.3, 0.1, 0.1]),
        "complexity_penalty": cfg["objective"].get("complexity_penalty", 0.02),
        "min_trades": cfg["objective"].get("min_trades", 10),
        "n_entry_rules": int(cfg["policy"].get("n_entry_rules", 3)),
        "n_exit_rules": int(cfg["policy"].get("n_exit_rules", 1)),
    }

    result = run_dehb(
        cache=cache,
        data=train,
        config_space=cs,
        obj_config=obj_cfg,
        backtest_config=cfg["backtest"],
        budget=int(cfg["optimiser"].get("budget", 50)),
        seed=int(cfg["optimiser"].get("seed", 42)),
    )

    report = generate_report(result=result, wf_result=None, output_path=output_dir)
    print(f"Best score: {report['summary']['best_score']:.4f}")
    print(f"Wrote: {output_dir / 'result.json'}")
    return report


# %% 7. Run package flow using file-based config
report_file_cfg = run_experiment(base_cfg, {}, examples_dir / "results_file_config")


# %% 8. Run package flow using overridden config
report_override_cfg = run_experiment(base_cfg, alt_overrides, examples_dir / "results_override_config")


# %% 9. Compare outputs across config variants
comparison = pd.DataFrame(
    [
        {
            "run": "file_config",
            "objective": base_cfg["objective"]["name"],
            "seed": base_cfg["optimiser"]["seed"],
            "budget": base_cfg["optimiser"]["budget"],
            "best_score": report_file_cfg["summary"]["best_score"],
            "evals": report_file_cfg["summary"]["n_evaluations"],
        },
        {
            "run": "override_config",
            "objective": cfg_alt["objective"]["name"],
            "seed": cfg_alt["optimiser"]["seed"],
            "budget": cfg_alt["optimiser"]["budget"],
            "best_score": report_override_cfg["summary"]["best_score"],
            "evals": report_override_cfg["summary"]["n_evaluations"],
        },
    ]
)
print(comparison)


# %% 10. Persist effective config snapshots
effective_dir = examples_dir / "effective_configs"
effective_dir.mkdir(parents=True, exist_ok=True)

with (effective_dir / "base_effective.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(base_cfg, f, sort_keys=False)
with (effective_dir / "override_effective.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg_alt, f, sort_keys=False)

print("Saved artifacts:")
for p in sorted(examples_dir.iterdir()):
    print(" -", p.name)


