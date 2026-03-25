from __future__ import annotations

from pathlib import Path

import click
import yaml

from cobra_py.data.loader import load_ohlcv
from cobra_py.data.preprocessor import preprocess
from cobra_py.indicators.precompute import precompute_all
from cobra_py.indicators.registry import DEFAULT_REGISTRY, build_registry_from_config, list_indicator_specs
from cobra_py.reporting.report import generate_report
from cobra_py.search.dehb_runner import run_dehb
from cobra_py.search.nevergrad_runner import run_nevergrad
from cobra_py.search.space import build_config_space
from cobra_py.search.tpe_runner import run_tpe
from cobra_py.validation.walk_forward import walk_forward_validate


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        root = Path(__file__).resolve().parents[1]
        config_path = str(root / "configs" / "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_config_sections(cfg: dict) -> dict:
    for section in ("data", "indicators", "policy", "backtest", "objective", "optimiser", "validation", "output"):
        cfg.setdefault(section, {})
    return cfg


@click.group()
def main() -> None:
    """COBRA-py CLI."""


@main.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--output", "output_path", default=None, type=click.Path())
@click.option("--objective", default=None, type=str)
@click.option("--seed", default=None, type=int)
@click.option("--budget", default=None, type=int)
def run(data_path: str, config_path: str | None, output_path: str | None, objective: str | None, seed: int | None, budget: int | None) -> None:
    cfg = _ensure_config_sections(_load_config(config_path))

    if objective is not None:
        cfg["objective"]["name"] = objective
    if seed is not None:
        cfg["optimiser"]["seed"] = seed
    if budget is not None:
        cfg["optimiser"]["budget"] = budget
    if output_path is not None:
        cfg["output"]["path"] = output_path

    data_cfg = cfg["data"]
    ind_cfg = cfg.get("indicators", {})
    active_registry = build_registry_from_config(
        DEFAULT_REGISTRY,
        include=ind_cfg.get("include"),
        exclude=ind_cfg.get("exclude"),
        param_ranges=ind_cfg.get("param_ranges"),
    )

    data = load_ohlcv(data_path, freq=data_cfg.get("freq"), min_bars=int(data_cfg.get("min_bars", 500)))
    train, test = preprocess(data, data_cfg)

    cache = precompute_all(train, active_registry, n_jobs=int(cfg["indicators"].get("n_jobs", -1)))
    cs = build_config_space(
        n_entry_rules=int(cfg["policy"].get("n_entry_rules", 3)),
        n_exit_rules=int(cfg["policy"].get("n_exit_rules", 1)),
        registry=active_registry,
        seed=int(cfg["optimiser"].get("seed", 42)),
        backtest_config=cfg["backtest"],
    )

    obj_cfg = {
        "objective": cfg["objective"].get("name", "sharpe"),
        "composite_weights": cfg["objective"].get("composite_weights", [0.5, 0.3, 0.1, 0.1]),
        "max_drawdown_cap": cfg["objective"].get("max_drawdown_cap", 0.20),
        "complexity_penalty": cfg["objective"].get("complexity_penalty", 0.02),
        "min_trades": cfg["objective"].get("min_trades", 10),
        "n_entry_rules": int(cfg["policy"].get("n_entry_rules", 3)),
        "n_exit_rules": int(cfg["policy"].get("n_exit_rules", 1)),
    }
    bt_cfg = cfg["backtest"]

    opt_name = cfg["optimiser"].get("name", "dehb")
    budget_n = int(cfg["optimiser"].get("budget", 200))
    seed_n = int(cfg["optimiser"].get("seed", 42))

    if opt_name == "nevergrad":
        result = run_nevergrad(
            cache,
            train,
            cs,
            obj_cfg,
            bt_cfg,
            budget_n,
            seed=seed_n,
            optimiser_name=str(cfg["optimiser"].get("nevergrad_algorithm", "NGOpt")),
            num_workers=int(cfg["optimiser"].get("nevergrad_num_workers", 1)),
        )
    elif opt_name == "tpe":
        result = run_tpe(
            cache,
            train,
            cs,
            obj_cfg,
            bt_cfg,
            budget_n,
            seed=seed_n,
            multivariate=bool(cfg["optimiser"].get("tpe_multivariate", True)),
            group=bool(cfg["optimiser"].get("tpe_group", True)),
            n_startup_trials=int(cfg["optimiser"].get("tpe_n_startup_trials", 20)),
            constant_liar=bool(cfg["optimiser"].get("tpe_constant_liar", False)),
        )
    else:
        if opt_name != "dehb":
            raise click.ClickException("Unknown optimiser. Use one of: dehb, nevergrad, tpe")
        result = run_dehb(
            cache,
            train,
            cs,
            obj_cfg,
            bt_cfg,
            budget_n,
            seed=seed_n,
            mutation_factor=float(cfg["optimiser"].get("dehb_mutation_factor", 0.8)),
            crossover_rate=float(cfg["optimiser"].get("dehb_crossover_rate", 0.7)),
            population_size=int(cfg["optimiser"].get("dehb_population_size", 24)),
            dehb_backend=str(cfg["optimiser"].get("dehb_backend", "auto")),
            min_fidelity=float(cfg["optimiser"].get("min_fidelity", 0.2)),
            max_fidelity=float(cfg["optimiser"].get("max_fidelity", 1.0)),
            n_workers=int(cfg["optimiser"].get("n_workers", 1)),
        )

    wf_result = None
    if bool(cfg["validation"].get("walk_forward", True)):
        def optimise_fold(train_df, full_cfg):
            fold_cache = precompute_all(train_df, active_registry, n_jobs=1)
            fold_cs = build_config_space(
                n_entry_rules=int(full_cfg["policy"].get("n_entry_rules", 3)),
                n_exit_rules=int(full_cfg["policy"].get("n_exit_rules", 1)),
                registry=active_registry,
                seed=seed_n,
                backtest_config=bt_cfg,
            )
            fold_budget = max(20, budget_n // 5)
            if opt_name == "nevergrad":
                return run_nevergrad(
                    fold_cache,
                    train_df,
                    fold_cs,
                    obj_cfg,
                    bt_cfg,
                    fold_budget,
                    seed=seed_n,
                    optimiser_name=str(cfg["optimiser"].get("nevergrad_algorithm", "NGOpt")),
                    num_workers=int(cfg["optimiser"].get("nevergrad_num_workers", 1)),
                )
            if opt_name == "tpe":
                return run_tpe(
                    fold_cache,
                    train_df,
                    fold_cs,
                    obj_cfg,
                    bt_cfg,
                    fold_budget,
                    seed=seed_n,
                    multivariate=bool(cfg["optimiser"].get("tpe_multivariate", True)),
                    group=bool(cfg["optimiser"].get("tpe_group", True)),
                    n_startup_trials=int(cfg["optimiser"].get("tpe_n_startup_trials", 20)),
                    constant_liar=bool(cfg["optimiser"].get("tpe_constant_liar", False)),
                )
            return run_dehb(
                fold_cache,
                train_df,
                fold_cs,
                obj_cfg,
                bt_cfg,
                fold_budget,
                seed=seed_n,
                mutation_factor=float(cfg["optimiser"].get("dehb_mutation_factor", 0.8)),
                crossover_rate=float(cfg["optimiser"].get("dehb_crossover_rate", 0.7)),
                population_size=int(cfg["optimiser"].get("dehb_population_size", 24)),
                dehb_backend=str(cfg["optimiser"].get("dehb_backend", "auto")),
                min_fidelity=float(cfg["optimiser"].get("min_fidelity", 0.2)),
                max_fidelity=float(cfg["optimiser"].get("max_fidelity", 1.0)),
                n_workers=int(cfg["optimiser"].get("n_workers", 1)),
            )

        wf_result = walk_forward_validate(
            data=test,
            optimise_fn=optimise_fold,
            config=cfg,
            n_splits=int(cfg["validation"].get("n_splits", 3)),
            train_pct=float(cfg["validation"].get("train_pct", 0.7)),
            registry=active_registry,
        )

    payload = generate_report(result, wf_result, cfg["output"].get("path", "./results/"))
    metric_name = payload["summary"].get("best_metric_name", payload["summary"].get("objective", "metric"))
    metric_value = payload["summary"].get("best_metric_value")
    if isinstance(metric_value, (int, float)):
        click.echo(f"Best {metric_name}: {float(metric_value):.4f}")
    click.echo(f"Best score: {payload['summary']['best_score']:.4f}")
    click.echo(f"Output directory: {cfg['output'].get('path', './results/')}")


@main.command()
@click.option("--result", "result_path", required=True, type=click.Path(exists=True))
def report(result_path: str) -> None:
    p = Path(result_path)
    click.echo(p.read_text(encoding="utf-8"))


@main.command("indicators")
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
def indicators_cmd(config_path: str | None) -> None:
    """List available indicators and parameter ranges (after config filters)."""
    cfg = _ensure_config_sections(_load_config(config_path))
    ind_cfg = cfg.get("indicators", {})
    active_registry = build_registry_from_config(
        DEFAULT_REGISTRY,
        include=ind_cfg.get("include"),
        exclude=ind_cfg.get("exclude"),
        param_ranges=ind_cfg.get("param_ranges"),
    )
    payload = {
        "count": len(active_registry),
        "indicators": list_indicator_specs(active_registry),
    }
    click.echo(yaml.safe_dump(payload, sort_keys=False))


@main.command()
@click.option("--policy", "policy_path", required=True, type=click.Path(exists=True))
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
def validate(policy_path: str, data_path: str) -> None:
    click.echo("Validation command placeholder for MVP. Use 'run' to evaluate policies.")


@main.command()
@click.option("--data", "data_path", required=True, type=click.Path(exists=True))
@click.option("--seeds", multiple=True, type=int, required=True)
@click.option("--objective", default="sharpe", type=str)
@click.option("--config", "config_path", default=None, type=click.Path(exists=True))
@click.option("--output", "output_path", default="./results/ensemble", type=click.Path())
def sweep(data_path: str, seeds: tuple[int, ...], objective: str, config_path: str | None, output_path: str) -> None:
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    for s in seeds:
        seed_output_path = str(output_root / f"seed_{s}")
        run.main(
            args=["--data", data_path, "--objective", objective, "--seed", str(s), "--output", seed_output_path, "--config", config_path] if config_path else ["--data", data_path, "--objective", objective, "--seed", str(s), "--output", seed_output_path],
            standalone_mode=False,
        )
    click.echo(f"Sweep complete. Results under: {output_root}")


if __name__ == "__main__":
    main()


