from cobra_py.helpers import list_available_objectives, list_available_optimisers, summarise_reports


def test_list_available_optimisers_contains_expected_options():
    opts = list_available_optimisers()
    assert "dehb" in opts
    assert "nevergrad" in opts
    assert "tpe" in opts


def test_list_available_objectives_contains_ulcer_and_sortino():
    objs = list_available_objectives()
    assert "sortino" in objs
    assert "ulcer" in objs


def test_summarise_reports_includes_human_and_ulcer_fields():
    report = {
        "summary": {
            "optimiser_name": "dehb(random-search-mvp)",
            "objective": "ulcer",
            "best_metric_name": "Ulcer index",
            "best_metric_value": 0.05,
            "best_score": 0.07,
            "n_evaluations": 20,
        },
        "best_metrics": {
            "total_return": 0.2,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.4,
            "calmar_ratio": 0.8,
            "ulcer_index": 0.05,
            "max_drawdown": -0.1,
        },
    }

    df = summarise_reports({"run_a": report})
    row = df.iloc[0]

    assert row["run"] == "run_a"
    assert row["best_metric_name"] == "Ulcer index"
    assert row["ulcer_index"] == 0.05
