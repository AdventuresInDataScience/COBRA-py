from __future__ import annotations

from click.testing import CliRunner

import cobra_py.cli as cli_module


def test_sweep_forwards_output_path_per_seed(tmp_path, monkeypatch):
    data_file = tmp_path / "data.csv"
    data_file.write_text("x\n1\n", encoding="utf-8")

    captured_args: list[list[str]] = []

    def _fake_run_main(*, args, standalone_mode):
        captured_args.append(list(args))
        return None

    monkeypatch.setattr(cli_module.run, "main", _fake_run_main)

    runner = CliRunner()
    output_root = tmp_path / "sweep_output"
    result = runner.invoke(
        cli_module.main,
        [
            "sweep",
            "--data",
            str(data_file),
            "--seeds",
            "11",
            "--seeds",
            "22",
            "--objective",
            "sharpe",
            "--output",
            str(output_root),
        ],
    )

    assert result.exit_code == 0
    assert len(captured_args) == 2

    seed11 = str(output_root / "seed_11")
    seed22 = str(output_root / "seed_22")

    assert "--output" in captured_args[0]
    assert "--output" in captured_args[1]
    assert seed11 in captured_args[0]
    assert seed22 in captured_args[1]
