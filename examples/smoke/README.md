# Smoke Example Assets

This folder contains the quick end-to-end smoke test assets:

- `smoke_data.csv`: small synthetic OHLCV dataset
- `smoke_config.yaml`: low-budget config for fast validation runs
- `results/`: expected output location for smoke runs

Run from project root:

```bash
uv run cobra-py run --data examples/smoke/smoke_data.csv --config examples/smoke/smoke_config.yaml
```
