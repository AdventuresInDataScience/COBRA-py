# COBRA-py Documentation

This folder contains the high-level design and implementation documents for COBRA-py.

## Canonical Project Root

The canonical package root is the workspace root: COBRA-py.

Key directories at root:

- `cobra_py/`: source package
- `configs/`: default runtime config
- `examples/`: runnable scripts
- `tests/`: test suite
- `docs/`: design and planning documents
- `Archive/`: historical material

## Run and Verify

From the COBRA-py root:

```bash
uv sync
uv run pytest tests -q
uv run cobra-py run --data smoke_data.csv --config smoke_config.yaml
```

Optional showcase run:

```bash
uv run python examples/spy_showcase.py
```

## Documents In This Folder

- `COBRA-py_Summary_Design.md`: architecture and rationale
- `COBRA-py_Implementation_Plan.md`: execution plan and milestones
- `quickstart.ipynb`: notebook walkthrough

## Notes

- Use `uv run ...` for commands to ensure the managed environment is used.
- Smoke outputs are written to `smoke_results/`.
- Example outputs are written under `examples/showcase_results/`.
