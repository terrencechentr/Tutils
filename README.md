# tutils

`tutils` is a lightweight utility kit for machine-learning and data-science workflows. It provides handy helpers for random seeding, timing, logging, JSONL IO, matrix filtering, quick plotting, and model memory estimation.

## Installation

- From Git directly  
  `pip install git+https://github.com/yourname/tutils.git`

- As a Git submodule inside another repo (keeps a clean src layout)  
  1) `git submodule add https://github.com/yourname/tutils.git third_party/tutils`  
  2) `pip install -e third_party/tutils`  or  `pip install third_party/tutils`

- Local development / debugging  
  `pip install -e .`

## Features
- Random helpers: `set_seed`, `random_str`, `str2base62`
- Timing: `Timer`
- Logging: `get_logger`, `Colors`
- IO: `load_jsonl`, `dump_jsonl`, `append_jsonl`, `transform_jsonl`
- Matrix helpers: `topk_matrix`, `maxp_matrix`
- Plotting: `plot_box`, `plot_violin`, `plot_histogram`, `plot_multi_features`
- Resource stats: `calculate_model_memory`, `_format_bytes`

## Quick Example

```python
from tutils import (
    set_seed, Timer, get_logger, load_jsonl, plot_box,
    calculate_model_memory,
)

set_seed(42)
logger = get_logger("demo")

with Timer("step"):
    data = load_jsonl("data.jsonl")
    logger.info("loaded %s items", len(data))

plot_box({"acc": [0.7, 0.75, 0.8]}, title="acc distribution")
```

## Development Notes
- Python 3.8+
- Dependencies: numpy, pandas, matplotlib, torch, colorlog, tqdm
- Build: `python -m build` (requires `pip install build`)
- When publishing as a submodule, keep only core files (`src/`, `pyproject.toml`, `setup.py`, `README.md`) and avoid committing `build/`, `dist/`, `*.egg-info/`.

