# Data Directory Layout

This repository does not include datasets or experiment outputs.

Expected structure:

```text
data/
  raw/
    advbench_100.csv
    harmBench_100.csv
  combined/
  DTA_ablation/
  eval/
  ...
```

Minimum input requirement for most attack scripts:

- `data/raw/advbench_100.csv` or `data/raw/harmBench_100.csv`
- CSV must contain either a `goal` column or a `harmful` column

All result directories under `data/` are treated as generated artifacts and are ignored by Git by default.
