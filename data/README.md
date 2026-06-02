# Data

Raw tourism data is **not redistributed** in this repository (respect the source
terms; only officially published files should be used). This folder ships only a
skeleton plus a cached OpenStreetMap walking graph for offline reproducibility.

## What's included
- `raw/osm/macau_walk.graphml` — cached OSM pedestrian network for the historic
  centre (Map data © OpenStreetMap contributors, ODbL).
- `raw/MANIFEST.txt` — provenance notes.

## What you must acquire (see `docs/04_data_sources.md` for exact URLs/steps)
1. **DSEC visitor arrivals** (monthly, by entry point) — official spreadsheets from
   the Macau Statistics and Census Service. Place the `.xlsx` files under
   `data/raw/dsec/`.
2. **MGTO per-attraction statistics** (optional) — from MGTO annual tourism reports.
   A synthetic proxy fallback is generated automatically if these are absent.
3. **OpenStreetMap** — already cached here; re-downloaded automatically by EXP-01 if
   the cache is removed.

## Build the processed datasets
```bash
python -m src.ingest_data --source all      # writes data/processed/*.parquet
# or, to use the synthetic MGTO fallback only:
python -m src.ingest_data --source dsec --fallback
```

Do **not** commit raw or processed data files (see `.gitignore`); they are derived
from third-party sources and can be large.
