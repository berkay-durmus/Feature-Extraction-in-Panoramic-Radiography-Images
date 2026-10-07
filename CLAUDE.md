# CLAUDE.md

Feature extraction for lower third molars (teeth 48, 38) and the mandibular canal from
annotated panoramic radiographs. The README is the source of truth for input format, feature
definitions and Docker usage.

## Commands

```bash
pip install -e ".[dev]"
pytest                                   # tests build their own synthetic geometry
ruff format . && ruff check --fix .
panoramic-features <dir> -o out.xlsx     # also: python -m panoramic_features
python scripts/make_dummy_dataset.py --out demo-data
python scripts/make_figure.py            # regenerates docs/images/overview.png
```

## Architecture

`pipeline.extract_features` loads one image and XML, rasterises polygons to boolean masks and
runs each feature in its own `try`; a failure becomes `NaN` for that cell only. Features live
in `src/panoramic_features/features/` and take plain masks, never mutating them.
`geometry.orient_tooth` rotates a tooth so its roots point down; `roots._split_roots` separates
the roots. Output columns and their order are defined in `export.HEADERS` and `pipeline.FEATURES`.

## Conventions

- Missing values are `float("nan")`, never `-1`.
- Thresholds are named module constants; keep comments short.
- Never push to `main`; use a branch.
- Never add real patient data to the repo.
