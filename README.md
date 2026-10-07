# Feature Extraction in Panoramic Radiography Images

Extracts radiographic risk features of the lower third molars (teeth **48** and **38**) and the
mandibular canal from panoramic radiographs, using manually annotated polygons. The output is a
table with one row per tooth, intended as input for downstream classification.

## Install

```bash
pip install -e .            # runtime
pip install -e ".[dev]"     # + pytest, ruff
```

Requires Python 3.10+.

## Usage

Put each image and its annotation in one folder, sharing the file name stem
(`12.jpg` + `12.xml`):

```bash
panoramic-features path/to/folder -o FeatureList.xlsx
panoramic-features path/to/folder -o features.csv --missing -1
```

| Option | Description |
|---|---|
| `-o, --output` | `.xlsx` or `.csv` (default `FeatureList.xlsx`) |
| `--missing VALUE` | Value for features that cannot be computed (default: empty cell) |
| `-v` | Debug logging |

Files without a partner, unreadable images and failing individual features are logged and
skipped; they never abort the run. A failed feature is written as missing for that tooth only.

Python API:

```python
from panoramic_features.pipeline import extract_features

rows = extract_features("12.jpg", "12.xml")  # [Row(image_id, tooth, values), ...]
```

## Annotation format

XML with polygons under `outputs/object/item`, each item having a `name` and a `polygon` with
`x1,y1,x2,y2,...`. Recognised labels: `48`, `38` (teeth) and `Sağ M3`, `Sol M3` (right/left
mandibular canal near the third molar).

## Features

| Column | Meaning |
|---|---|
| `Obscuration` | Contrast `(R - m) / (R + m)` between the tooth/canal overlap (mean `m`) and the reference tooth density `R` (middle multi-Otsu threshold). Positive when the overlap is darker. |
| `Deflection` | Largest bend angle (degrees) between the coronal and apical half of a root, after orienting the tooth so its roots point down. |
| `NarrowingRoots` | Largest relative width loss over 10% of a root's length (0 = none, 1 = pinched off). |
| `NarrowingCanal` | `1 - w_low / w_high` of the canal width along its centreline (5th/95th percentile). |
| `DiscontinuityCanal` | Fraction of the canal outline lying inside the tooth polygon. |
| `CanalDeviation` | Standard deviation (degrees) of the local direction of the canal centreline. |

Tooth orientation uses the principal axis of the tooth polygon; for near-horizontal (impacted)
teeth the root end is assumed to point away from the midline. Thresholds are module-level
constants in `src/panoramic_features/features/` and have not been tuned on clinical data.

## Development

```bash
pytest
ruff check . && ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE).
