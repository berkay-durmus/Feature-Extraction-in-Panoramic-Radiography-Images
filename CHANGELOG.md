# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Docker image (multi-stage, non-root), `docker-compose.yml` and `Makefile` shortcuts.
- CI builds and smoke-tests the image and runs on every branch push.
- `scripts/make_dummy_dataset.py` for generating synthetic annotated radiographs and
  `scripts/make_figure.py` for the README figure.
- Detailed README, this changelog, `CONTRIBUTING.md`.

## [1.0.0]

Restructured into an installable package; the original scripts are replaced.

### Added
- `panoramic-features` command line, Python API, XLSX and CSV export (`--missing`).
- Test suite on synthetic geometry, `ruff` configuration and a GitHub Actions workflow.

### Changed
- Images and annotations are paired by file name stem instead of by the first number in the
  name, and unmatched files are reported.
- Annotation items are read one by one instead of aligning two separate tag lists.
- Missing values are empty cells (or `--missing`) instead of `-1`.
- Root deflection is the bend angle between the coronal and apical half of a root; root
  narrowing is the largest relative width loss along a root (continuous, previously 0/1).
- Canal deviation is the standard deviation of the centreline direction in degrees, with
  ends trimmed proportionally instead of a fixed 50/100 points.

### Fixed
- Canal narrowing measured the vertical extent of the contour in image coordinates; it now
  measures canal width along the centreline.
- Features modified the shared mask arrays in place, so results depended on call order.
- One failing feature (or an empty canal) discarded both teeth of an image; failures are now
  isolated per feature and tooth.
- Cropping near the image border used negative slice indices and returned wrong regions.
- Tooth orientation raised `UnboundLocalError` when no contour matched a fixed pixel-area range.
- The right tooth's root narrowing skeletonised the unrotated image.
- Debug figures were created for every image and never closed.
- `requirements.txt` was a UTF-16 `pip freeze` of an entire environment; dependencies now live
  in `pyproject.toml`.

## [0.1.0]

Original scripts (`main.py` and one module per feature).
