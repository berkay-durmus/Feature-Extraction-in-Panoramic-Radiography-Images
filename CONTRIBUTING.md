# Contributing

```bash
git clone https://github.com/berkay-durmus/Feature-Extraction-in-Panoramic-Radiography-Images.git
cd Feature-Extraction-in-Panoramic-Radiography-Images
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Before opening a pull request:

```bash
ruff format . && ruff check --fix .
pytest
```

- Work on a branch and open a pull request; do not push to `main`.
- A new or changed feature needs a test on synthetic geometry in `tests/test_features.py`.
- Thresholds belong in named constants at the top of their module.
- Add a line to the `[Unreleased]` section of `CHANGELOG.md`.
- Never commit real patient data, including in tests and screenshots.
