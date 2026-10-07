"""Render the overview figure used in the README.

python scripts/make_figure.py                       # synthetic sample
python scripts/make_figure.py IMAGE XML -o out.png  # your own annotated radiograph
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_dummy_dataset import render, write_xml  # noqa: E402

from panoramic_features.annotations import parse_annotation  # noqa: E402
from panoramic_features.imaging import load_gray  # noqa: E402
from panoramic_features.pipeline import FEATURES, extract_features  # noqa: E402

COLORS = {"48": "#e4572e", "Sağ M3": "#17bebb", "38": "#f3a712", "Sol M3": "#76b041"}
PAIRS = (("48", "Sağ M3"), ("38", "Sol M3"))
MARGIN = 70


def draw(ax, polygons, labels, bounds=None, fontsize=11):
    for label in labels:
        for points in polygons.get(label, []):
            pts = np.array(points)
            ax.add_patch(plt.Polygon(pts, fill=False, ec=COLORS[label], lw=1.8))
            if bounds is None:
                x, y = pts[:, 0].mean(), pts[:, 1].min() - 25
                ax.text(x, y, label, color=COLORS[label], ha="center", fontsize=fontsize,
                        weight="bold")  # fmt: skip


def crop_bounds(polygons, labels, shape):
    pts = np.vstack([np.array(p) for lab in labels for p in polygons.get(lab, [])])
    (x0, y0), (x1, y1) = pts.min(axis=0) - MARGIN, pts.max(axis=0) + MARGIN
    return max(int(x0), 0), min(int(x1), shape[1]), max(int(y0), 0), min(int(y1), shape[0])


def figure(image_path: Path, xml_path: Path, out: Path) -> None:
    image, polygons = load_gray(image_path), parse_annotation(xml_path)
    rows = {r.tooth: r for r in extract_features(image_path, xml_path)}

    fig = plt.figure(figsize=(15, 8.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1, 1, 1.1])

    ax = fig.add_subplot(grid[0, :])
    ax.imshow(image, cmap="gray")
    draw(ax, polygons, COLORS)
    ax.set_title("Annotated radiograph")
    ax.axis("off")

    for column, (tooth, canal) in enumerate(PAIRS):
        x0, x1, y0, y1 = crop_bounds(polygons, (tooth, canal), image.shape)
        ax = fig.add_subplot(grid[1, column])
        ax.imshow(image[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0))
        draw(ax, polygons, (tooth, canal), bounds=True)
        ax.set_title(f"Tooth {tooth} and canal")
        ax.axis("off")

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    header = f"{'':<20}{'48':>8}{'38':>8}"
    lines = [
        f"{name:<20}" + "".join(f"{rows[t].values[name]:>8.3f}" for t in ("48", "38"))
        for name in FEATURES
    ]
    ax.text(0, 0.95, "Extracted features", weight="bold", transform=ax.transAxes)
    ax.text(0, 0.82, "\n".join([header, *lines]), family="monospace", va="top",
            transform=ax.transAxes)  # fmt: skip

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("xml", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("docs/images/overview.png"))
    args = parser.parse_args()

    if args.image and args.xml:
        figure(args.image, args.xml, args.output)
        return
    with tempfile.TemporaryDirectory() as tmp:
        sample, polygons = render(np.random.default_rng(3))
        cv2.imwrite(f"{tmp}/1.jpg", sample)
        write_xml(Path(tmp) / "1.xml", polygons)
        figure(Path(tmp) / "1.jpg", Path(tmp) / "1.xml", args.output)


if __name__ == "__main__":
    main()
