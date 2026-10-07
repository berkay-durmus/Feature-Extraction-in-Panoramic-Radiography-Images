"""Render the overview figure used in the README from a synthetic sample."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_dummy_dataset import render  # noqa: E402

from panoramic_features.features.roots import _split_roots  # noqa: E402
from panoramic_features.pipeline import (
    FEATURES,  # noqa: E402
    extract_features,  # noqa: E402
)

COLORS = {"48": "#e4572e", "Sağ M3": "#17bebb", "38": "#f3a712", "Sol M3": "#76b041"}


def main(out: Path) -> None:
    rng = np.random.default_rng(3)
    image, polygons = render(rng)

    work = out.parent / "_sample"
    work.mkdir(exist_ok=True)
    from make_dummy_dataset import write_xml

    cv2.imwrite(str(work / "1.jpg"), image)
    write_xml(work / "1.xml", polygons)
    row = extract_features(work / "1.jpg", work / "1.xml")[0]

    mask = np.zeros(image.shape, np.uint8)
    cv2.fillPoly(mask, [polygons["48"]], 1)
    roots = _split_roots(mask.astype(bool), "right")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.4, 1, 1.1]})
    ax = axes[0]
    ax.imshow(image, cmap="gray")
    for label, points in polygons.items():
        ax.add_patch(plt.Polygon(points, fill=False, ec=COLORS[label], lw=2))
        ax.text(*points.mean(axis=0) + (0, -150), label, color=COLORS[label], ha="center")
    ax.set_title("Annotated radiograph (synthetic)")

    ax = axes[1]
    canvas = np.zeros((max(r.shape[0] for r in roots), sum(r.shape[1] + 6 for r in roots), 3))
    x = 0
    for color, root in zip(("#e4572e", "#17bebb"), roots, strict=False):
        canvas[: root.shape[0], x : x + root.shape[1]][root] = plt.matplotlib.colors.to_rgb(color)
        x += root.shape[1] + 6
    ax.imshow(canvas)
    ax.set_title("Tooth 48, roots separated")

    ax = axes[2]
    ax.axis("off")
    lines = [f"{name:<20}{row.values[name]:>8.3f}" for name in FEATURES]
    ax.text(0, 0.8, "Features, tooth 48", weight="bold", transform=ax.transAxes)
    ax.text(0, 0.7, "\n".join(lines), family="monospace", va="top", transform=ax.transAxes)

    for a in axes[:2]:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images/overview.png"))
