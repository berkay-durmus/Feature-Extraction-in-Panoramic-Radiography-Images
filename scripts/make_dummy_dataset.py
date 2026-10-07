"""Generate a synthetic annotated dataset (images + XML) for trying the tool.

python scripts/make_dummy_dataset.py --out demo-data --count 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SHAPE = (600, 1200)
CROWN_HALF_WIDTH, CROWN_HEIGHT, ROOT_LENGTH = 50, 80, 200


def tooth_outline(bend_a: int, bend_b: int) -> np.ndarray:
    """Crown plus two roots in local coordinates; the bends shift the root tips sideways."""
    return np.array(
        [
            (-50, 0), (50, 0), (50, 80), (45, 100),
            (45 + bend_b, ROOT_LENGTH), (10 + bend_b, ROOT_LENGTH),
            (8, 110), (-8, 110),
            (-10 + bend_a, ROOT_LENGTH), (-45 + bend_a, ROOT_LENGTH),
            (-45, 100), (-50, 80),
        ],
        dtype=float,
    )  # fmt: skip


def place(points: np.ndarray, origin: tuple[float, float], angle: float) -> np.ndarray:
    rad = np.radians(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return np.rint(points @ rot.T + origin).astype(np.int32)


def canal_outline(x0: int, x1: int, y: int, width: int, wave: float) -> np.ndarray:
    xs = np.arange(x0, x1 + 1, 10)
    centre = y + wave * np.sin(xs / 40.0)
    upper = np.column_stack([xs, centre - width / 2])
    lower = np.column_stack([xs, centre + width / 2])[::-1]
    return np.rint(np.vstack([upper, lower])).astype(np.int32)


def render(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    teeth, canals = {}, {}
    for tooth, canal, cx, mirror in (("48", "Sağ M3", 260, 1), ("38", "Sol M3", 940, -1)):
        bends = [int(rng.integers(-25, 6)), int(rng.integers(-5, 26))]
        outline = tooth_outline(*bends)
        outline[:, 0] *= mirror
        top = (cx, 190 + rng.integers(-15, 16))
        teeth[tooth] = place(outline, top, rng.uniform(-18, 18))
        canals[canal] = canal_outline(
            cx - 190, cx + 190, int(top[1]) + 185, int(rng.integers(26, 40)), rng.uniform(0, 14)
        )

    image = cv2.GaussianBlur(rng.normal(95, 25, SHAPE).astype(np.float32), (0, 0), 6)
    for polygon in teeth.values():
        cv2.fillPoly(image, [polygon], 215)
    darkening = np.ones(SHAPE, np.float32)
    for polygon in canals.values():
        cv2.fillPoly(darkening, [polygon], 0.75)
    image = image * darkening + rng.normal(0, 6, SHAPE)
    return np.clip(image, 0, 255).astype(np.uint8), {**teeth, **canals}


def write_xml(path: Path, polygons: dict[str, np.ndarray]) -> None:
    items = []
    for label, points in polygons.items():
        coords = "".join(f"<x{i}>{x}</x{i}><y{i}>{y}</y{i}>" for i, (x, y) in enumerate(points, 1))
        items.append(f"<item><name>{label}</name><polygon>{coords}</polygon></item>")
    path.write_text(
        f"<annotation><outputs><object>{''.join(items)}</object></outputs></annotation>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("demo-data"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for number in range(1, args.count + 1):
        image, polygons = render(rng)
        cv2.imwrite(str(args.out / f"{number}.jpg"), image)
        write_xml(args.out / f"{number}.xml", polygons)
    print(f"Wrote {args.count} image/annotation pairs to {args.out}")


if __name__ == "__main__":
    main()
