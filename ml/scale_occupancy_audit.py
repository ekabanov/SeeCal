"""Audit SCALE mass support and source-specific visual occupancy response.

The three datasets expose different annotations, so occupancy values are only
comparable *within* a source:

* Nutrition5K: calibrated depth-derived food footprint / valid central footprint
* NutritionVerse: summed COCO food-segmentation area / image area
* FPB: union of YOLO food bounding boxes / image area

The audit reports both record-level and equal-group correlations between
occupancy, ground-truth mass, and Probe B P50. It also compares unique-group
mass support in the training manifest with frozen FPB test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr

import depth_features


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_eval(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["paired"]}


def _correlation(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    if len(x) != len(y):
        raise ValueError("correlation inputs must have equal length")
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(x), "pearson_r": None, "spearman_rho": None}
    return {
        "n": len(x),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }


def _rectangle_union_area(
    rectangles: Iterable[tuple[float, float, float, float]],
) -> float:
    """Return exact union area for axis-aligned normalized rectangles."""
    rects = [
        (
            max(0.0, min(1.0, x0)),
            max(0.0, min(1.0, y0)),
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
        )
        for x0, y0, x1, y1 in rectangles
        if x1 > x0 and y1 > y0
    ]
    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (y0, y1)
            for x0, y0, x1, y1 in rects
            if x0 < right and x1 > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered += end - start
        area += (right - left) * covered
    return area


def _transform_point(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    mode: str,
    image_size: int = 224,
    resize_short_side: int = 232,
) -> tuple[float, float]:
    if mode == "center_crop":
        scale = resize_short_side / min(width, height)
        offset_x = (width * scale - image_size) / 2
        offset_y = (height * scale - image_size) / 2
    elif mode == "letterbox":
        scale = image_size / max(width, height)
        offset_x = -(image_size - width * scale) / 2
        offset_y = -(image_size - height * scale) / 2
    else:
        raise ValueError(f"unknown geometry mode: {mode}")
    return (x * scale - offset_x, y * scale - offset_y)


def _mask_fraction(
    polygons: list[list[float]],
    *,
    width: int,
    height: int,
    mode: str,
    image_size: int = 224,
) -> float:
    mask = Image.new("1", (image_size, image_size))
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [
            _transform_point(
                polygon[index],
                polygon[index + 1],
                width=width,
                height=height,
                mode=mode,
                image_size=image_size,
            )
            for index in range(0, len(polygon), 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return float(np.asarray(mask, dtype=np.uint8).mean())


def _transform_binary_mask(
    mask: np.ndarray,
    *,
    mode: str,
    image_size: int = 224,
    resize_short_side: int = 232,
) -> float:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    width, height = image.size
    if mode == "center_crop":
        scale = resize_short_side / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.NEAREST,
        )
        left = round((resized.width - image_size) / 2)
        top = round((resized.height - image_size) / 2)
        output = resized.crop((left, top, left + image_size, top + image_size))
    elif mode == "letterbox":
        scale = image_size / max(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.NEAREST,
        )
        output = Image.new("L", (image_size, image_size))
        output.paste(
            resized,
            (
                round((image_size - resized.width) / 2),
                round((image_size - resized.height) / 2),
            ),
        )
    else:
        raise ValueError(f"unknown geometry mode: {mode}")
    return float((np.asarray(output) > 0).mean())


def _fpb_occupancy_by_id(
    manifest: list[dict[str, Any]], label_dir: Path
) -> dict[str, dict[str, float]]:
    result = {
        "raw_annotation": {},
        "center_crop_224": {},
        "letterbox_224": {},
    }
    for row in manifest:
        stem = str(row["id"]).split("fpb:test:", 1)[-1]
        label_path = label_dir / f"{stem}.txt"
        with Image.open(row["image_path"]) as image:
            width, height = image.size
        rectangles = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            _, x, y, box_width, box_height, _weight = fields[:6]
            cx, cy, w, h = map(float, (x, y, box_width, box_height))
            rectangles.append(
                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            )
        record_id = str(row["id"])
        result["raw_annotation"][record_id] = _rectangle_union_area(rectangles)
        for mode, key in (
            ("center_crop", "center_crop_224"),
            ("letterbox", "letterbox_224"),
        ):
            transformed = []
            for x0, y0, x1, y1 in rectangles:
                left, top = _transform_point(
                    x0 * width,
                    y0 * height,
                    width=width,
                    height=height,
                    mode=mode,
                )
                right, bottom = _transform_point(
                    x1 * width,
                    y1 * height,
                    width=width,
                    height=height,
                    mode=mode,
                )
                transformed.append(
                    (left / 224, top / 224, right / 224, bottom / 224)
                )
            result[key][record_id] = _rectangle_union_area(transformed)
    return result


def _nutritionverse_occupancy_by_id(
    manifest: list[dict[str, Any]], annotations_path: Path
) -> dict[str, dict[str, float]]:
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    area_by_image: dict[int, float] = defaultdict(float)
    polygons_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        area_by_image[int(annotation["image_id"])] += float(annotation["area"])
        polygons_by_image[int(annotation["image_id"])].extend(
            polygon
            for polygon in annotation.get("segmentation", [])
            if isinstance(polygon, list)
        )
    occupancy_by_stem: dict[str, dict[str, float]] = {}
    for image_id, image in images.items():
        denominator = float(image["width"]) * float(image["height"])
        occupancy_by_stem[Path(image["file_name"]).stem] = {
            "raw_annotation": min(1.0, area_by_image[image_id] / denominator),
            "center_crop_224": _mask_fraction(
                polygons_by_image[image_id],
                width=int(image["width"]),
                height=int(image["height"]),
                mode="center_crop",
            ),
            "letterbox_224": _mask_fraction(
                polygons_by_image[image_id],
                width=int(image["width"]),
                height=int(image["height"]),
                mode="letterbox",
            ),
        }
    result: dict[str, dict[str, float]] = {
        "raw_annotation": {},
        "center_crop_224": {},
        "letterbox_224": {},
    }
    for row in manifest:
        stem = str(row["id"]).split("nv-real:", 1)[-1]
        if stem in occupancy_by_stem:
            for key, value in occupancy_by_stem[stem].items():
                result[key][str(row["id"])] = value
    return result


def _nutrition5k_occupancy_by_id(
    manifest: list[dict[str, Any]], overhead_root: Path
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    result: dict[str, dict[str, float]] = {
        "raw_annotation": {},
        "center_crop_224": {},
        "letterbox_224": {},
    }
    skipped = []
    for row in manifest:
        group_id = str(row["group_id"])
        depth_path = overhead_root / group_id / "depth_raw.png"
        try:
            depth = depth_features.load_depth(depth_path)
            plane = depth_features.plane_fit(depth)
            rows, cols, z, height = depth_features._height_above_plane(depth, plane)
            central = depth_features._central_disk_mask(depth.shape)[rows, cols]
            footprint = (z / depth_features.FOCAL_PX) ** 2
            food = (
                central
                & (height > depth_features.HEIGHT_NOISE_FLOOR_M)
                & (height <= depth_features.HEIGHT_FOOD_MAX_M)
            )
            denominator = float(np.sum(footprint[central]))
            if denominator <= 0:
                raise ValueError("no valid central footprint")
            record_id = str(row["id"])
            result["raw_annotation"][record_id] = float(
                np.sum(footprint[food]) / denominator
            )
            food_image = np.zeros(depth.shape, dtype=bool)
            food_image[rows[food], cols[food]] = True
            result["center_crop_224"][record_id] = _transform_binary_mask(
                food_image, mode="center_crop"
            )
            result["letterbox_224"][record_id] = _transform_binary_mask(
                food_image, mode="letterbox"
            )
        except Exception as error:
            skipped.append({"id": str(row["id"]), "error": str(error)})
    return result, skipped


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    occupancy = [float(row["occupancy"]) for row in rows]
    truth = [float(row["truth_mass_g"]) for row in rows]
    prediction = [float(row["prediction_mass_g"]) for row in rows]
    return {
        "records": len(rows),
        "truth_vs_occupancy": _correlation(occupancy, truth),
        "prediction_vs_occupancy": _correlation(occupancy, prediction),
        "truth_vs_prediction": _correlation(truth, prediction),
        "occupancy": _distribution(occupancy),
        "truth_mass_g": _distribution(truth),
        "prediction_mass_g": _distribution(prediction),
    }


def _equal_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return [
        {
            "group_id": group_id,
            "occupancy": statistics.fmean(
                float(row["occupancy"]) for row in group_rows
            ),
            "truth_mass_g": statistics.fmean(
                float(row["truth_mass_g"]) for row in group_rows
            ),
            "prediction_mass_g": statistics.fmean(
                float(row["prediction_mass_g"]) for row in group_rows
            ),
        }
        for group_id, group_rows in sorted(grouped.items())
    ]


def _distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p10": quantile(0.1),
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def _mass_support(
    training_manifest: list[dict[str, Any]],
    fpb_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in training_manifest:
        grouped[(str(row["source"]), str(row["group_id"]))].append(
            float(row["total_mass_g"])
        )
    by_source: dict[str, list[float]] = defaultdict(list)
    for (source, _group_id), masses in grouped.items():
        by_source[source].append(statistics.median(masses))
    fpb_groups: dict[str, list[float]] = defaultdict(list)
    for row in fpb_manifest:
        fpb_groups[str(row["group_id"])].append(float(row["total_mass_g"]))
    fpb = [statistics.median(values) for values in fpb_groups.values()]
    result = {
        "training_unique_groups_by_source": {
            source: _distribution(values)
            for source, values in sorted(by_source.items())
        },
        "fpb_test_unique_groups": _distribution(fpb),
    }
    for source, values in sorted(by_source.items()):
        training = sorted(values)
        p90 = _distribution(training)["p90"]
        maximum = max(training)
        result[f"fpb_fraction_above_{source}_train_p90"] = sum(
            value > p90 for value in fpb
        ) / len(fpb)
        result[f"fpb_fraction_above_{source}_train_max"] = sum(
            value > maximum for value in fpb
        ) / len(fpb)
    return result


def _join(
    manifest: list[dict[str, Any]],
    evaluation: dict[str, dict[str, Any]],
    occupancy: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for source in manifest:
        record_id = str(source["id"])
        if record_id not in evaluation or record_id not in occupancy:
            continue
        prediction = evaluation[record_id]
        rows.append(
            {
                "id": record_id,
                "group_id": str(source["group_id"]),
                "occupancy": occupancy[record_id],
                "truth_mass_g": float(source["total_mass_g"]),
                "prediction_mass_g": float(prediction["p50_g"]),
            }
        )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "nutrition5k": _read_jsonl(args.nutrition5k_manifest),
        "nutritionverse": _read_jsonl(args.nutritionverse_manifest),
        "fpb": _read_jsonl(args.fpb_manifest),
    }
    evaluations = {
        "nutrition5k": _read_eval(args.nutrition5k_evaluation),
        "nutritionverse": _read_eval(args.nutritionverse_evaluation),
        "fpb": _read_eval(args.fpb_evaluation),
    }
    n5k_occupancy, n5k_skipped = _nutrition5k_occupancy_by_id(
        manifests["nutrition5k"], args.nutrition5k_overhead_root
    )
    occupancy_variants = {
        "nutrition5k": n5k_occupancy,
        "nutritionverse": _nutritionverse_occupancy_by_id(
            manifests["nutritionverse"], args.nutritionverse_annotations
        ),
        "fpb": _fpb_occupancy_by_id(manifests["fpb"], args.fpb_label_dir),
    }
    domains = {}
    for domain in ("nutrition5k", "nutritionverse", "fpb"):
        domains[domain] = {
            "occupancy_proxy": {
                "nutrition5k": "depth_food_footprint_fraction",
                "nutritionverse": "coco_segmentation_area_fraction",
                "fpb": "yolo_bbox_union_area_fraction",
            }[domain],
            "variants": {},
        }
        for variant, occupancy in occupancy_variants[domain].items():
            joined = _join(
                manifests[domain], evaluations[domain], occupancy
            )
            groups = _equal_group(joined)
            domains[domain]["variants"][variant] = {
                "record_level": _summarize_rows(joined),
                "equal_group": _summarize_rows(groups),
            }
    output = {
        "schema_version": 1,
        "warning": (
            "Occupancy proxies differ by source and must not be compared "
            "as if they were the same measurement."
        ),
        "domains": domains,
        "mass_support": _mass_support(
            _read_jsonl(args.training_manifest), manifests["fpb"]
        ),
        "nutrition5k_skipped": n5k_skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("datasets/scale_v2_nc_1024/train.jsonl"),
    )
    parser.add_argument(
        "--nutrition5k-manifest",
        type=Path,
        default=Path("datasets/scale_v2_nc_1024/test-nutrition5k.jsonl"),
    )
    parser.add_argument(
        "--nutritionverse-manifest",
        type=Path,
        default=Path("datasets/scale_v2_nc_1024/test-nutritionverse-real.jsonl"),
    )
    parser.add_argument(
        "--fpb-manifest",
        type=Path,
        default=Path("datasets/food-portion-benchmark/scale-zero-shot/test.jsonl"),
    )
    parser.add_argument(
        "--nutrition5k-evaluation",
        type=Path,
        default=Path(
            "runs/factored/scale-v2-probe-b-nv-1024/"
            "eval-nutrition5k-overhead-calibrated.json"
        ),
    )
    parser.add_argument(
        "--nutritionverse-evaluation",
        type=Path,
        default=Path(
            "runs/factored/scale-v2-probe-b-nv-1024/"
            "eval-nutritionverse-real-calibrated.json"
        ),
    )
    parser.add_argument(
        "--fpb-evaluation",
        type=Path,
        default=Path(
            "runs/factored/scale-v2-probe-b-nv-1024/"
            "eval-fpb-test-zero-shot.json"
        ),
    )
    parser.add_argument(
        "--nutrition5k-overhead-root",
        type=Path,
        default=Path("Nutrition5K/imagery/realsense_overhead"),
    )
    parser.add_argument(
        "--nutritionverse-annotations",
        type=Path,
        default=Path(
            "datasets/nutritionverse-real/extracted/nutritionverse-manual/"
            "nutritionverse-manual/images/_annotations.coco.json"
        ),
    )
    parser.add_argument(
        "--fpb-label-dir",
        type=Path,
        default=Path(
            "/Users/jevgenikabanov/.cache/huggingface/hub/"
            "datasets--issai--Food_Portion_Benchmark/snapshots/"
            "53fcacf4b9dbe24c1c6ffa5a2cdb9d8c502e482f/"
            "FPB_Dataset/RGB/test/labels"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/factored/scale-v2-probe-b-nv-1024/occupancy-audit.json"
        ),
    )
    args = parser.parse_args()
    result = build(args)
    summary = {
        domain: {
            variant: {
                "truth_vs_occupancy": metrics["equal_group"][
                    "truth_vs_occupancy"
                ],
                "prediction_vs_occupancy": metrics["equal_group"][
                    "prediction_vs_occupancy"
                ],
            }
            for variant, metrics in values["variants"].items()
        }
        for domain, values in result["domains"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
