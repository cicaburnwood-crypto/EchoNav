"""File-based RGB inference and local planning; no robot actuation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import MODEL_NAME
from .planning import build_costmap, make_planner


def main() -> None:
    parser = argparse.ArgumentParser(description=MODEL_NAME)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser(
        "infer", help="Predict BEV from chronological RGB files"
    )
    infer.add_argument("--frames", nargs="+", required=True, type=Path)
    infer.add_argument("--head", required=True, type=Path)
    infer.add_argument("--backbone", required=True, type=Path)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument(
        "--history-batch-size", type=int, default=9, choices=range(1, 10)
    )
    infer.add_argument("--output", required=True, type=Path)
    plan = commands.add_parser(
        "plan", help="Plan on saved predictions without loading the model"
    )
    plan.add_argument("--prediction", required=True, type=Path)
    plan.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Z"))
    plan.add_argument(
        "--start", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Z")
    )
    plan.add_argument("--planner", choices=("astar", "dijkstra"), default="astar")
    plan.add_argument("--robot-radius", type=float, default=0.10)
    plan.add_argument("--confidence-weight", type=float, default=1.0)
    plan.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    if args.command == "infer":
        if not 1 <= len(args.frames) <= 10:
            parser.error("Provide 1–10 RGB files in oldest-to-latest order")
        from .runtime import EchoNav

        model = EchoNav(
            args.head,
            args.backbone,
            device=args.device,
            history_batch_size=args.history_batch_size,
        )
        prediction = model.predict(args.frames)
        if any(not np.isfinite(v).all() for v in prediction.values()):
            raise RuntimeError("Model returned a nonfinite prediction")
        args.output.mkdir(parents=True)
        np.savez_compressed(
            args.output / "prediction.npz",
            **prediction,
            model_name=np.asarray(MODEL_NAME),
            extent_m=np.asarray(10.0),
        )
        Image.fromarray(prediction["semantic"]).save(args.output / "bev.png")
        # No absolute filenames, user names or device identifiers in metadata.
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "frames": len(args.frames),
                    "input_order": "oldest_to_latest",
                    "output_size": 256,
                    "extent_m": 10.0,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"{MODEL_NAME}: wrote prediction.npz and bev.png")
    else:
        with np.load(args.prediction, allow_pickle=False) as p:
            if (
                str(p["model_name"].item()) != MODEL_NAME
                or float(p["extent_m"]) != 10.0
            ):
                raise ValueError("Expected an EchoNav Gen.1 fixed-metric prediction")
            if p["occupancy_probability"].shape != (256, 256):
                raise ValueError("Expected native 256×256 prediction")
            grid = build_costmap(
                p["occupancy_probability"],
                p["support_probability"],
                p["confidence"],
                robot_radius_m=args.robot_radius,
                confidence_weight=args.confidence_weight,
            )
            semantic = np.array(p["semantic"], copy=True)
        start, goal = grid.metric_to_cell(*args.start), grid.metric_to_cell(*args.goal)
        result = make_planner(args.planner).plan(grid, start, goal)
        record = asdict(result)
        record.update(
            model=MODEL_NAME,
            planner=args.planner,
            path_xz_m=[grid.cell_to_metric(c) for c in result.cells],
            goal_xz_m=args.goal,
            start_xz_m=args.start,
            robot_radius_m=args.robot_radius,
            confidence_weight=args.confidence_weight,
        )
        args.output.mkdir(parents=True)
        (args.output / "plan.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        canvas = Image.fromarray(semantic).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        if len(result.cells) > 1:
            draw.line([(c, r) for r, c in result.cells], fill=(20, 120, 220), width=2)
        for (r, c), color in ((start, (0, 170, 90)), (goal, (210, 40, 40))):
            draw.ellipse((c - 2, r - 2, c + 2, r + 2), fill=color)
        canvas.resize((768, 768), Image.Resampling.NEAREST).save(
            args.output / "path.png"
        )
        print(
            json.dumps(
                {"status": result.status, "planner": args.planner, "cost": result.cost}
            )
        )
        if result.status != "ok":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
