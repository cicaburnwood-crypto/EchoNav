# EchoNav Gen.1

**RGB history → learned bird's-eye-view prediction → interchangeable planar planning.**

EchoNav Gen.1 predicts a local metric map of observed and occluded space from a short RGB sequence. It exposes occupancy, predicted support, an observed-free gate, and confidence to a navigation backend. This repository contains the inference implementation, pretrained BEV-head weights, A* and Dijkstra backends, and tests. It does not contain training infrastructure or robot-control software.

## Planner-independent navigation

**EchoNav Gen.1 can use any interchangeable planar planning algorithm as its navigation backend, provided an adapter satisfies the map, coordinate and path contract.** A* and Dijkstra are included as reference implementations; neither is part of the learned model. Changing the planner does not require retraining or changing the perception weights.

The backend receives a positive-cost 2-D grid, a start cell and a goal cell. It returns a path or an explicit failure. Sampling-based, graph-based or continuous planar planners can be integrated by adapting this representation and validating their collision-checking semantics. Integration does not automatically provide kinematic feasibility, dynamic-obstacle handling or safety guarantees.

## Installation

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build for your machine, then run from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch build using https://pytorch.org/get-started/locally/
python -m pip install -e '.[test]'
pytest -q
```

The reference CUDA path uses BF16 computation with FP32 head parameters. A CPU FP32 path is available but substantially slower and is not promised to be bitwise equal to CUDA. The model never selects an idle GPU, terminates processes or allocates a multi-GPU job on its own. Choose an available device explicitly.

The default batches up to nine history proposals together. On smaller GPUs, use `--history-batch-size 1` (Python: `history_batch_size=1`) to process them serially without dropping frames, changing weights or reducing resolution. This reduces peak activation memory at the cost of speed; minor floating-point differences between batching modes are possible. The default ten-frame path can exceed 8 GB VRAM. No automatic precision downgrade is performed after an out-of-memory error.

### Weights

1. Download `echonav_gen1.pt` from this repository's **v1.0.0 GitHub Release** into `weights/`. It contains only the inference head, model name and format identifier. No optimizer, data manifest, source paths or training metadata are included.
2. Obtain the **VGGT-Omega-1B-512** backbone from the [official model distribution](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt), following its access and license requirements. Save it as `weights/backbone.pt`. The backbone is not redistributed here.
3. Verify both files before inference:

```bash
mkdir -p weights
# Optional: GitHub CLI, run inside this cloned repository.
gh release download v1.0.0 --pattern echonav_gen1.pt --dir weights
python scripts/verify_weights.py --head weights/echonav_gen1.pt --backbone weights/backbone.pt
```

| File | SHA-256 |
|---|---|
| `echonav_gen1.pt` | `a7f5dbda0843fe2bbf562f0a2daea5eeb4969290e2867ee1eeb0c5681d0fed60` |
| `backbone.pt` | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |

Only load trusted checkpoints. The implementation uses PyTorch's restricted `weights_only=True` loader and strict parameter loading. An arbitrary checkpoint from a different model family is not compatible.

## Inference

Provide **1–10 RGB frames, explicitly ordered oldest to latest**. The intended history is up to approximately five seconds at 2 Hz. Frames should belong to the same camera trajectory. The final frame defines the output coordinate system. No ground-truth depth, camera intrinsics/extrinsics, camera height or scale value is an inference input.

```bash
echonav infer \
  --frames rgb/frame_000.png rgb/frame_001.png rgb/frame_002.png \
  --head weights/echonav_gen1.pt \
  --backbone weights/backbone.pt \
  --device cuda:0 \
  --output outputs/prediction
```

RGB is resized with preserved aspect ratio and white padding to **384×512**, normalized to `[0,1]`. Internally the sequence is reversed once at the frozen backbone boundary so the latest frame receives the first-frame camera/register token role. Do not reverse input files yourself.

Outputs:

- `prediction.npz`: native 256×256 probabilities, confidence and categorical BEV.
- `bev.png`: black = occupied (`0`), gray = unknown/unsupported (`112`), white = free (`255`).
- `metadata.json`: public model contract and frame count; no input filenames or machine identifiers.

Output directories must be new. Existing results are never silently overwritten.

### Output semantics

| Array | Meaning |
|---|---|
| `occupancy_probability` | Fused predicted occupancy probability |
| `support_probability` | Predicted spatial support; not a GT visibility mask |
| `observed_gate_probability` | Predicted **observed-free** gate; not a mask of all visible surfaces |
| `completion_occupancy_probability` | Occupancy from the completion expert before observed-free fusion |
| `confidence` | Model-derived navigation confidence; **not calibrated collision probability** |
| `semantic` | Support-thresholded occupied/free map with unknown outside predicted support |

Let the observed-free gate be `g`, support be `s`, and completion occupancy be `p`. The fused occupancy is `(1-g)·p`. The completion expert predicts positive Beta evidence; navigation confidence multiplies support, classification certainty and a mixture-variance-derived confidence. See [the probability implementation](src/echonav/model/probability.py) for the exact equations.

## Planning

Goal and start coordinates are `(x,z)` metres in the **latest camera's local BEV frame**: x right, z forward. The square spans 10×10 metres around the ego origin; forward is up in the image. Cell size is `10/256 = 0.0390625 m`. Array coordinates are `(row,column)`, not `(x,z)`.

```bash
echonav plan --prediction outputs/prediction/prediction.npz \
  --goal 0.0 2.0 --planner astar --robot-radius 0.10 \
  --output outputs/astar

echonav plan --prediction outputs/prediction/prediction.npz \
  --goal 0.0 2.0 --planner dijkstra --robot-radius 0.10 \
  --output outputs/dijkstra
```

Planning is CPU-only and does not load model weights. Each result includes `plan.json` with both grid cells and metric waypoints, and `path.png`. A blocked or unreachable request returns an explicit status and exit code 2. The planner does not clear the start cell, snap goals, fill unknown areas, or choose goals from GT. For destinations outside the local map, the application must supply an in-bounds subgoal.

### Cost and collision policy

- Occupancy ≥ 0.5 or support < 0.5 is blocked. Unknown is **not** treated as free.
- Obstacles, unknown cells and the map boundary are inflated by the requested robot radius, rounded up to a whole grid cell.
- On traversable cells, `q = clip((1-occupancy)·confidence, 1e-4, 1)` and `cost = 1 - weight·log(q)`. The default confidence weight is 1.0; it can be set to zero.
- Both backends use eight-connected movement without diagonal corner cutting. Edge cost is metric step length times the mean endpoint cost.
- A* uses an admissible octile heuristic scaled by the minimum finite map cost; Dijkstra uses zero heuristic. Both optimize the same cost, not necessarily the shortest geometric path.

These are transparent reference navigation policies, not learned parameters. Set the footprint and margins for your platform and independently validate safety. Geometry inflation cannot correct a missed or hallucinated obstacle.

### Swap in your own backend

```python
from echonav.planning import PlanResult, register_planner, make_planner

class MyPlanarPlanner:
    def plan(self, grid, start, goal):
        # Adapt grid.costs (+inf = blocked) and grid.resolution_m to your solver.
        # Return collision-checked (row, column) cells, or a failure status.
        return PlanResult(status="unreachable")

register_planner("my_planner", MyPlanarPlanner)
backend = make_planner("my_planner")
```

The snippet is an interface example, not a functional solver. Custom backends are registered through the Python API; the CLI intentionally exposes only the two bundled planners. Never return `ok` without a valid collision-checked path.

### Python API

```python
from echonav.runtime import EchoNav
from echonav.planning import build_costmap, make_planner

model = EchoNav("weights/echonav_gen1.pt", "weights/backbone.pt", device="cuda:0")
prediction = model.predict(["rgb/frame_000.png", "rgb/frame_001.png"])
grid = build_costmap(prediction["occupancy_probability"],
                     prediction["support_probability"], prediction["confidence"])
plan = make_planner("astar").plan(grid, grid.metric_to_cell(0, 0),
                                 grid.metric_to_cell(0, 2))
```

## Model architecture

| Component | EchoNav Gen.1 |
|---|---|
| Backbone | Frozen VGGT-Omega aggregation trunk, one pass per RGB window |
| Intermediate features | Layers 4, 11, 17, 23; 2048-dimensional local/global token pairs |
| Patch fusion | Separate 48-channel local and global streams, DPT-lite top-down fusion, 96-channel query features |
| Prefix context | One camera token plus 16 register tokens per frame; role-separated 8-layer, width-1024 context trunk |
| Latest-frame reasoning | Dense metric cell queries, three deformable-attention decoder layers |
| History | Two-layer shared proposals with per-cell temporal attention and a null-history option |
| Spatial output | Shared refinement, separate routing and evidential heads; 256² native output |
| Learned head | 113,786,432 parameters; 607 state tensors |

The inference release excludes unused scale modules and backbone geometry prediction heads. It neither predicts nor consumes an explicit geometry conditioning signal. No model retraining was performed for packaging.

## Reproducibility and limitations

See [VALIDATION.md](VALIDATION.md) for release checks. Unit tests cover planner equivalence, blocked endpoints, corner cutting, map boundaries, coordinates, invalid probabilities, preprocessing and head shape contracts. Unit tests are not a navigation benchmark.

Occluded-space predictions are hypotheses. Domain shifts, thin obstacles, glass, moving objects, camera changes, slopes and stairs can invalidate them. The output is a local 2-D map, not persistent SLAM, localization, global planning, control or certified collision avoidance. External applications must provide goals, localization where needed, feedback control and independent safety checks. Do not deploy these research outputs as the sole safety mechanism.

## Licensing and acknowledgements

This is a **research-code release**, not an unrestricted commercial-use grant. The bundled VGGT-Omega-derived code is subject to the [FAIR Noncommercial Research License](LICENSE). Files derived from DINOv3 retain their copyright notices and the [DINOv3 license](licenses/DINOv3.txt). See [THIRD_PARTY.md](THIRD_PARTY.md). Obtain the external backbone through its official distribution and follow its terms. No dataset images, datasets, robot logs or private infrastructure are distributed.

The public model name is **EchoNav Gen.1**. Paper bibliographic metadata is not fabricated here; add the official citation when the manuscript reference is finalized.
