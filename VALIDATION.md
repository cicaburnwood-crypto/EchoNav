# EchoNav Gen.1 release validation

## What was verified

- All 607 retained inference-head tensors match the accepted source weights exactly. Packaging removes unused scale modules but does not retrain, quantize or alter retained values.
- CPU head comparison against the accepted implementation: bitwise-identical occupancy, support, observed-free gate, confidence and categorical output for 1, 3 and 10 frames on a small-query synthetic fixture with full-width trained modules.
- End-to-end CUDA comparison at the native 384×512 RGB input and 256×256 BEV output: bitwise-identical versions of those five output maps for 1, 3 and 10 frames. This includes preprocessing, latest-first backbone anchoring, frozen backbone aggregation and BEV decoding.
- The ten-frame end-to-end fixture uses three chronological real images with the oldest image repeated seven times; it is an implementation-equivalence test, **not** a ten-frame accuracy benchmark. History proposal batch size was 1 for this test. Default batching was separately checked for 1 and 3 frames.
- A* and Dijkstra returned the same path cost on a saved real-image prediction. Random weighted-grid tests also compare optimal costs and failure states.
- 22 CPU unit tests passed. Source lint and wheel packaging passed.
- All tracked source/doc filenames and contents, and non-tensor checkpoint metadata, were scanned for retired development labels, private host/account references, absolute personal paths, IP addresses and recognizable key/token patterns. None were found. No scan is a universal guarantee against every possible secret format.

## Validation environment

Python 3.11, PyTorch 2.10.0 with CUDA 13.0, NumPy 1.26.4, Pillow 11.3.0, pytest 9.1.1. Native comparison used an NVIDIA GeForce RTX 5070 Laptop GPU. No machine hostname, device UUID, user identity or private input path is distributed.

The default ten-frame parallel-history execution exceeded this GPU's available memory. Serial history proposals completed successfully without dropping frames. Other hardware and PyTorch/CUDA builds can have different memory requirements and numerical results. CPU FP32 is supported but has not been established as bitwise equal to BF16 CUDA inference.

## Re-run available checks

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/verify_weights.py --head weights/echonav_gen1.pt --backbone weights/backbone.pt
```

The repository tests are self-contained and do not require model downloads or GPU access. Native inference requires the released head and separately obtained backbone. The private source implementation and third-party dataset frames used for the equivalence audit are intentionally not distributed; the native equivalence result is a release audit, not a self-contained public accuracy evaluation.

## Scope

These checks establish implementation consistency and basic planner correctness, not reconstruction accuracy, calibration, closed-loop navigation performance or safe deployment. No benchmark scores or paper claims are inferred from these tests.
