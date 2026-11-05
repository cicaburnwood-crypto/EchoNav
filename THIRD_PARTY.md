# Third-party notices

EchoNav Gen.1 uses an aggregation-only subset of [VGGT-Omega](https://github.com/facebookresearch/vggt-omega). The source under `src/echonav/backbone/` preserves upstream copyright notices. Imports are namespaced under EchoNav; only cached aggregation features are used. The trained camera, depth and text-alignment heads are not included in the runtime model.

The VGGT-Omega-derived material remains subject to the [FAIR Noncommercial Research License](LICENSE). Patch-embedding and transformer-layer files that carry DINOv3 notices also retain the [DINOv3 license](licenses/DINOv3.txt), sourced from [the official project](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). These notices must not be removed when redistributing.

The backbone weights are **not bundled**. Users obtain them from the official VGGT-Omega distribution under its access and license terms. EchoNav's head release does not include dataset files or third-party backbone tensors.

Other runtime dependencies (PyTorch, NumPy and Pillow) are installed separately and retain their respective licenses. No license here replaces or removes an upstream restriction.
