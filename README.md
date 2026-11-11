# EchoNav Gen.1

[Project Website](https://cicaburnwood-crypto.github.io/EchoNav/) · [Paper (arXiv)](https://arxiv.org/abs/2610.27616)

## Installation

Use Python 3.10 or newer and install a suitable [PyTorch build](https://pytorch.org/get-started/locally/) for your machine.

```bash
git clone https://github.com/cicaburnwood-crypto/EchoNav.git
cd EchoNav
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For inference, download [echonav_gen1.pt](https://huggingface.co/Wolfie-Jr/EchoNav/resolve/main/echonav_gen1.pt?download=true) from [Hugging Face](https://huggingface.co/Wolfie-Jr/EchoNav) and save it as `weights/echonav_gen1.pt`. Download the [VGGT-Omega-1B-512 backbone](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) and save it as `weights/backbone.pt`.

## Interchangeable planar navigation backends

EchoNav supports interchangeable planar planning algorithms. A* and Dijkstra are included; other planners can be integrated through an adapter that accepts the 2-D cost grid, start and goal, and returns a valid path or an explicit failure. Changing the planner does not require retraining or modifying the perception weights.

Custom backends can be registered through the Python API with `register_planner` and selected with `make_planner` in `echonav.planning`.
