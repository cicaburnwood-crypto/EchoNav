from PIL import Image
import pytest
import torch
from echonav.config import HEAD_ARGUMENTS
from echonav.model.head import BEVHead, assemble_bev
from echonav.preprocess import RGBResizePad


def test_preprocessing():
    tensor = RGBResizePad()(Image.new("RGB", (400, 100), "black"))
    assert tensor.shape == (3, 384, 512)
    assert tensor[:, 0, :].eq(1).all()
    assert tensor[:, 192, :].eq(0).all()


def test_output_semantics():
    out = assemble_bev(
        torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), assemble_runtime_outputs=True
    )
    assert out["occupancy_probability"].shape == (1, 8, 8)
    assert torch.allclose(out["occupancy_probability"], torch.full((1, 8, 8), 0.25))
    assert set(out["fov_complete_semantic"].unique().tolist()) == {255}
    assert torch.isfinite(out["navigation_confidence"]).all()


@pytest.mark.parametrize("frames", [1, 3, 10])
def test_head_shapes_without_weights(frames):
    # Small architectural fixture, not a pretrained-quality or GPU test.
    config = {
        **HEAD_ARGUMENTS,
        "merged_bev_size": 8,
        "vggt_token_dim": 32,
        "prefix_context_hidden_dim": 32,
        "prefix_context_heads": 4,
        "prefix_context_layers": 2,
        "hidden_dim": 16,
        "patch_stream_dim": 8,
        "query_content_dim": 8,
        "heads": 4,
        "frame_reliability_hidden_dim": 16,
    }
    torch.manual_seed(7)
    head = BEVHead(**config).eval()
    extraction = {
        "tokens": {i: torch.randn(1, frames, 16, 32) for i in config["cached_layers"]},
        "camera_register_tokens": torch.randn(1, frames, 17, 32),
        "patch_grid": (4, 4),
    }
    with torch.no_grad():
        out = head(extraction)["merged_bev"]
    assert out["occupancy_probability"].shape == (1, 8, 8)
    assert torch.isfinite(out["navigation_confidence"]).all()
    assert set(out["fov_complete_semantic"].unique().tolist()) <= {0, 112, 255}
    assert not any(k.startswith("scale_") for k in head.state_dict())
