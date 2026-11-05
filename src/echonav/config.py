"""Fixed architecture of the released EchoNav Gen.1 weights."""

MODEL_NAME = "EchoNav Gen.1"
SCHEMA = "echonav-gen1-inference"
HEAD_SHA256 = "a7f5dbda0843fe2bbf562f0a2daea5eeb4969290e2867ee1eeb0c5681d0fed60"
BACKBONE_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"

HEAD_ARGUMENTS = dict(
    cached_layers=(4, 11, 17, 23),
    spatial_scales=(4.0, 2.0, 1.0, 0.5),
    vggt_token_dim=2048,
    hidden_dim=96,
    query_content_dim=64,
    heads=8,
    latest_decoder_layers=3,
    temporal_proposal_layers=2,
    self_attention_mode="linear",
    deformable_samples=4,
    cross_query_chunk_size=65536,
    merged_bev_size=256,
    merged_extent_m=10.0,
    query_fourier_bands=4,
    shared_refinement_layers=2,
    routing_refinement_layers=1,
    evidence_refinement_layers=1,
    patch_stream_dim=48,
    prefix_context_hidden_dim=1024,
    prefix_context_heads=16,
    prefix_context_layers=8,
    maximum_history=10,
    maximum_prefix_tokens=17,
    frame_reliability_hidden_dim=256,
    frame_reliability_minimum=0.25,
    frame_reliability_maximum=1.75,
    temporal_null_initial_probability=0.90,
    history_proposal_batch_size=9,
)
