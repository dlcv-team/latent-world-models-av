"""Tests for attribution visualization pipeline (B7)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from evaluation.gradcam import (
    AttributionPipeline,
    CLIPAttribution,
    DINOv2Attribution,
    EmbeddingL2Target,
    ViTS16Attribution,
    VQVAEAttribution,
    VJEPA2Attribution,
)


@pytest.fixture
def mock_vit_encoder():
    """Mock ViT-S/16 encoder for testing."""
    encoder = MagicMock()
    encoder.backbone = MagicMock()
    encoder.backbone.blocks = [MagicMock() for _ in range(12)]
    encoder.backbone.blocks[-1].norm1 = MagicMock()
    return encoder


@pytest.fixture
def mock_dinov2_encoder():
    """Mock DINOv2 encoder for testing."""
    encoder = MagicMock()
    encoder.backbone = MagicMock()
    encoder.backbone.blocks = [MagicMock() for _ in range(12)]
    encoder.backbone.blocks[-1].attn = MagicMock()
    encoder.backbone.blocks[-1].attn.attn_drop = MagicMock()
    return encoder


@pytest.fixture
def mock_clip_encoder():
    """Mock CLIP encoder for testing."""
    encoder = MagicMock()
    encoder.backbone = MagicMock()
    encoder.backbone.transformer = MagicMock()
    encoder.backbone.transformer.resblocks = [MagicMock() for _ in range(12)]
    encoder.backbone.transformer.resblocks[-1].ln_1 = MagicMock()
    return encoder


@pytest.fixture
def mock_vqvae_encoder():
    """Mock VQ-VAE encoder with fallback active."""
    encoder = MagicMock()
    encoder.fallback_active = True
    encoder.backbone = MagicMock()
    encoder.backbone.blocks = [MagicMock() for _ in range(12)]
    encoder.backbone.blocks[-1].attn = MagicMock()
    encoder.backbone.blocks[-1].attn.attn_drop = MagicMock()
    return encoder


@pytest.fixture
def mock_vjepa_encoder():
    """Mock V-JEPA encoder for testing."""
    encoder = MagicMock()
    encoder.backbone = MagicMock()
    encoder.backbone.encoder = MagicMock()
    encoder.backbone.encoder.layer = [MagicMock() for _ in range(12)]
    encoder.backbone.encoder.layer[-1].attention = MagicMock()
    encoder.backbone.encoder.layer[-1].attention.self = MagicMock()
    encoder._encode = MagicMock(return_value=torch.randn(1, 256, 1024))
    return encoder


class TestAttributionMethods:
    """Test individual attribution method classes."""

    def test_vit_reshape_transform(self, mock_vit_encoder):
        """Test ViT patch-to-spatial reshape."""
        method = ViTS16Attribution(mock_vit_encoder, device="cpu")
        # Simulate ViT output: (B, 197, 384) with CLS token
        tensor = torch.randn(1, 197, 384)
        reshaped = method.reshape_transform(tensor, height=14, width=14)
        # Should be (B, 384, 14, 14)
        assert reshaped.shape == (1, 384, 14, 14)

    def test_clip_reshape_transform(self, mock_clip_encoder):
        """Test CLIP patch-to-spatial reshape."""
        method = CLIPAttribution(mock_clip_encoder, device="cpu")
        # Simulate CLIP ViT-B/32 output: (B, 50, 768) with CLS token
        tensor = torch.randn(1, 50, 768)
        reshaped = method.reshape_transform(tensor, height=7, width=7)
        # Should be (B, 768, 7, 7)
        assert reshaped.shape == (1, 768, 7, 7)

    def test_dinov2_attribution_shape(self, mock_dinov2_encoder):
        """Test DINOv2 attribution output shape."""
        method = DINOv2Attribution(mock_dinov2_encoder, device="cpu")

        # Mock attention weights
        def mock_forward_hook(module, input, output):
            # Return mock attention: (1, 6, 257, 257) for DINOv2-S/14
            return torch.randn(1, 6, 257, 257)

        mock_dinov2_encoder.backbone.blocks[-1].attn.attn_drop.register_forward_hook = (
            lambda fn: MagicMock()
        )
        mock_dinov2_encoder.backbone.return_value = torch.randn(1, 384)

        # Patch the hook to inject mock attention
        with patch.object(
            mock_dinov2_encoder.backbone.blocks[-1].attn.attn_drop,
            'register_forward_hook',
            side_effect=lambda fn: (MagicMock(), fn(None, None, torch.randn(1, 6, 257, 257)))[0]
        ):
            input_tensor = torch.randn(1, 3, 224, 224)
            # This will fail gracefully and return zeros
            attribution = method.compute_attribution(input_tensor)
            assert attribution.shape == (224, 224)
            assert attribution.dtype == np.float32

    def test_vqvae_fallback_detection(self, mock_vqvae_encoder):
        """Test VQ-VAE fallback to DINOv2 method."""
        method = VQVAEAttribution(mock_vqvae_encoder, device="cpu")
        assert mock_vqvae_encoder.fallback_active is True

        # Should use DINOv2 method when fallback is active
        input_tensor = torch.randn(1, 3, 224, 224)
        attribution = method.compute_attribution(input_tensor)
        # Should return zeros (graceful failure) or valid attribution
        assert attribution.shape == (224, 224)

    def test_vqvae_primary_path(self):
        """Test VQVAEAttribution with real VQGAN (fallback_active=False)."""
        mock_encoder = MagicMock()
        mock_encoder.fallback_active = False

        # Mock backbone with conv_out layer
        mock_conv_out = MagicMock()
        mock_encoder.backbone.conv_out = mock_conv_out

        # Mock forward to return features
        mock_encoder.return_value = torch.randn(1, 384)

        # Captured features will be injected via the hook
        captured_features = []

        def mock_register_hook(hook_fn):
            # Simulate forward pass: hook captures spatial features
            spatial_features = torch.randn(1, 256, 16, 16)
            hook_fn(None, None, spatial_features)
            captured_features.append(spatial_features)
            # Return a mock handle
            return MagicMock(remove=MagicMock())

        mock_conv_out.register_forward_hook = mock_register_hook

        method = VQVAEAttribution(mock_encoder, device="cpu")
        input_tensor = torch.rand(1, 3, 224, 224)

        # Should NOT raise NotImplementedError
        attribution_map = method.compute_attribution(input_tensor)

        assert attribution_map.shape == (224, 224)
        assert attribution_map.min() >= 0 and attribution_map.max() <= 1

    def test_vjepa_temporal_input(self, mock_vjepa_encoder):
        """Test V-JEPA handles temporal input."""
        method = VJEPA2Attribution(mock_vjepa_encoder, device="cpu")
        # V-JEPA expects (B, T, 3, H, W)
        input_tensor = torch.randn(1, 16, 3, 224, 224)
        attribution = method.compute_attribution(input_tensor)
        assert attribution.shape == (224, 224)


class TestEmbeddingL2Target:
    """Test custom EmbeddingL2Target for headless encoders."""

    def test_single_embedding(self):
        """Test L2 norm computation for single embedding vector."""
        target = EmbeddingL2Target()
        # Simple embedding: [3, 4] should have L2 norm = 5
        embedding = torch.tensor([3.0, 4.0])
        result = target(embedding)

        assert result.shape == ()  # Scalar
        assert torch.isclose(result, torch.tensor(5.0), atol=1e-6)

    def test_batched_embeddings(self):
        """Test L2 norm computation for batch of embeddings."""
        target = EmbeddingL2Target()
        # Batch of 2 embeddings
        embeddings = torch.tensor([
            [3.0, 4.0],  # norm = 5
            [5.0, 12.0],  # norm = 13
        ])
        result = target(embeddings)

        assert result.shape == (2,)
        expected = torch.tensor([5.0, 13.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_realistic_encoder_output(self):
        """Test with realistic encoder embedding dimensions."""
        target = EmbeddingL2Target()
        # ViT-S/16 outputs (batch=1, embedding_dim=384)
        embedding = torch.randn(1, 384)
        result = target(embedding)

        assert result.shape == (1,)
        assert result > 0  # L2 norm is positive
        # Verify against manual computation
        expected = torch.linalg.vector_norm(embedding, dim=-1)
        assert torch.allclose(result, expected)

    def test_zero_embedding(self):
        """Test edge case: zero embedding should have zero norm."""
        target = EmbeddingL2Target()
        zero_embedding = torch.zeros(384)
        result = target(zero_embedding)

        assert result == 0.0

    def test_gradient_flows(self):
        """Test that gradients can flow through target computation."""
        target = EmbeddingL2Target()
        embedding = torch.randn(1, 384, requires_grad=True)

        result = target(embedding)
        loss = result.sum()  # Aggregate for backprop
        loss.backward()

        # Gradients should exist and be non-zero
        assert embedding.grad is not None
        assert not torch.allclose(embedding.grad, torch.zeros_like(embedding))


class TestAttributionCorrectness:
    """Test that attribution methods produce meaningful outputs."""

    def test_vit_uses_embedding_target(self, mock_vit_encoder):
        """Test that ViT attribution uses EmbeddingL2Target, not targets=None."""
        from unittest.mock import patch

        method = ViTS16Attribution(mock_vit_encoder, device="cpu")

        # Mock GradCAM to capture the targets parameter
        with patch('evaluation.gradcam.GradCAM') as MockGradCAM:
            mock_cam_instance = MagicMock()
            mock_cam_instance.return_value = np.random.rand(1, 224, 224)
            MockGradCAM.return_value = mock_cam_instance

            input_tensor = torch.randn(1, 3, 224, 224)
            try:
                method.compute_attribution(input_tensor)
            except:
                pass  # May fail due to mocking, but we just need to check the call

            # Verify GradCAM was called with EmbeddingL2Target instance
            call_kwargs = mock_cam_instance.call_args[1]
            assert 'targets' in call_kwargs
            targets = call_kwargs['targets']
            assert targets is not None, "targets should not be None"
            assert len(targets) == 1
            assert isinstance(targets[0], EmbeddingL2Target)

    def test_clip_uses_embedding_target(self, mock_clip_encoder):
        """Test that CLIP attribution uses EmbeddingL2Target, not targets=None."""
        from unittest.mock import patch

        method = CLIPAttribution(mock_clip_encoder, device="cpu")

        with patch('evaluation.gradcam.GradCAM') as MockGradCAM:
            mock_cam_instance = MagicMock()
            mock_cam_instance.return_value = np.random.rand(1, 224, 224)
            MockGradCAM.return_value = mock_cam_instance

            input_tensor = torch.randn(1, 3, 224, 224)
            try:
                method.compute_attribution(input_tensor)
            except:
                pass

            call_kwargs = mock_cam_instance.call_args[1]
            assert 'targets' in call_kwargs
            targets = call_kwargs['targets']
            assert targets is not None, "targets should not be None"
            assert len(targets) == 1
            assert isinstance(targets[0], EmbeddingL2Target)


class TestVQVAEAttributionStructure:
    """Test VQ-VAE attribution structure and fail-fast validation."""

    def test_vqvae_attribution_validates_primary_path_structure(self):
        """Test that VQVAEAttribution validates conv_out exists on init for primary VQ."""
        from encoders.vqvae import VQVAEWrapper

        # Create encoder with pretrained=False (fast, uses primary path with random weights)
        encoder = VQVAEWrapper(pretrained=False).eval()

        # Should not be using fallback
        assert encoder.fallback_active is False

        # Attribution should initialize successfully (conv_out exists)
        method = VQVAEAttribution(encoder, device="cpu")
        assert method.encoder is encoder

    def test_vqvae_attribution_raises_on_missing_backbone(self):
        """Test that VQVAEAttribution raises AttributeError if backbone is missing."""
        mock_encoder = MagicMock(spec=['fallback_active'])
        mock_encoder.fallback_active = False
        # Don't add 'backbone' to spec - accessing it will raise AttributeError

        with pytest.raises(AttributeError, match="missing 'backbone' attribute"):
            VQVAEAttribution(mock_encoder, device="cpu")

    def test_vqvae_attribution_raises_on_missing_conv_out(self):
        """Test that VQVAEAttribution raises AttributeError if conv_out is missing."""
        mock_encoder = MagicMock()
        mock_encoder.fallback_active = False
        mock_encoder.backbone = MagicMock()
        # Deliberately don't set mock_encoder.backbone.conv_out
        del mock_encoder.backbone.conv_out  # MagicMock auto-creates, so delete it

        with pytest.raises(AttributeError, match="missing 'conv_out' layer"):
            VQVAEAttribution(mock_encoder, device="cpu")

    def test_vqvae_attribution_skips_validation_for_fallback(self):
        """Test that VQVAEAttribution skips validation when fallback is active."""
        mock_encoder = MagicMock()
        mock_encoder.fallback_active = True
        # Don't set backbone or conv_out - should not raise because fallback is active

        # Should initialize successfully (fallback path doesn't need conv_out)
        method = VQVAEAttribution(mock_encoder, device="cpu")
        assert method.encoder is mock_encoder

    def test_vqvae_encoder_has_conv_out_layer(self):
        """Test that real VQVAEWrapper encoder has the conv_out layer (regression test)."""
        from encoders.vqvae import VQVAEWrapper
        from encoders._vqgan_arch import Encoder

        # Use pretrained=False for fast test (no network download)
        encoder = VQVAEWrapper(pretrained=False).eval()

        # Skip if fallback is active (shouldn't happen with pretrained=False, but be safe)
        if encoder.fallback_active:
            pytest.skip("Fallback is active, cannot test primary VQ structure")

        # Verify structure
        assert hasattr(encoder, 'backbone'), "VQVAEWrapper should have backbone attribute"
        assert isinstance(encoder.backbone, Encoder), "backbone should be Encoder instance"
        assert hasattr(encoder.backbone, 'conv_out'), "Encoder should have conv_out layer"
        assert isinstance(encoder.backbone.conv_out, torch.nn.Conv2d), "conv_out should be Conv2d"


class TestAttributionPipeline:
    """Test the complete attribution pipeline orchestration."""

    def test_frame_selection_stratification(self):
        """Test that frame selection is stratified correctly."""
        # Create mock scenario classifications
        scenario_classifications = {
            "highway": list(range(10)),
            "urban": list(range(10, 20)),
            "intersection": list(range(20, 30)),
            "other": list(range(30, 40)),
        }

        # Mock dataset
        mock_dataset = MagicMock()

        # Create minimal pipeline instance (__init__ no longer loads encoders)
        pipeline = AttributionPipeline(
            split="p0_test",
            n_per_scenario=5,
            seed=42,
        )

        selected = pipeline.select_frames(mock_dataset, scenario_classifications)

        # Should have 5 frames per scenario × 4 scenarios = 20 frames
        assert len(selected) == 20

        # Count per scenario
        scenario_counts = {}
        for _, scenario in selected:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        # Each scenario should have exactly 5 frames
        for scenario in scenario_classifications.keys():
            assert scenario_counts[scenario] == 5

    def test_frame_selection_reproducibility(self):
        """Test that frame selection is deterministic with same seed."""
        scenario_classifications = {
            "highway": list(range(100)),
            "urban": list(range(100, 200)),
            "intersection": list(range(200, 300)),
            "other": list(range(300, 400)),
        }

        mock_dataset = MagicMock()

        pipeline1 = AttributionPipeline(
            split="p0_test",
            n_per_scenario=5,
            seed=42,
        )
        selected1 = pipeline1.select_frames(mock_dataset, scenario_classifications)

        pipeline2 = AttributionPipeline(
            split="p0_test",
            n_per_scenario=5,
            seed=42,
        )
        selected2 = pipeline2.select_frames(mock_dataset, scenario_classifications)

        # Same seed should produce identical selections
        assert selected1 == selected2

    def test_insufficient_frames_skips_scenario(self):
        """Test that scenarios with insufficient samples are skipped with warning."""
        scenario_classifications = {
            "highway": list(range(3)),  # Only 3 frames, need 5 - will be skipped
            "urban": list(range(10)),   # 10 frames, need 5 - will be sampled
        }

        mock_dataset = MagicMock()

        pipeline = AttributionPipeline(
            split="p0_test",
            n_per_scenario=5,
            seed=42,
        )

        # Should skip 'highway' and sample from 'urban'
        selected = pipeline.select_frames(mock_dataset, scenario_classifications)

        # Should have 5 frames from 'urban' only
        assert len(selected) == 5
        # All frames should be from 'urban' scenario
        for _, scenario in selected:
            assert scenario == "urban"

    def test_all_scenarios_insufficient_raises_error(self):
        """Test that error is raised when all scenarios have insufficient samples."""
        scenario_classifications = {
            "highway": list(range(3)),  # Only 3 frames, need 5
            "urban": list(range(2)),    # Only 2 frames, need 5
        }

        mock_dataset = MagicMock()

        pipeline = AttributionPipeline(
            split="p0_test",
            n_per_scenario=5,
            seed=42,
        )

        with pytest.raises(ValueError, match="No scenarios have sufficient samples"):
            pipeline.select_frames(mock_dataset, scenario_classifications)

    def test_overlay_generation(self):
        """Test attribution overlay generation."""
        pipeline = AttributionPipeline(
            split="p0_test",
        )

        # Create mock image and attribution
        image_rgb = np.random.rand(224, 224, 3).astype(np.float32)
        attribution_map = np.random.rand(224, 224).astype(np.float32)

        overlay, title = pipeline.generate_attribution_overlay(
            image_rgb,
            attribution_map,
            encoder_name="vit_s16",
            scenario="highway",
            frame_idx=0,
            is_fallback=False,
        )

        # Overlay should be RGB image
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8 or overlay.dtype == np.float64

        # Title should contain encoder and scenario
        assert "vit_s16" in title
        assert "highway" in title

    def test_vq_fallback_tagging(self):
        """Test that VQ fallback is properly tagged in overlay title."""
        pipeline = AttributionPipeline(
            split="p0_test",
        )

        image_rgb = np.random.rand(224, 224, 3).astype(np.float32)
        attribution_map = np.random.rand(224, 224).astype(np.float32)

        overlay, title = pipeline.generate_attribution_overlay(
            image_rgb,
            attribution_map,
            encoder_name="vqvae",
            scenario="urban",
            frame_idx=5,
            is_fallback=True,
        )

        # Title should contain fallback caveat
        assert "fallback" in title.lower()
        assert "DINOv2" in title

    def test_method_name_mapping(self):
        """Test that encoder method names are correctly mapped."""
        pipeline = AttributionPipeline(
            split="p0_test",
        )

        assert "GradCAM" in pipeline._get_method_name("vit_s16", False)
        assert "SelfAttention" in pipeline._get_method_name("dinov2_s14", False)
        assert "GradCAM" in pipeline._get_method_name("clip_b32", False)
        assert "TemporalAttention" in pipeline._get_method_name("vjepa2", False)
        assert "RealClip" in pipeline._get_method_name("vjepa2", False)
        assert "TemporalAttention" in pipeline._get_method_name("vjepa2_rep1", False)
        assert "T1" in pipeline._get_method_name("vjepa2_rep1", False)

        # VQ with fallback should use DINOv2's method with fallback notation
        vq_fallback_name = pipeline._get_method_name("vqvae", True)
        assert "fallback" in vq_fallback_name.lower()
        assert "SelfAttention" in vq_fallback_name  # Uses DINOv2's method

    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "test_outputs"
        assert not output_dir.exists()

        pipeline = AttributionPipeline(
            split="p0_test",
            output_dir=output_dir,
        )

        assert output_dir.exists()


class TestIntegration:
    """Integration tests for the complete pipeline (requires real data)."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires NuScenes dataset and GPU")
    def test_full_pipeline_smoke(self):
        """Smoke test the complete pipeline on smoke_test split with reduced samples."""
        # This test would run the full pipeline on a small dataset
        # Skipped by default as it requires real data and GPU

        pipeline = AttributionPipeline(
            split="smoke_test",
            device="cuda",
            n_per_scenario=2,  # Reduced for smoke test
            seed=42,
        )

        report = pipeline.run()

        # Verify output structure
        assert report["n_frames"] == 8  # 2 per scenario × 4 scenarios
        assert len(report["encoders"]) == 5

        # Verify files exist
        output_dir = Path("outputs/attribution")
        png_files = list(output_dir.glob("*.png"))
        pdf_files = list(output_dir.glob("*.pdf"))
        json_file = output_dir / "figures_method_report.json"

        assert len(png_files) == 40  # 8 frames × 5 encoders
        assert len(pdf_files) == 5  # 1 per encoder
        assert json_file.exists()

        # Verify JSON schema
        with open(json_file) as f:
            loaded_report = json.load(f)
        assert loaded_report == report
        for enc_info in loaded_report["encoders"].values():
            assert "method" in enc_info
            assert "fallback_used" in enc_info
            assert "overlay_paths" in enc_info
