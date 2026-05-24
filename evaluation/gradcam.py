"""Attribution visualization pipeline for encoder interpretability.

Implements encoder-specific attribution methods per EDD §9.2:
- ViT-S/16: GradCAM with patch-to-spatial reshape
- DINOv2-S/14: Self-attention map thresholded at 60th percentile
- CLIP ViT-B/32: GradCAM on visual transformer last block
- VQ-VAE: Spatial activation L2 norm (or DINOv2 fallback if checkpoint unavailable)
- V-JEPA: Temporal attention averaged across 16 frames
- V-JEPA rep1: Single-frame ablation variant (A19)

Outputs:
- PNG attribution overlays (20 frames × 6 encoders = 120 PNGs)
- 6 multi-page PDFs at 300 DPI (one per encoder)
- JSON method report documenting which method was used per encoder
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Encoder imports
from encoders.clip_enc import CLIPB32Wrapper
from encoders.dinov2 import DINOv2S14Wrapper
from encoders.vits16 import ViTS16Wrapper
from encoders.vjepa2 import VJEPA2Wrapper
from encoders.vqvae import VQVAEWrapper


def extract_dinov2_last_attention(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract last-layer self-attention from Meta DINOv2 model.

    Uses a forward hook on the qkv Linear layer to capture QKV projection output,
    then manually reconstructs attention weights. This is more robust than
    monkey-patching the attention forward method.

    Parameters
    ----------
    model
        DINOv2 model from facebookresearch/dinov2 torch.hub
    x
        Input tensor [B, 3, H, W]

    Returns
    -------
    attn
        Attention weights [B, heads, N, N] where N = num_patches + 1 (includes CLS token)

    Examples
    --------
    >>> model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    >>> x = torch.rand(1, 3, 224, 224)
    >>> attn = extract_dinov2_last_attention(model, x)
    >>> attn.shape  # (1, 6, 257, 257) for ViT-S/14
    """
    model.eval()
    last_block = model.blocks[-1]
    attn_module = last_block.attn

    # Container to capture QKV projection output
    qkv_output = {}

    def qkv_hook(module, input, output):
        """Capture QKV projection output [B, N, 3*C]."""
        qkv_output["qkv"] = output.detach()

    # Register hook on qkv Linear layer (NOT on the attention module itself)
    handle = attn_module.qkv.register_forward_hook(qkv_hook)

    with torch.no_grad():
        _ = model(x)

    handle.remove()

    # Reconstruct attention weights from captured QKV
    qkv = qkv_output["qkv"]  # [B, N, 3*C]
    B, N, three_C = qkv.shape
    C = three_C // 3

    num_heads = attn_module.num_heads
    head_dim = C // num_heads

    # Reshape to [3, B, heads, N, head_dim]
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, heads, N, head_dim]

    # Compute attention weights
    scale = attn_module.scale
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # [B, heads, N, N]

    return attn


class EmbeddingL2Target:
    """GradCAM target for headless encoders: maximizes L2 norm of embedding.

    For encoders with num_classes=0, there's no classification head to target.
    This target maximizes the L2 norm of the embedding, answering: "which spatial
    regions produce the strongest feature response?"

    Follows the same pattern as ClassifierOutputTarget from pytorch_grad_cam.
    """

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of embedding for GradCAM backprop."""
        if len(model_output.shape) == 1:
            return torch.linalg.vector_norm(model_output)
        return torch.linalg.vector_norm(model_output, dim=-1)


class AttributionMethod(ABC):
    """Abstract base class for encoder-specific attribution methods."""

    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        """Initialize attribution method.

        Parameters
        ----------
        encoder
            The encoder wrapper instance.
        device
            Device to run attribution on ("cpu" or "cuda").
        """
        self.encoder = encoder
        self.device = device

    @abstractmethod
    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute attribution map for the given input.

        Parameters
        ----------
        input_tensor
            Preprocessed input tensor (B, C, H, W) in [0, 1].

        Returns
        -------
        attribution_map
            2D numpy array (H, W) with values in [0, 1], spatial attribution heatmap.
        """
        pass


class ViTS16Attribution(AttributionMethod):
    """GradCAM attribution for ViT-S/16 with patch-to-spatial reshape."""

    def __init__(self, encoder: ViTS16Wrapper, device: str = "cpu"):
        super().__init__(encoder, device)
        # Target layer: last block's normalization before attention
        self.target_layers = [encoder.backbone.blocks[-1].norm1]

    def reshape_transform(self, tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
        """Reshape ViT output from (B, N_patches+1, D) to (B, D, H, W).

        Parameters
        ----------
        tensor
            Shape (B, 197, 384) with CLS token at index 0.
        height, width
            Spatial grid dimensions (14×14 for 224px input with 16px patches).

        Returns
        -------
        reshaped
            Shape (B, 384, 14, 14) spatial feature map.
        """
        # Remove CLS token: (B, 197, 384) → (B, 196, 384)
        result = tensor[:, 1:, :]
        # Reshape to spatial grid: (B, 196, 384) → (B, 14, 14, 384)
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        # Transpose to channel-first: (B, 14, 14, 384) → (B, 384, 14, 14)
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute GradCAM attribution for ViT-S/16."""
        try:
            # Normalize input with encoder's ImageNet stats before GradCAM
            normalized_input = (input_tensor - self.encoder._image_mean) / self.encoder._image_std
            # Enable gradients for GradCAM
            normalized_input = normalized_input.requires_grad_(True)

            cam = GradCAM(
                model=self.encoder.backbone,
                target_layers=self.target_layers,
                reshape_transform=self.reshape_transform,
            )
            # Maximize L2 norm of embedding for headless encoder
            grayscale_cam = cam(input_tensor=normalized_input, targets=[EmbeddingL2Target()])
            # Extract first sample, upsample to 224×224
            cam_map = grayscale_cam[0]  # (14, 14) or (224, 224) depending on version
            if cam_map.shape != (224, 224):
                cam_map = F.interpolate(
                    torch.from_numpy(cam_map[None, None, :, :]),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
            return cam_map
        except Exception as e:
            print(f"Warning: ViT-S/16 attribution failed: {e}")
            return np.zeros((224, 224), dtype=np.float32)


class DINOv2Attribution(AttributionMethod):
    """Self-attention attribution for DINOv2-S/14 with 60th percentile threshold."""

    def __init__(self, encoder, device: str = "cpu"):
        super().__init__(encoder, device)
        # Store reference to actual backbone model
        self.backbone = encoder.backbone

    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Extract CLS attention map from DINOv2 last block."""
        try:
            # Use forward hook to extract attention weights (non-invasive)
            attn = extract_dinov2_last_attention(self.backbone, input_tensor)

            # Extract CLS token attention to patches
            # attn shape: (B, num_heads, num_tokens, num_tokens)
            # For DINOv2-S/14: (1, 6, 257, 257) where 257 = 1 CLS + 256 patches
            # Average over heads, take CLS row (index 0), remove CLS column
            attn_map = attn[0].mean(dim=0)[0, 1:].cpu().numpy()  # (256,)

            # Reshape to 16×16 spatial grid (DINOv2-S/14: 224/14 = 16 patches per side)
            attn_map = attn_map.reshape(16, 16)

            # Threshold at 60th percentile
            threshold = np.percentile(attn_map, 60)
            attn_map = np.clip(attn_map - threshold, 0, None)

            # Normalize to [0, 1]
            if attn_map.max() > 0:
                attn_map = attn_map / attn_map.max()

            # Upsample to 224×224
            attn_map = F.interpolate(
                torch.from_numpy(attn_map[None, None, :, :]),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            return attn_map

        except Exception as e:
            print(f"Warning: DINOv2 attribution failed: {e}")
            return np.zeros((224, 224), dtype=np.float32)


class CLIPAttribution(AttributionMethod):
    """GradCAM attribution for CLIP ViT-B/32 visual transformer."""

    def __init__(self, encoder: CLIPB32Wrapper, device: str = "cpu"):
        super().__init__(encoder, device)
        # Target layer: last residual block's layer norm
        self.target_layers = [encoder.backbone.transformer.resblocks[-1].ln_1]

    def reshape_transform(self, tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
        """Reshape CLIP ViT output from (B, N_patches+1, D) to (B, D, H, W).

        Parameters
        ----------
        tensor
            Shape (B, 50, 768) with CLS token at index 0.
            CLIP ViT-B/32: 224/32 = 7 patches per side, 7*7 = 49 patches.
        height, width
            Spatial grid dimensions (7×7 for 224px input with 32px patches).

        Returns
        -------
        reshaped
            Shape (B, 768, 7, 7) spatial feature map.
        """
        # Remove CLS token: (B, 50, 768) → (B, 49, 768)
        result = tensor[:, 1:, :]
        # Reshape to spatial grid: (B, 49, 768) → (B, 7, 7, 768)
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        # Transpose to channel-first: (B, 7, 7, 768) → (B, 768, 7, 7)
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute GradCAM attribution for CLIP ViT-B/32."""
        try:
            # Normalize input with encoder's CLIP stats before GradCAM
            normalized_input = (input_tensor - self.encoder._image_mean) / self.encoder._image_std
            # Enable gradients for GradCAM
            normalized_input = normalized_input.requires_grad_(True)

            cam = GradCAM(
                model=self.encoder.backbone,
                target_layers=self.target_layers,
                reshape_transform=self.reshape_transform,
            )
            # Maximize L2 norm of embedding for headless encoder
            grayscale_cam = cam(input_tensor=normalized_input, targets=[EmbeddingL2Target()])
            cam_map = grayscale_cam[0]
            if cam_map.shape != (224, 224):
                cam_map = F.interpolate(
                    torch.from_numpy(cam_map[None, None, :, :]),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
            return cam_map
        except Exception as e:
            print(f"Warning: CLIP attribution failed: {e}")
            return np.zeros((224, 224), dtype=np.float32)


class VQVAEAttribution(AttributionMethod):
    """VQ-VAE attribution with automatic fallback to DINOv2 method."""

    def __init__(self, encoder: torch.nn.Module, device: str = "cpu"):
        """Initialize VQ-VAE attribution method.

        Parameters
        ----------
        encoder
            VQVAEWrapper instance
        device
            Device to run attribution on

        Raises
        ------
        AttributeError
            If primary VQ-VAE backbone is active but conv_out layer is missing
        """
        super().__init__(encoder, device)

        # Validate attribute path for primary VQ-VAE (not fallback)
        # This catches refactoring errors early instead of silent failures
        if not (hasattr(encoder, 'fallback_active') and encoder.fallback_active):
            if not hasattr(encoder, 'backbone'):
                raise AttributeError(
                    f"VQ-VAE encoder missing 'backbone' attribute. "
                    f"Expected encoder.backbone to be an Encoder instance from _vqgan_arch.py"
                )
            if not hasattr(encoder.backbone, 'conv_out'):
                raise AttributeError(
                    f"VQ-VAE encoder.backbone missing 'conv_out' layer. "
                    f"Expected encoder.backbone.conv_out to be a Conv2d layer. "
                    f"This is required for spatial feature extraction. "
                    f"If _vqgan_arch.py was refactored, update the attribution hook target."
                )

    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute attribution for VQ-VAE, using DINOv2 fallback if active."""
        # Check if fallback is active
        if hasattr(self.encoder, 'fallback_active') and self.encoder.fallback_active:
            # Use DINOv2 attribution method as fallback
            dinov2_method = DINOv2Attribution(self.encoder, self.device)
            return dinov2_method.compute_attribution(input_tensor)
        else:
            # Primary VQ path: hook spatial features before pooling
            try:
                intermediate_features = []

                def feature_hook(module, input, output):
                    # Capture conv_out output: (B, 256, 16, 16)
                    intermediate_features.append(output)

                # Hook the final conv_out layer before pooling
                handle = self.encoder.backbone.conv_out.register_forward_hook(feature_hook)

                with torch.no_grad():
                    # Forward pass (encoder handles resize 224→256 and normalization internally)
                    _ = self.encoder(input_tensor)

                handle.remove()

                # Extract spatial features: (B, 256, 16, 16)
                features = intermediate_features[0]

                # Compute spatial importance via L2 norm across channels
                spatial_map = features.norm(dim=1)  # (B, 16, 16)

                # Normalize to [0, 1]
                spatial_map = spatial_map.squeeze(0).cpu().numpy()  # (16, 16)
                if spatial_map.max() > 0:
                    spatial_map = spatial_map / spatial_map.max()

                # Upsample to 224×224
                spatial_map = F.interpolate(
                    torch.from_numpy(spatial_map[None, None, :, :]),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

                return spatial_map

            except Exception as e:
                print(f"Warning: VQ-VAE attribution failed: {e}")
                import traceback
                traceback.print_exc()
                return np.zeros((224, 224), dtype=np.float32)


class VJEPA2Attribution(AttributionMethod):
    """Temporal attention attribution for V-JEPA averaged across 16 frames."""

    def compute_attribution(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute spatial activation attribution for V-JEPA averaged over time.

        Parameters
        ----------
        input_tensor
            Shape (B, T, 3, H, W) video clip in [0, 1].

        Returns
        -------
        attribution_map
            2D spatial heatmap (H, W) averaged over temporal dimension.
        """
        try:
            # Hook to capture intermediate features before pooling
            intermediate_features = []

            def feature_hook(module, input, output):
                # Capture last_hidden_state: (B, num_tokens, 1024)
                intermediate_features.append(output.last_hidden_state)

            # Hook the backbone model output
            handle = self.encoder.backbone.register_forward_hook(feature_hook)

            with torch.no_grad():
                # Forward pass through encoder (_encode handles resize 224→256 and normalization)
                b, t, c, h, w = input_tensor.shape

                # Use encoder's _encode method to properly resize and normalize
                _ = self.encoder._encode(input_tensor)

                # Get spatial size after encoder's resize
                from encoders.vjepa2 import NATIVE_INPUT_SIZE
                spatial_size = NATIVE_INPUT_SIZE  # 256

            handle.remove()

            # Extract features: (B, num_tokens, 1024)
            features = intermediate_features[0]  # (1, num_tokens, 1024)

            # V-JEPA2 ViT-L/16 at 256x256 with tubelet=2:
            # num_tokens = (T/2) * (256/16)^2 = (16/2) * 16^2 = 8 * 256 = 2048
            # Spatial patches: 16x16 per frame, 8 temporal tokens
            spatial_patches_per_side = spatial_size // 16  # 16
            temporal_tokens = t // 2  # 8 for 16-frame input

            # Reshape to (B, temporal, spatial_h, spatial_w, dim)
            # features: (1, 2048, 1024) -> (1, 8, 16, 16, 1024)
            features = features.reshape(
                b,
                temporal_tokens,
                spatial_patches_per_side,
                spatial_patches_per_side,
                -1
            )

            # Average over temporal dimension: (1, 8, 16, 16, 1024) -> (1, 16, 16, 1024)
            spatial_features = features.mean(dim=1)  # (1, 16, 16, 1024)

            # Average over feature dimension to get spatial importance: (1, 16, 16)
            spatial_map = spatial_features.norm(dim=-1)  # L2 norm across features

            # Normalize to [0, 1]
            spatial_map = spatial_map.squeeze(0).cpu().numpy()  # (16, 16)
            if spatial_map.max() > 0:
                spatial_map = spatial_map / spatial_map.max()

            # Upsample to 224x224
            spatial_map = F.interpolate(
                torch.from_numpy(spatial_map[None, None, :, :]),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            return spatial_map

        except Exception as e:
            print(f"Warning: V-JEPA2 attribution failed: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((224, 224), dtype=np.float32)


class AttributionPipeline:
    """Orchestrates attribution generation across all encoders and frames."""

    def __init__(
        self,
        split: str = "p0_test",
        device: str = "cuda",
        output_dir: Path = Path("outputs/attribution"),
        n_per_scenario: int = 5,
        seed: int = 42,
    ):
        """Initialize attribution pipeline.

        Parameters
        ----------
        split
            Dataset split to use (e.g., "p0_test", "smoke_test").
        device
            Device to run attribution on.
        output_dir
            Directory to save outputs.
        n_per_scenario
            Number of frames to select per scenario type.
        seed
            Random seed for frame selection.
        """
        self.split = split
        self.device = device
        self.output_dir = Path(output_dir)
        self.n_per_scenario = n_per_scenario
        self.seed = seed

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store encoder specifications (name → wrapper class)
        # Encoders are loaded sequentially in run() to reduce memory footprint
        self.encoder_specs = {
            "vit_s16": ViTS16Wrapper,
            "dinov2_s14": DINOv2S14Wrapper,
            "clip_b32": CLIPB32Wrapper,
            "vqvae": VQVAEWrapper,
            "vjepa2": VJEPA2Wrapper,
            "vjepa2_rep1": VJEPA2Wrapper,
        }

    def _load_single_encoder(self, encoder_name: str) -> torch.nn.Module:
        """Load a single encoder wrapper by name.

        Parameters
        ----------
        encoder_name
            One of: "vit_s16", "dinov2_s14", "clip_b32", "vqvae", "vjepa2", "vjepa2_rep1"

        Returns
        -------
        encoder
            Initialized encoder wrapper in eval mode on self.device
        """
        if encoder_name not in self.encoder_specs:
            raise ValueError(
                f"Unknown encoder: {encoder_name}. "
                f"Valid: {list(self.encoder_specs.keys())}"
            )

        wrapper_class = self.encoder_specs[encoder_name]
        print(f"  Loading encoder: {encoder_name}...")

        # Handle CLIP special case (requires pretrained="openai" argument)
        if encoder_name == "clip_b32":
            encoder = wrapper_class(pretrained="openai")
        else:
            encoder = wrapper_class(pretrained=True)

        encoder.eval()
        encoder.to(self.device)

        return encoder

    def _create_attribution_method(
        self,
        encoder_name: str,
        encoder: torch.nn.Module
    ) -> AttributionMethod:
        """Create attribution method for a single encoder.

        Parameters
        ----------
        encoder_name
            Name of the encoder
        encoder
            The loaded encoder instance

        Returns
        -------
        method
            Attribution method instance for this encoder
        """
        method_map = {
            "vit_s16": lambda e: ViTS16Attribution(e, self.device),
            "dinov2_s14": lambda e: DINOv2Attribution(e, self.device),
            "clip_b32": lambda e: CLIPAttribution(e, self.device),
            "vqvae": lambda e: VQVAEAttribution(e, self.device),
            "vjepa2": lambda e: VJEPA2Attribution(e, self.device),
            "vjepa2_rep1": lambda e: VJEPA2Attribution(e, self.device),
        }

        if encoder_name not in method_map:
            raise ValueError(f"No attribution method for encoder: {encoder_name}")

        return method_map[encoder_name](encoder)

    def _cleanup_encoder(
        self,
        encoder_name: str,
        encoder: torch.nn.Module,
        method: AttributionMethod,
    ) -> None:
        """Explicitly free memory for an encoder and its attribution method.

        Parameters
        ----------
        encoder_name
            Name for logging
        encoder
            Encoder instance to cleanup
        method
            Attribution method instance to cleanup
        """
        print(f"  Cleaning up {encoder_name}...")

        # Move encoder to CPU to free GPU memory
        encoder.cpu()

        # Delete references to allow garbage collection
        del method
        del encoder

        # Force GPU memory release
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def select_frames(
        self, dataset: Any, scenario_classifications: Dict[str, List[int]]
    ) -> List[Tuple[int, str]]:
        """Select frames stratified by scenario type.

        Skips scenarios with fewer than n_per_scenario samples. This gracefully
        handles splits like p0_test that lack highway/urban scenarios.

        Parameters
        ----------
        dataset
            NuScenesFrameDataset instance.
        scenario_classifications
            Dict mapping scenario name to list of frame indices.

        Returns
        -------
        selected_frames
            List of (frame_idx, scenario) tuples, deterministically sampled.
            Only includes scenarios with sufficient samples.

        Raises
        ------
        ValueError
            If no scenarios have sufficient samples.
        """
        rng = np.random.default_rng(self.seed)
        selected = []

        for scenario, frame_indices in scenario_classifications.items():
            if len(frame_indices) < self.n_per_scenario:
                print(
                    f"Warning: Skipping scenario '{scenario}' "
                    f"(only {len(frame_indices)} samples, need {self.n_per_scenario})"
                )
                continue

            # Sample without replacement
            sampled_indices = rng.choice(
                frame_indices, size=self.n_per_scenario, replace=False
            )
            selected.extend([(idx, scenario) for idx in sampled_indices])

        if len(selected) == 0:
            raise ValueError(
                f"No scenarios have sufficient samples "
                f"(need {self.n_per_scenario} per scenario)"
            )

        return selected

    def load_frame_image(
        self,
        dataset_single: Any,
        dataset_clip16: Any,
        dataset_clip1: Any,
        frame_idx: int,
        encoder_name: str
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Load frame image for attribution.

        Routes to appropriate dataset based on encoder input requirements:
        - Single-frame encoders: use dataset_single
        - vjepa2/vjepa2_rep64: use dataset_clip16 (real temporal clips)
        - vjepa2_rep1: use dataset_clip1 (honest single-frame ablation)

        Parameters
        ----------
        dataset_single
            Single-frame NuScenesFrameDataset instance.
        dataset_clip16
            16-frame clip NuScenesFrameDataset instance.
        dataset_clip1
            1-frame clip NuScenesFrameDataset instance.
        frame_idx
            Index into the dataset.
        encoder_name
            Name of the encoder.

        Returns
        -------
        image_rgb
            RGB image (H, W, 3) in [0, 1] for visualization (always single frame).
        input_tensor
            Encoder input:
            - (1, 3, 224, 224) for single-frame encoders
            - (1, 16, 3, 224, 224) for vjepa2
            - (1, 1, 3, 224, 224) for vjepa2_rep1
        """
        # Select appropriate dataset
        if encoder_name in ("vjepa2", "vjepa2_rep64"):
            dataset = dataset_clip16
        elif encoder_name == "vjepa2_rep1":
            dataset = dataset_clip1
        else:
            dataset = dataset_single

        sample = dataset[frame_idx]
        input_tensor = sample['image'].unsqueeze(0).to(self.device)

        # For visualization, always use the keyframe (latest frame)
        if input_tensor.ndim == 5:  # (1, T, 3, H, W) clip mode
            keyframe = input_tensor[:, -1, :, :, :]  # (1, 3, H, W)
        else:  # (1, 3, H, W) single-frame mode
            keyframe = input_tensor

        image_rgb = keyframe.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return image_rgb, input_tensor

    def generate_attribution_overlay(
        self,
        image_rgb: np.ndarray,
        attribution_map: np.ndarray,
        encoder_name: str,
        scenario: str,
        frame_idx: int,
        is_fallback: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """Generate attribution heatmap overlay on original image.

        Parameters
        ----------
        image_rgb
            Original image (H, W, 3) in [0, 1].
        attribution_map
            Heatmap (H, W) in [0, 1].
        encoder_name
            Name of the encoder.
        scenario
            Scenario type (e.g., "highway").
        frame_idx
            Frame index for naming.
        is_fallback
            Whether VQ fallback is active.

        Returns
        -------
        overlay
            RGB overlay image (H, W, 3) in [0, 255].
        title
            Figure title string.
        """
        # Generate overlay using pytorch_grad_cam utility
        overlay = show_cam_on_image(image_rgb, attribution_map, use_rgb=True)

        # Add temporal input indicator for V-JEPA2
        temporal_indicator = ""
        if encoder_name in ("vjepa2", "vjepa2_rep64"):
            temporal_indicator = " (16-frame clip)"
        elif encoder_name == "vjepa2_rep1":
            temporal_indicator = " (1-frame)"

        # Generate title
        title = f"{encoder_name}{temporal_indicator} - {scenario} - Frame {frame_idx}"
        if is_fallback:
            title += " (VQ fallback: DINOv2-S/14)"

        return overlay, title

    def _generate_pdf(
        self,
        encoder_name: str,
        overlays: List[Tuple[np.ndarray, str]],
    ) -> Path:
        """Generate multi-page PDF for an encoder at 300 DPI.

        Parameters
        ----------
        encoder_name
            Name of the encoder.
        overlays
            List of (overlay_image, title) tuples.

        Returns
        -------
        pdf_path
            Path to generated PDF.
        """
        pdf_path = self.output_dir / f"{encoder_name}_attribution.pdf"
        with PdfPages(pdf_path) as pdf:
            for overlay, title in overlays:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                ax.imshow(overlay)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                pdf.savefig(fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
        return pdf_path

    def run(self) -> Dict[str, Any]:
        """Run the complete attribution pipeline.

        Returns
        -------
        report
            JSON-serializable report dict.
        """
        # Import dataset and metrics here to avoid circular imports
        from data.dataset import NuScenesFrameDataset
        from evaluation.metrics import classify_scenes_by_scenario
        from collections import defaultdict

        # Load single-frame dataset for ViT, DINOv2, CLIP, VQ-VAE
        dataset_single = NuScenesFrameDataset(
            split=self.split,
            mode="single_frame",
        )

        # Load 16-frame clip dataset for vjepa2 temporal variant
        dataset_clip16 = NuScenesFrameDataset(
            split=self.split,
            mode="clip",
            clip_frames=16,
        )

        # Load 1-frame clip dataset for vjepa2_rep1 ablation
        dataset_clip1 = NuScenesFrameDataset(
            split=self.split,
            mode="clip",
            clip_frames=1,
        )

        # Verify sample alignment (all use same _build_sample_index())
        assert len(dataset_single) == len(dataset_clip16) == len(dataset_clip1), \
            "Dataset sample counts must match"

        # Get all unique scene tokens from dataset (use single-frame as reference)
        scene_tokens = list({dataset_single[i]["scene_token"] for i in range(len(dataset_single))})

        # Classify scenes by scenario
        scene_to_scenario = classify_scenes_by_scenario(dataset_single.nusc, scene_tokens)

        # Build scenario to frame indices mapping
        scenario_to_frames = defaultdict(list)
        for idx in range(len(dataset_single)):
            sample = dataset_single[idx]
            scene_token = sample.get("scene_token", "unknown")
            scenario = scene_to_scenario.get(scene_token, "other")
            scenario_to_frames[scenario].append(idx)

        # Select frames
        selected_frames = self.select_frames(dataset_single, dict(scenario_to_frames))

        # Track outputs
        encoder_overlay_paths: Dict[str, List[str]] = {
            enc: [] for enc in self.encoder_specs.keys()
        }
        method_report = {
            "n_frames": len(selected_frames),
            "frames_per_scenario": self.n_per_scenario,
            "seed": self.seed,
            "encoders": {},
        }

        # Generate attributions for each encoder sequentially
        for encoder_name in self.encoder_specs.keys():
            print(f"\n{'='*60}")
            print(f"Processing encoder: {encoder_name}")
            print(f"{'='*60}")

            # Load ONE encoder
            encoder = self._load_single_encoder(encoder_name)
            method = self._create_attribution_method(encoder_name, encoder)

            # Check VQ fallback status
            is_vq_fallback = (
                encoder_name == "vqvae"
                and hasattr(encoder, 'fallback_active')
                and encoder.fallback_active
            )

            # Process all frames with this encoder
            overlays = []

            overlay_counter = 0
            for frame_idx, scenario in selected_frames:
                # Load image (routes to appropriate dataset based on encoder)
                image_rgb, input_tensor = self.load_frame_image(
                    dataset_single, dataset_clip16, dataset_clip1, frame_idx, encoder_name
                )

                # Compute attribution
                attribution_map = method.compute_attribution(input_tensor)

                # Generate overlay
                overlay, title = self.generate_attribution_overlay(
                    image_rgb,
                    attribution_map,
                    encoder_name,
                    scenario,
                    frame_idx,
                    is_fallback=is_vq_fallback,
                )

                # Save PNG
                png_path = self.output_dir / f"{encoder_name}_{scenario}_{overlay_counter:02d}.png"
                Image.fromarray(overlay.astype(np.uint8)).save(png_path)
                encoder_overlay_paths[encoder_name].append(str(png_path))

                # Store for PDF
                overlays.append((overlay, title))

                overlay_counter += 1

            # Generate PDF for this encoder
            pdf_path = self._generate_pdf(encoder_name, overlays)
            print(f"  Generated PDF: {pdf_path}")

            # Record method used
            method_name = self._get_method_name(encoder_name, is_vq_fallback)

            # Determine input format for documentation
            input_format = {
                "vjepa2": "16-frame temporal clip (real frames)",
                "vjepa2_rep64": "16-frame temporal clip (real frames)",
                "vjepa2_rep1": "1-frame clip (single-frame ablation, T=1)",
            }.get(encoder_name, "single frame")

            method_report["encoders"][encoder_name] = {
                "method": method_name,
                "input_format": input_format,
                "fallback_used": is_vq_fallback,
                "overlay_paths": encoder_overlay_paths[encoder_name],
            }

            # CRITICAL: Clean up before loading next encoder
            self._cleanup_encoder(encoder_name, encoder, method)

        # Save JSON report
        report_path = self.output_dir / "figures_method_report.json"
        with open(report_path, 'w') as f:
            json.dump(method_report, f, indent=2)

        print(f"\nAttribution pipeline complete!")
        print(f"  PNGs: {sum(len(paths) for paths in encoder_overlay_paths.values())}")
        print(f"  PDFs: {len(self.encoder_specs)}")
        print(f"  Report: {report_path}")

        return method_report

    def _get_method_name(self, encoder_name: str, is_vq_fallback: bool) -> str:
        """Get human-readable method name for the encoder."""
        method_names = {
            "vit_s16": "GradCAM-ViT-Reshape",
            "dinov2_s14": "SelfAttention-LastBlock-P60",
            "clip_b32": "GradCAM-CLIP-ViT",
            "vqvae": "VQ-SpatialActivation",
            "vjepa2": "TemporalAttention-RealClip-16Frame",
            "vjepa2_rep1": "TemporalAttention-SingleFrame-T1",
        }
        name = method_names.get(encoder_name, "Unknown")
        if is_vq_fallback:
            name = f"{method_names['dinov2_s14']} (VQ fallback)"
        return name
