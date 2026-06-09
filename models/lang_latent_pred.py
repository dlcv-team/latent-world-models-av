"""Language-conditioned latent world model (P2).

Extends M1's action-conditioned :class:`~models.latent_pred.LatentPredictor`
with a natural-language pathway: a scene caption (task C6) is encoded by a
**frozen** CLIP ViT-B/32 text encoder (512-d), projected to the 384-d latent
space by a small trainable ``Linear(512, 384, bias=False)``, and concatenated
with the current latent ``z_t`` (384-d) and the Fourier action embedding
``a_embed`` (384-d).  The predictor MLP therefore takes a **1152-d** input::

    [ z_t (384) | a_embed (384) | text_proj(text_embed) (384) ] -> (1152,)

and predicts the next ``horizon`` latent states, exactly like M1.

Initialisation from M1
----------------------
:meth:`LanguageConditionedLatentPredictor.init_shared_layers_from` copies the
two shared hidden layers verbatim from a trained M1 predictor and copies M1's
first-layer weights into the ``z_t``/``a_embed`` input columns, while
**zero-initialising the new text columns**.  The consequence is a clean
invariant: immediately after initialisation the language model reproduces the
action-only M1 predictor *exactly* for any caption, so training only has to
learn what language *adds* on top of the action signal.  The text columns and
the projection are both trainable, so the model can then learn to use text.

The frozen CLIP encoder is deliberately **not** registered as a submodule:
it is a fixed feature extractor, so it stays out of ``parameters()`` /
``state_dict()`` (the predictor checkpoint holds only the trainable projection
and MLP).  ``forward`` consumes a *precomputed* CLIP text embedding, so the
model and the per-bucket evaluation run without a CLIP download; call
:meth:`load_text_encoder` / :meth:`encode_text` to produce embeddings from raw
captions when the open_clip weights are available.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from config import CanonicalConfig, load_canonical

# Architectural defaults; mirror configs/canonical.yaml so a model can be
# built in a unit test without reading the config (from_canonical is the
# authoritative path).
DEFAULT_Z_DIM = 384
DEFAULT_A_DIM = 384
DEFAULT_TEXT_DIM = 384
DEFAULT_HORIZON = 4
DEFAULT_HIDDEN = 512
DEFAULT_CLIP_TEXT_DIM = 512  # CLIP ViT-B/32 text embedding width

# open_clip identifiers for the frozen text encoder.  Use the *quickgelu*
# variant: open_clip's plain ``ViT-B-32`` config defaults to standard GELU,
# but the OpenAI weights were trained with QuickGELU, so loading them into the
# plain config shifts every embedding.  This matches ``encoders/clip_enc.py``
# (``MODEL_NAME = "ViT-B-32-quickgelu"``) so text and image towers agree.
CLIP_MODEL_ID = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"


class LanguageConditionedLatentPredictor(nn.Module):
    """Action + language conditioned latent predictor (1152-d input MLP).

    Parameters
    ----------
    z_dim, a_dim, text_dim
        Widths of the current-latent, action-embedding, and projected-text
        components of the input (default 384 each -> 1152-d input).
    horizon
        Number of future latent states to predict (default 4).
    hidden
        Hidden width of the predictor MLP (default 512).
    clip_text_dim
        Native CLIP text-embedding width fed to the projection (default 512).
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        a_dim: int = DEFAULT_A_DIM,
        text_dim: int = DEFAULT_TEXT_DIM,
        horizon: int = DEFAULT_HORIZON,
        hidden: int = DEFAULT_HIDDEN,
        clip_text_dim: int = DEFAULT_CLIP_TEXT_DIM,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.text_dim = int(text_dim)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.clip_text_dim = int(clip_text_dim)

        # Trainable projection: frozen CLIP text embedding -> latent space.
        self.text_proj = nn.Linear(self.clip_text_dim, self.text_dim, bias=False)

        # Predictor MLP. First layer takes z_t + a_embed + projected-text.
        in_dim = self.z_dim + self.a_dim + self.text_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden),  # 1152 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),  # 512 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.z_dim * self.horizon),  # 512 -> 1536
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_canonical(
        cls, cfg: Optional[CanonicalConfig] = None
    ) -> "LanguageConditionedLatentPredictor":
        """Construct from ``configs/canonical.yaml`` (same dims as M1 + CLIP).

        Weights are standard-initialised; the canonical workflow is to call
        :meth:`init_shared_layers_from` with a trained M1 predictor right
        after construction so training starts from the action-only solution.
        """
        if cfg is None:
            cfg = load_canonical()
        lp_cfg = cfg.latent_predictor()
        fae_cfg = lp_cfg["fourier_action_embed"]
        clip_dim = int(cfg.encoder("clip_b32")["output_dim_native"])
        return cls(
            z_dim=cfg.target_embedding_dim,
            a_dim=int(fae_cfg["out_dim"]),
            text_dim=cfg.target_embedding_dim,
            horizon=int(lp_cfg["prediction_horizon"]),
            hidden=DEFAULT_HIDDEN,
            clip_text_dim=clip_dim,
        )

    def init_shared_layers_from(self, m1: nn.Module) -> None:
        """Initialise the shared layers from a trained M1 ``LatentPredictor``.

        Copies the two hidden layers verbatim and M1's first-layer weights
        into the ``z_t``/``a_embed`` columns, zero-initialising the new text
        columns.  After this call the language model reproduces ``m1`` exactly
        for any text input (see module docstring).

        Training-dynamics note: because the text columns start at zero,
        ``text_proj`` receives zero gradient on the very first optimisation
        step (its gradient flows through those columns).  The columns
        themselves get a non-zero gradient immediately (``text_proj`` output
        is non-zero at standard init), so the text pathway starts learning on
        step 1 and ``text_proj`` from step 2 -- a one-step warm-up, not a
        dead branch.

        Raises
        ------
        ValueError
            If ``m1``'s ``(z_dim, a_dim, horizon, hidden)`` do not match this
            model's, so a mismatched checkpoint can't be silently mis-copied.
        """
        for attr in ("z_dim", "a_dim", "horizon", "hidden"):
            if getattr(m1, attr, None) != getattr(self, attr):
                raise ValueError(
                    f"cannot init from M1: {attr} mismatch "
                    f"(m1={getattr(m1, attr, None)!r}, self={getattr(self, attr)!r})"
                )

        shared_in = self.z_dim + self.a_dim
        with torch.no_grad():
            # First layer: shared (z_t|a_embed) columns from M1, text columns 0.
            self.net[0].weight.zero_()
            self.net[0].weight[:, :shared_in].copy_(m1.net[0].weight)
            self.net[0].bias.copy_(m1.net[0].bias)
            # Hidden layers are identical in shape -> copy verbatim.
            self.net[2].weight.copy_(m1.net[2].weight)
            self.net[2].bias.copy_(m1.net[2].bias)
            self.net[4].weight.copy_(m1.net[4].weight)
            self.net[4].bias.copy_(m1.net[4].bias)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        z_t: torch.Tensor,
        a_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future latents from state + action + (CLIP) text embedding.

        Parameters
        ----------
        z_t
            ``(B, z_dim)`` current-frame latent.
        a_embed
            ``(B, a_dim)`` Fourier action embedding (zeroed for the
            action-unconditional variant, as in M1).
        text_embed
            ``(B, clip_text_dim)`` *raw* CLIP text embedding (output of
            :meth:`encode_text`).  Projected to ``text_dim`` internally.

        Returns
        -------
        torch.Tensor
            ``(B, horizon, z_dim)`` predicted future latents.

        Raises
        ------
        ValueError
            If ``text_embed``'s last dim is not ``clip_text_dim`` -- catches
            passing the already-projected ``text_dim``-d vector (the shapes
            would otherwise only clash inside ``text_proj`` at runtime).
        """
        if text_embed.shape[-1] != self.clip_text_dim:
            raise ValueError(
                f"text_embed has last dim {text_embed.shape[-1]}, expected the "
                f"raw CLIP width clip_text_dim={self.clip_text_dim}. forward() "
                f"consumes the *raw* encode_text() output and applies text_proj "
                f"internally; do not pass the projected {self.text_dim}-d vector."
            )
        text_proj = self.text_proj(text_embed)  # (B, text_dim)
        x = torch.cat([z_t, a_embed, text_proj], dim=-1)  # (B, 1152)
        return self.net(x).view(-1, self.horizon, self.z_dim)

    # -- frozen CLIP text encoder (lazily loaded) --------------------------

    @property
    def text_encoder(self) -> Any:
        """The frozen CLIP model, or raise if it has not been loaded."""
        clip = self.__dict__.get("_clip_model")
        if clip is None:
            raise RuntimeError(
                "CLIP text encoder not loaded; call load_text_encoder() first."
            )
        return clip

    def load_text_encoder(self, device: str | torch.device = "cpu") -> None:
        """Load + freeze the CLIP ViT-B/32 text encoder via ``open_clip``.

        Stored outside ``nn.Module`` registration so it stays out of
        ``parameters()`` / ``state_dict()`` (predictor checkpoints exclude the
        ~600 MB of frozen CLIP weights; cf. ``encoders.base`` wrappers, where
        the backbone *is* the model and is registered).  Consequence: the
        predictor's ``.to()`` / ``.cuda()`` deliberately do **not** move CLIP
        -- pass ``device=`` here to co-locate it; :meth:`encode_text` always
        returns on the predictor's device regardless.  Downloads the
        open_clip weights on first use (network required).
        """
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_ID, pretrained=CLIP_PRETRAINED
        )
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        # Bypass nn.Module.__setattr__ so CLIP is NOT registered as a submodule.
        self.__dict__["_clip_model"] = model
        self.__dict__["_tokenizer"] = open_clip.get_tokenizer(CLIP_MODEL_ID)
        self.__dict__["_clip_device"] = torch.device(device)

    @torch.no_grad()
    def encode_text(self, captions: list[str]) -> torch.Tensor:
        """Encode captions to raw ``(B, clip_text_dim)`` CLIP text embeddings.

        The encoder is frozen; the trainable part is :attr:`text_proj`, applied
        inside :meth:`forward`.
        """
        clip = self.text_encoder  # raises if not loaded
        tokenizer = self.__dict__["_tokenizer"]
        clip_device = self.__dict__.get("_clip_device", torch.device("cpu"))
        tokens = tokenizer(list(captions)).to(clip_device)
        feats = clip.encode_text(tokens)
        # Land on the predictor's device so forward() can consume the result
        # directly even when CLIP lives elsewhere (CLIP is intentionally not
        # moved by the predictor's .to()/.cuda() -- see load_text_encoder).
        return feats.float().to(next(self.parameters()).device)
