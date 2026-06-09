"""Unit tests for :mod:`models.lang_latent_pred` (P2 language conditioning).

Covers:

* Architecture: 1152-d input (z_t 384 + action 384 + projected text 384),
  the trainable ``Linear(512, 384, bias=False)`` text projection, output
  reshaped to ``(B, horizon, z_dim)``.
* ``init_shared_layers_from``: the key invariant that, immediately after
  initialising from an M1 :class:`~models.latent_pred.LatentPredictor`,
  the language model **exactly reproduces** the action-only predictor for
  any text input (the new text columns are zero-initialised), and that the
  shared hidden layers are copied verbatim.
* The text pathway is actually wired (once the text columns are non-zero,
  the output depends on the text embedding).
* ``parameters()`` excludes the frozen CLIP encoder; ``encode_text``
  errors clearly before the encoder is loaded.

A real CLIP load is exercised in one test that *skips* when the open_clip
weights can't be fetched (offline), so the suite runs without network.
"""

from __future__ import annotations

import pytest
import torch

from models.lang_latent_pred import LanguageConditionedLatentPredictor
from models.latent_pred import LatentPredictor

Z_DIM, A_DIM, TEXT_DIM, HORIZON, HIDDEN, CLIP_DIM = 384, 384, 384, 4, 512, 512


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20260603)
    yield


def _make_model() -> LanguageConditionedLatentPredictor:
    return LanguageConditionedLatentPredictor(
        z_dim=Z_DIM,
        a_dim=A_DIM,
        text_dim=TEXT_DIM,
        horizon=HORIZON,
        hidden=HIDDEN,
        clip_text_dim=CLIP_DIM,
    )


def _make_m1() -> LatentPredictor:
    return LatentPredictor(z_dim=Z_DIM, a_dim=A_DIM, horizon=HORIZON, hidden=HIDDEN)


def _inputs(batch: int = 8):
    z_t = torch.randn(batch, Z_DIM)
    a_embed = torch.randn(batch, A_DIM)
    text_embed = torch.randn(batch, CLIP_DIM)
    return z_t, a_embed, text_embed


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def test_forward_output_shape():
    model = _make_model()
    z_t, a_embed, text_embed = _inputs(batch=5)
    out = model(z_t, a_embed, text_embed)
    assert out.shape == (5, HORIZON, Z_DIM)


def test_forward_rejects_wrong_width_text_embed():
    # forward expects the *raw* CLIP embedding (clip_text_dim) and projects
    # internally; passing the already-projected text_dim vector (or any wrong
    # width) must fail fast with a clear error, not a cryptic matmul error
    # inside text_proj.
    model = _make_model()
    z_t, a_embed, _ = _inputs(batch=3)
    projected = torch.randn(3, TEXT_DIM)  # 384-d: the easy mistake to make
    with pytest.raises(ValueError, match="clip_text_dim"):
        model(z_t, a_embed, projected)


def test_predictor_first_layer_input_is_1152():
    model = _make_model()
    first_linear = model.net[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == Z_DIM + A_DIM + TEXT_DIM == 1152
    assert first_linear.out_features == HIDDEN


def test_predictor_output_layer_is_z_dim_times_horizon():
    model = _make_model()
    assert model.net[-1].out_features == Z_DIM * HORIZON


def test_text_projection_is_linear_512_to_384_without_bias():
    model = _make_model()
    assert isinstance(model.text_proj, torch.nn.Linear)
    assert model.text_proj.in_features == CLIP_DIM == 512
    assert model.text_proj.out_features == TEXT_DIM == 384
    assert model.text_proj.bias is None


def test_parameters_exclude_unloaded_clip_encoder():
    model = _make_model()
    names = [n for n, _ in model.named_parameters()]
    assert names, "model should expose trainable parameters"
    # Only the trainable projection + predictor MLP are parameters; the
    # (frozen, lazily-loaded) CLIP encoder is not part of the predictor.
    assert all(n.startswith("text_proj") or n.startswith("net") for n in names)


# ---------------------------------------------------------------------------
# init_shared_layers_from: the M1 reproduction invariant
# ---------------------------------------------------------------------------


def test_init_reproduces_m1_for_any_text():
    m1 = _make_m1()
    model = _make_model()
    model.init_shared_layers_from(m1)

    z_t, a_embed, text_embed = _inputs()
    with torch.no_grad():
        out_lang = model(z_t, a_embed, text_embed)
        out_m1 = m1(z_t, a_embed)
    assert torch.allclose(out_lang, out_m1, atol=1e-6), (
        "at init the language model must exactly reproduce the action-only "
        "M1 predictor (text columns are zero-initialised)"
    )


def test_init_output_is_text_independent():
    m1 = _make_m1()
    model = _make_model()
    model.init_shared_layers_from(m1)

    z_t, a_embed, _ = _inputs()
    text_a = torch.randn(z_t.shape[0], CLIP_DIM)
    text_b = torch.randn(z_t.shape[0], CLIP_DIM) * 7.0 + 3.0
    with torch.no_grad():
        out_a = model(z_t, a_embed, text_a)
        out_b = model(z_t, a_embed, text_b)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_init_copies_hidden_layers_verbatim():
    m1 = _make_m1()
    model = _make_model()
    model.init_shared_layers_from(m1)
    # net[2] and net[4] have identical shapes in both models -> copied as-is.
    assert torch.equal(model.net[2].weight, m1.net[2].weight)
    assert torch.equal(model.net[2].bias, m1.net[2].bias)
    assert torch.equal(model.net[4].weight, m1.net[4].weight)
    assert torch.equal(model.net[4].bias, m1.net[4].bias)


def test_init_layer0_shares_za_columns_and_zeros_text_columns():
    m1 = _make_m1()
    model = _make_model()
    model.init_shared_layers_from(m1)
    w = model.net[0].weight
    shared = Z_DIM + A_DIM  # 768
    assert torch.equal(w[:, :shared], m1.net[0].weight)
    assert torch.count_nonzero(w[:, shared:]) == 0
    assert torch.equal(model.net[0].bias, m1.net[0].bias)


def test_text_pathway_is_wired_once_columns_are_nonzero():
    m1 = _make_m1()
    model = _make_model()
    model.init_shared_layers_from(m1)

    # Simulate training having moved the text columns off zero.
    with torch.no_grad():
        model.net[0].weight[:, Z_DIM + A_DIM :].normal_(0.0, 0.1)

    z_t, a_embed, _ = _inputs()
    text_a = torch.randn(z_t.shape[0], CLIP_DIM)
    text_b = torch.randn(z_t.shape[0], CLIP_DIM)
    with torch.no_grad():
        out_a = model(z_t, a_embed, text_a)
        out_b = model(z_t, a_embed, text_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "once the text columns are non-zero the output must depend on text"
    )


def test_init_rejects_dimension_mismatch():
    model = _make_model()
    bad_m1 = LatentPredictor(z_dim=256, a_dim=A_DIM, horizon=HORIZON, hidden=HIDDEN)
    with pytest.raises(ValueError):
        model.init_shared_layers_from(bad_m1)


# ---------------------------------------------------------------------------
# from_canonical
# ---------------------------------------------------------------------------


def test_from_canonical_dims(cfg):
    model = LanguageConditionedLatentPredictor.from_canonical(cfg)
    assert model.net[0].in_features == 1152
    assert model.net[-1].out_features == cfg.target_embedding_dim * 4
    assert model.text_proj.out_features == cfg.target_embedding_dim


# ---------------------------------------------------------------------------
# CLIP text encoder (frozen); real-weights test skips offline
# ---------------------------------------------------------------------------


def test_exported_from_models_package():
    import models

    assert models.LanguageConditionedLatentPredictor is (
        LanguageConditionedLatentPredictor
    )
    assert "LanguageConditionedLatentPredictor" in models.__all__


def test_encode_text_errors_before_encoder_loaded():
    model = _make_model()
    with pytest.raises(RuntimeError, match="text encoder"):
        model.encode_text(["a rainy night drive"])


def test_encode_text_returns_float32_on_predictor_device():
    # Contract: encode_text output always lands on the predictor's own device
    # (and float32), even though the frozen CLIP encoder is deliberately NOT
    # moved by ``.to()``/``.cuda()`` (it lives outside nn.Module registration
    # so predictor checkpoints exclude its weights).
    model = _make_model()

    class _FakeClip:
        def encode_text(self, tokens):
            return torch.ones(tokens.shape[0], CLIP_DIM, dtype=torch.float16)

    model.__dict__["_clip_model"] = _FakeClip()
    model.__dict__["_tokenizer"] = lambda caps: torch.zeros(
        len(caps), 77, dtype=torch.long
    )
    model.__dict__["_clip_device"] = torch.device("cpu")

    emb = model.encode_text(["a", "b"])
    assert emb.shape == (2, CLIP_DIM)
    assert emb.dtype == torch.float32
    assert emb.device == next(model.parameters()).device


def test_load_clip_text_encoder_is_frozen_and_encodes():
    model = _make_model()
    try:
        model.load_text_encoder()
    except Exception as exc:  # pragma: no cover - network/offline dependent
        pytest.skip(f"open_clip CLIP weights unavailable offline: {exc}")

    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    emb = model.encode_text(["a rainy night drive", "clear daytime highway"])
    assert emb.shape == (2, CLIP_DIM)
    assert emb.dtype == torch.float32


def test_text_encoder_uses_quickgelu_matching_openai_weights():
    # The OpenAI CLIP weights were trained with QuickGELU; loading them into
    # the plain ViT-B-32 (standard GELU) config silently shifts every
    # embedding. open_clip warns on that mismatch -- assert there is none.
    import warnings

    model = _make_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.load_text_encoder()
        except Exception as exc:  # pragma: no cover - network/offline dependent
            pytest.skip(f"open_clip CLIP weights unavailable offline: {exc}")
    assert not any("QuickGELU" in str(w.message) for w in caught), (
        "CLIP text encoder must use the quickgelu variant to match OpenAI weights"
    )
