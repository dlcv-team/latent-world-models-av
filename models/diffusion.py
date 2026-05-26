"""Diffusion noise schedule and DDIM sampler for the Latent DiT (Tier 2).

Provides the forward diffusion process (adding noise) and the reverse
sampling process (denoising) used to train and run the
:class:`~models.latent_dit.LatentDiT`.

**Components:**

- :class:`CosineNoiseSchedule` -- cosine beta schedule per Nichol &
  Dhariwal (2021, "Improved Denoising Diffusion Probabilistic Models").
  Computes cumulative products of alphas and related quantities needed
  for forward diffusion and loss computation.

- :class:`DDIMSampler` -- deterministic DDIM sampling (Song et al.,
  2020, "Denoising Diffusion Implicit Models") that produces denoised
  output in fewer steps than the full training schedule.

This module is standalone PyTorch -- it does not depend on the
``diffusers`` library.  Hyperparameters are in ``configs/dit.yaml``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch import nn


class CosineNoiseSchedule(nn.Module):
    """Cosine beta schedule for diffusion (Nichol & Dhariwal, 2021).

    Computes ``alphas_cumprod`` using a cosine schedule::

        f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
        alpha_bar_t = f(t) / f(0)

    where ``s`` is a small offset to prevent singularity at ``t = 0``.

    All derived quantities are stored as buffers (not parameters) and
    follow the module to the correct device via ``.to()``.

    Parameters
    ----------
    n_steps
        Total number of diffusion timesteps (T).  Defaults to 1000.
    s
        Cosine schedule offset.  Defaults to 0.008.
    """

    def __init__(self, n_steps: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        self.n_steps = n_steps

        # Compute alphas_cumprod via cosine schedule
        steps = torch.arange(n_steps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / n_steps) + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod = alphas_cumprod[:n_steps]  # (T,), indexed 0..T-1

        # Derive betas from consecutive alpha_bar ratios
        # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = torch.zeros(n_steps, dtype=torch.float64)
        betas[0] = 1.0 - alphas_cumprod[0]
        betas[1:] = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = betas.clamp(0.0, 0.999)

        # Convert to float32 and register
        alphas_cumprod = alphas_cumprod.float()
        betas = betas.float()

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def _extract(
        self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size
    ) -> torch.Tensor:
        """Index ``arr`` by timestep ``t`` and reshape for broadcasting.

        Parameters
        ----------
        arr
            Schedule array of shape ``(T,)``.
        t
            Integer timestep tensor of shape ``(B,)``.
        x_shape
            Shape of the data tensor to broadcast against.

        Returns
        -------
        torch.Tensor
            Values at timestep ``t``, reshaped to ``(B, 1, ..., 1)``
            for broadcasting with data of shape ``x_shape``.
        """
        out = arr.gather(0, t.long())
        # Reshape: (B,) -> (B, 1, 1, ...) matching x_shape dims
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean data.

        Computes ``x_noisy = sqrt(alpha_bar_t) * x_0 +
        sqrt(1 - alpha_bar_t) * noise``.

        Parameters
        ----------
        x_0
            Clean data of shape ``(B, ...)``.
        t
            Integer timestep tensor of shape ``(B,)``.
        noise
            Optional pre-sampled noise.  If ``None``, sampled from
            ``N(0, I)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(x_noisy, noise)`` -- the noised data and the noise that
            was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        x_noisy = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_noisy, noise


class DDIMSampler:
    """Deterministic DDIM sampler for accelerated inference.

    Constructs a subsequence of the full training schedule and performs
    deterministic reverse-process steps (eta = 0).

    Parameters
    ----------
    noise_schedule
        A :class:`CosineNoiseSchedule` instance.
    n_steps
        Number of sampling steps (must be <= training steps).
        Defaults to 50.
    """

    def __init__(
        self, noise_schedule: CosineNoiseSchedule, n_steps: int = 50
    ) -> None:
        self.schedule = noise_schedule
        self.n_steps = n_steps
        T = noise_schedule.n_steps

        # Build the subsequence of timesteps
        # e.g., T=1000, n_steps=50 -> stride=20 -> [0, 20, 40, ..., 980]
        stride = T // n_steps
        self.timesteps = list(range(0, T, stride))[:n_steps]
        # Reverse for sampling (high noise to low noise)
        self.timesteps = list(reversed(self.timesteps))

    @torch.no_grad()
    def sample(
        self,
        noise_pred_fn: Callable[..., torch.Tensor],
        shape: tuple[int, ...],
        cond_kwargs: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Run the full DDIM sampling loop.

        Parameters
        ----------
        noise_pred_fn
            A callable with signature
            ``fn(x_noisy, timestep=t, **cond_kwargs) -> noise_pred``.
            Typically the :class:`~models.latent_dit.LatentDiT` model.
        shape
            Shape of the output tensor, e.g., ``(B, horizon, z_dim)``.
        cond_kwargs
            Keyword arguments passed through to ``noise_pred_fn``
            (e.g., ``z_t``, ``a_embed``).
        device
            Device to run sampling on.

        Returns
        -------
        torch.Tensor
            Denoised output of the requested ``shape``.
        """
        alphas_cumprod = self.schedule.alphas_cumprod.to(device)

        # Start from pure Gaussian noise
        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(self.timesteps):
            t = torch.full(
                (shape[0],), t_val, device=device, dtype=torch.long
            )

            # Predict noise
            noise_pred = noise_pred_fn(x, timestep=t, **cond_kwargs)

            # Current alpha_bar
            alpha_bar_t = alphas_cumprod[t_val]

            # Predict x_0 from x_t and noise prediction
            pred_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_bar_t)

            # Get alpha_bar for the next (previous in time) step
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = alphas_cumprod[t_prev]
            else:
                # Last step: go to t=0 (alpha_bar = 1, i.e., clean data)
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # DDIM deterministic step (eta = 0)
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0
            #         + sqrt(1 - alpha_bar_{t-1}) * noise_direction
            noise_direction = (
                x - torch.sqrt(alpha_bar_t) * pred_x0
            ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)

            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
            )

        return x
