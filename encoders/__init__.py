"""Encoder wrappers for the benchmark.

Re-exports the abstract :class:`BaseEncoderWrapper`; concrete encoder
wrappers land in subsequent modules and are added here as they arrive.
"""

from encoders.base import BaseEncoderWrapper

__all__ = ["BaseEncoderWrapper"]
