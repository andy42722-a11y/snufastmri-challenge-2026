"""Deterministic synthetic Cartesian masks for Round 6 training."""

import hashlib

import numpy as np


ACS_MIN = 26
ACS_MAX = 33


def sample_key(seed, epoch, fname, slice_idx):
    """Return the P8/S11 SHA-256 digest for one training sample and epoch."""
    payload = f"{seed}|{epoch}|{fname}|{slice_idx}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def make_mask(width, rng, period=None, offset=None, acs_width=None):
    """Build ACS union periodic-grid mask as a float32 vector.

    Explicit parameters are supported for exact reference-mask validation.
    During training, all three parameters are drawn uniformly.
    """
    width = int(width)
    if period is None:
        period = (4, 8)[int(rng.integers(0, 2))]
    if period not in (4, 8):
        raise ValueError(f"period must be 4 or 8, got {period}")
    if offset is None:
        offset = int(rng.integers(0, period))
    if not 0 <= int(offset) < period:
        raise ValueError(f"offset {offset} outside [0, {period})")
    if acs_width is None:
        acs_width = int(rng.integers(ACS_MIN, ACS_MAX + 1))
    if not 0 < int(acs_width) <= width:
        raise ValueError(f"invalid ACS width {acs_width} for W={width}")

    mask = np.zeros(width, dtype=np.float32)
    columns = np.arange(width)
    mask[columns % period == int(offset)] = 1.0
    center = width // 2
    start = center - int(acs_width) // 2
    end = start + int(acs_width)
    mask[start:end] = 1.0
    return mask


def deterministic_mask(width, seed, epoch, fname, slice_idx):
    """Generate one mask without using global RNG state."""
    digest = sample_key(seed, epoch, fname, slice_idx)
    # Preserve the exact P8/S11 binary selector for period choice.
    period = (4, 8)[digest[0] & 1]
    rng_seed = int.from_bytes(digest[1:9], byteorder="big", signed=False)
    rng = np.random.default_rng(rng_seed)
    return make_mask(width, rng, period=period)


def mask_parameters(width, seed, epoch, fname, slice_idx):
    """Return deterministic parameters and mask for audit/tests."""
    digest = sample_key(seed, epoch, fname, slice_idx)
    period = (4, 8)[digest[0] & 1]
    rng_seed = int.from_bytes(digest[1:9], byteorder="big", signed=False)
    rng = np.random.default_rng(rng_seed)
    offset = int(rng.integers(0, period))
    acs_width = int(rng.integers(ACS_MIN, ACS_MAX + 1))
    return period, offset, acs_width, make_mask(
        width, rng, period=period, offset=offset, acs_width=acs_width
    )
