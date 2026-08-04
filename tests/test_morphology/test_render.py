"""Tests for shine.morphology.render: rendering learned galaxy stamps.

Uses a trivial stand-in "autoencoder" (latent shape == image shape,
decode = softplus) instead of a real, trained GalaxyAutoEncoder. This
exercises render_learned_galaxy's own logic (masking, shear, WCS,
vmap/jit compatibility) without depending on the ported Encoder/Decoder
architecture (already verified as a faithful port of pshear in
test_loader.py-style checkpoint tests) or on any checkpoint files.
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
import pytest
from jax.nn import softplus

from shine.morphology.render import render_learned_galaxy

STAMP = 8
SCALE = 0.1


class _StubAutoEncoder(eqx.Module):
    """Minimal decode-only stand-in for GalaxyAutoEncoder."""

    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    def decode(self, z, key=None):
        return softplus(z)


def _ae():
    return _StubAutoEncoder(nx=STAMP, ny=STAMP, scale=SCALE)


def _gsparams():
    return galsim.GSParams(minimum_fft_size=64, maximum_fft_size=64)


def _flat_psf():
    psf = jnp.zeros((STAMP, STAMP))
    return psf.at[STAMP // 2, STAMP // 2].set(1.0)


def _identity_wcs():
    return jnp.array([SCALE, 0.0, 0.0, SCALE])


def _random_z(key):
    return jax.random.normal(key, (1, STAMP, STAMP)) * 0.5


class TestRenderLearnedGalaxy:
    def test_output_shape_and_finite(self):
        stamp = render_learned_galaxy(
            _random_z(jax.random.key(0)), 0.0, 0.0,
            _flat_psf(), _identity_wcs(), 0.0, 0.0, True,
            _ae(), STAMP, SCALE, _gsparams(),
        )
        assert stamp.shape == (STAMP, STAMP)
        assert np.all(np.isfinite(np.asarray(stamp)))

    def test_invisible_source_is_zeroed(self):
        stamp = render_learned_galaxy(
            _random_z(jax.random.key(1)), 0.0, 0.0,
            _flat_psf(), _identity_wcs(), 0.0, 0.0, False,
            _ae(), STAMP, SCALE, _gsparams(),
        )
        np.testing.assert_array_equal(np.asarray(stamp), 0.0)

    def test_shear_changes_the_image(self):
        z = _random_z(jax.random.key(2))
        common = (_flat_psf(), _identity_wcs(), 0.0, 0.0, True, _ae(), STAMP, SCALE, _gsparams())

        base = render_learned_galaxy(z, 0.0, 0.0, *common)
        sheared = render_learned_galaxy(z, 0.1, 0.0, *common)

        assert not np.allclose(np.asarray(base), np.asarray(sheared))

    def test_rejects_mismatched_stamp_size(self):
        with pytest.raises(AssertionError):
            render_learned_galaxy(
                _random_z(jax.random.key(3)), 0.0, 0.0,
                _flat_psf(), _identity_wcs(), 0.0, 0.0, True,
                _ae(), STAMP + 4, SCALE, _gsparams(),
            )

    def test_rejects_mismatched_pixel_scale(self):
        with pytest.raises(AssertionError):
            render_learned_galaxy(
                _random_z(jax.random.key(4)), 0.0, 0.0,
                _flat_psf(), _identity_wcs(), 0.0, 0.0, True,
                _ae(), STAMP, SCALE * 2, _gsparams(),
            )

    def test_vmap_and_jit_compatible(self):
        n = 3
        keys = jax.random.split(jax.random.key(5), n)
        z_batch = jax.vmap(_random_z)(keys)
        psf_batch = jnp.stack([_flat_psf()] * n)
        wcs_batch = jnp.stack([_identity_wcs()] * n)
        dx_batch = jnp.zeros(n)
        dy_batch = jnp.zeros(n)
        vis_batch = jnp.ones(n, dtype=bool)

        render_fn = partial(
            render_learned_galaxy,
            ae=_ae(), stamp_size=STAMP, pixel_scale=SCALE, gsparams=_gsparams(),
        )
        stamps = jax.jit(
            jax.vmap(render_fn, in_axes=(0, None, None, 0, 0, 0, 0, 0))
        )(z_batch, 0.02, 0.0, psf_batch, wcs_batch, dx_batch, dy_batch, vis_batch)

        assert stamps.shape == (n, STAMP, STAMP)
        assert np.all(np.isfinite(np.asarray(stamps)))
