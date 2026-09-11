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

from shine.morphology.render import (
    detector_array_profile,
    render_decoded_galaxy,
    render_learned_galaxy,
)

STAMP = 8
SCALE = 0.1

# Local WCS Jacobian of quadrant 3-4.F: a +57.9 deg rotation, slightly
# anisotropic (0.0996 / 0.1006 "/px).
ROTATED_WCS = (0.05306, -0.08507, 0.08433, 0.05363)


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

    def test_matches_render_decoded_galaxy(self):
        z = _random_z(jax.random.key(6))
        common = (0.03, -0.02, _centred_psf(), jnp.array(ROTATED_WCS), 0.2, -0.1, True)
        from_z = render_learned_galaxy(z, *common, _ae(), STAMP, SCALE, _gsparams())
        from_img = render_decoded_galaxy(
            _ae().decode(z)[0], *common, STAMP, SCALE, _gsparams()
        )
        np.testing.assert_allclose(np.asarray(from_z), np.asarray(from_img), atol=1e-6)


def _centred_psf():
    # Odd-sized, so the delta sits exactly on the true centre (like the
    # real 21x21 PSF stamps).
    return jnp.zeros((5, 5)).at[2, 2].set(1.0)


def _elongated_galaxy(n=32):
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    u, v = (x - c) * 0.9 + (y - c) * 0.4, -(x - c) * 0.4 + (y - c) * 0.9
    img = np.exp(-0.5 * (u ** 2 / 9.0 + v ** 2 / 1.5))
    return jnp.asarray(500.0 * img / img.sum(), dtype=jnp.float32)


class TestDetectorFrame:
    """The decoded image and the PSF live on the detector grid: the WCS
    rotation must not rotate them (#4/#5 of the integration notes)."""

    N = 32

    def _render(self, wcs, g1=0.0, g2=0.0):
        gsp = galsim.GSParams(minimum_fft_size=64, maximum_fft_size=64)
        return np.asarray(render_decoded_galaxy(
            _elongated_galaxy(self.N), g1, g2, _centred_psf(), jnp.array(wcs),
            0.0, 0.0, True, self.N, SCALE, gsp,
        ))

    def test_detector_array_round_trips(self):
        gsp = galsim.GSParams(minimum_fft_size=64, maximum_fft_size=64)
        img = _elongated_galaxy(self.N)
        wcs = galsim.JacobianWCS(*ROTATED_WCS)
        drawn = detector_array_profile(img, wcs, gsp).drawImage(
            nx=self.N, ny=self.N, wcs=wcs, method="no_pixel"
        ).array
        np.testing.assert_allclose(np.asarray(drawn), np.asarray(img), atol=1e-4)

    def test_wcs_rotation_does_not_rotate_the_stamp(self):
        rotated = self._render(ROTATED_WCS)
        aligned = self._render((SCALE, 0.0, 0.0, SCALE))
        assert np.abs(rotated - aligned).max() < 1e-2 * aligned.max()

    def test_flux_is_conserved(self):
        stamp = self._render(ROTATED_WCS)
        np.testing.assert_allclose(stamp.sum(), 500.0, rtol=1e-3)

    def test_shear_is_applied_on_the_sky(self):
        # A sky-frame shear seen through a detector rotated by theta is a
        # detector-frame shear rotated by -2 theta: with an isotropic
        # galaxy, g on the sky must give the same stamp as the rotated
        # shear drawn through an aligned WCS.
        g1, g2 = 0.1, -0.05
        theta = np.arctan2(ROTATED_WCS[2], ROTATED_WCS[0])
        c, s = np.cos(2 * theta), np.sin(2 * theta)
        g1d, g2d = g1 * c + g2 * s, -g1 * s + g2 * c
        rotated_wcs = jnp.array(
            [SCALE * np.cos(theta), -SCALE * np.sin(theta),
             SCALE * np.sin(theta), SCALE * np.cos(theta)]
        )
        gsp = galsim.GSParams(minimum_fft_size=64, maximum_fft_size=64)
        y, x = np.mgrid[0:self.N, 0:self.N]
        c0 = (self.N - 1) / 2
        round_gal = jnp.asarray(np.exp(-0.5 * ((x - c0) ** 2 + (y - c0) ** 2) / 4.0),
                                dtype=jnp.float32)
        on_sky = render_decoded_galaxy(
            round_gal, g1, g2, _centred_psf(), rotated_wcs,
            0.0, 0.0, True, self.N, SCALE, gsp,
        )
        on_detector = render_decoded_galaxy(
            round_gal, g1d, g2d, _centred_psf(), jnp.array([SCALE, 0.0, 0.0, SCALE]),
            0.0, 0.0, True, self.N, SCALE, gsp,
        )
        on_sky, on_detector = np.asarray(on_sky), np.asarray(on_detector)
        assert np.abs(on_sky - on_detector).max() < 1e-2 * on_sky.max()
