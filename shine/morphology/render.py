"""Rendering of learned (AutoEncoder-decoded) galaxy stamps.

Mirrors ``render_one_galaxy`` in ``shine.euclid.scene`` — same WCS
Jacobian, sub-pixel offset, and invisible-source-masking conventions —
but replaces ``galsim.Exponential(flux, hlr).shear(e1, e2)`` with the
frozen AE's decoded image. Deliberately does not reuse
``GalaxyAutoEncoder.convolve()``: that method assumes a fixed pixel WCS
and no sub-pixel offset, whereas SHINE needs the real per-source,
per-exposure PSF/WCS/offset handled by the existing rendering pipeline.
"""

import jax.numpy as jnp
import jax_galsim as galsim

from .galaxy_ae import GalaxyAutoEncoder


def render_learned_galaxy(
    z_i: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    psf_img: jnp.ndarray,
    wcs_params: jnp.ndarray,
    dx_i: jnp.ndarray,
    dy_i: jnp.ndarray,
    visible_i: jnp.ndarray,
    ae: GalaxyAutoEncoder,
    stamp_size: int,
    pixel_scale: float,
    gsparams: galsim.GSParams,
) -> jnp.ndarray:
    """Render one galaxy stamp from an AE-decoded image.

    Intended to be called from inside a ``jax.vmap`` over sources, one
    tier at a time, exactly like ``render_one_galaxy`` in
    ``shine.euclid.scene._render_tier``.

    Unlike the parametric path, there is no intrinsic ellipticity to
    apply separately: the decoded image ``ae.decode(z_i)`` already
    encodes the galaxy's morphology (including its intrinsic shape), so
    only the cosmic shear ``(g1, g2)`` is applied on top of it.

    Args:
        z_i: Latent code for this source, shape ``ae``'s expected input
            shape (e.g. ``(latent_channels, h, w)``).
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        psf_img: PSF postage stamp for this source/exposure.
        wcs_params: Local WCS Jacobian ``(dudx, dudy, dvdx, dvdy)``.
        dx_i: Position offset in arcsec (x).
        dy_i: Position offset in arcsec (y).
        visible_i: Boolean (or 0/1) visibility flag for this source in
            this exposure. Invisible sources are rendered with safe
            fallback inputs and then zeroed out, to keep shapes static
            under ``vmap``.
        ae: Frozen :class:`~shine.morphology.galaxy_ae.GalaxyAutoEncoder`
            (already in inference mode; see
            :func:`shine.morphology.loader.load_frozen_autoencoder`).
        stamp_size: Output stamp side length in pixels. Must match the
            size the AE was trained/decoding at (``ae.nx``/``ae.ny``).
        pixel_scale: Pixel scale in arcsec/pixel. Must match ``ae.scale``,
            the scale the AE's decoded image is defined at.
        gsparams: GalSim FFT parameters for this stamp tier.

    Returns:
        Rendered stamp array of shape ``(stamp_size, stamp_size)``,
        zeroed out where ``visible_i`` is false.
    """
    assert stamp_size == ae.nx == ae.ny, (
        f"render_learned_galaxy stamp_size={stamp_size} does not match "
        f"the AE's trained stamp size ({ae.nx}x{ae.ny})"
    )
    assert abs(pixel_scale - ae.scale) < 1e-6, (
        f"render_learned_galaxy pixel_scale={pixel_scale} does not match "
        f"ae.scale={ae.scale} the AE was trained at"
    )

    v = visible_i
    z_i = jnp.where(v, z_i, jnp.zeros_like(z_i))
    dx_i = jnp.where(v, dx_i, 0.0)
    dy_i = jnp.where(v, dy_i, 0.0)

    psf_shape = psf_img.shape[-2:]
    safe_psf = jnp.zeros(psf_shape)
    safe_psf = safe_psf.at[psf_shape[0] // 2, psf_shape[1] // 2].set(1.0)
    psf_img = jnp.where(v, psf_img, safe_psf)
    wcs_params = jnp.where(
        v, wcs_params, jnp.array([pixel_scale, 0.0, 0.0, pixel_scale])
    )

    # ae.decode is deterministic (dropout disabled by inference_mode at
    # load time); the leading channel dim is squeezed to a plain 2D image.
    g = ae.decode(z_i, key=None)[0]
    gal = galsim.InterpolatedImage(
        galsim.Image(g, scale=pixel_scale), gsparams=gsparams
    )
    gal = gal.shear(g1=g1, g2=g2)

    psf = galsim.InterpolatedImage(
        galsim.Image(psf_img, scale=pixel_scale), gsparams=gsparams
    )
    final = galsim.Convolve([gal, psf], gsparams=gsparams)

    wcs = galsim.JacobianWCS(
        dudx=wcs_params[0], dudy=wcs_params[1],
        dvdx=wcs_params[2], dvdy=wcs_params[3],
    )

    pix_dx = dx_i / pixel_scale
    pix_dy = dy_i / pixel_scale

    stamp = final.drawImage(
        nx=stamp_size, ny=stamp_size, wcs=wcs,
        offset=galsim.PositionD(pix_dx, pix_dy),
    ).array

    return stamp * visible_i
