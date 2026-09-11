"""Rendering of learned (AutoEncoder-decoded) galaxy stamps.

Mirrors ``render_one_galaxy`` in ``shine.euclid.scene`` — same WCS
Jacobian, sub-pixel offset, and invisible-source-masking conventions —
but replaces ``galsim.Exponential(flux, hlr).shear(e1, e2)`` with the
frozen AE's decoded image. Deliberately does not reuse
``GalaxyAutoEncoder.convolve()``: that method assumes a fixed pixel WCS
and no sub-pixel offset, whereas SHINE needs the real per-source,
per-exposure PSF/WCS/offset handled by the existing rendering pipeline.

Frame convention. The AE was trained on plain cutouts of the science
image, so ``ae.decode(z)`` is an array on the **detector pixel grid**, and
so is every PSF stamp. :func:`detector_array_profile` turns such an array
into a sky-plane profile *through the local WCS Jacobian*, so that drawing
it back through the same Jacobian reproduces the array — whatever rotation
the Jacobian carries (+57.9 deg on quadrant 3-4.F). Wrapping it as
``Image(array, scale=pixel_scale)`` instead declares it to already be a sky
profile, and the drawn stamp comes out rotated by the WCS angle. The shear
is applied in between, on the sky, where ``(g1, g2)`` are defined.
"""

import jax.numpy as jnp
import jax_galsim as galsim

from .galaxy_ae import GalaxyAutoEncoder


def detector_array_profile(
    array: jnp.ndarray, wcs: galsim.JacobianWCS, gsparams: galsim.GSParams
) -> galsim.InterpolatedImage:
    """Wrap a detector-grid array as a sky-plane profile.

    The array's pixels are mapped onto the sky with the local WCS
    Jacobian, so ``profile.drawImage(nx, ny, wcs=wcs, method="no_pixel")``
    returns the array unchanged (flux included), whatever rotation or
    anisotropy ``wcs`` carries.

    Args:
        array: 2-D array sampled on the detector pixel grid (a decoded
            galaxy or a PSF stamp).
        wcs: Local WCS Jacobian at the source position.
        gsparams: GalSim FFT parameters for this stamp tier.

    Returns:
        The array as an ``InterpolatedImage`` in sky coordinates.
    """
    return galsim.InterpolatedImage(galsim.Image(array, wcs=wcs), gsparams=gsparams)


def render_decoded_galaxy(
    galaxy_img: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    psf_img: jnp.ndarray,
    wcs_params: jnp.ndarray,
    offset_x: jnp.ndarray,
    offset_y: jnp.ndarray,
    visible_i: jnp.ndarray,
    stamp_size: int,
    pixel_scale: float,
    gsparams: galsim.GSParams,
) -> jnp.ndarray:
    """Render one galaxy stamp from an already-decoded AE image.

    Split out of :func:`render_learned_galaxy` so the scene model can
    decode each latent code once and reuse the image for every exposure
    (the decoded image does not depend on the exposure; only the PSF, WCS
    and offset do).

    Drawn with ``method="no_pixel"``: the decoded image is already a
    pixelated image (the AE was trained against data with that exact
    convention), so an extra pixel-response convolution would count the
    pixel twice.

    Args:
        galaxy_img: Decoded galaxy, shape ``(stamp_size, stamp_size)``, on
            the detector pixel grid.
        g1: Global shear component 1 (scalar), in sky coordinates.
        g2: Global shear component 2 (scalar), in sky coordinates.
        psf_img: **Residual**-PSF postage stamp for this source/exposure
            (see ``shine.morphology.psf_residual``), not the full local
            PSF — the decoded image still contains the fixed reference
            PSF baked in from training.
        wcs_params: Local WCS Jacobian ``(dudx, dudy, dvdx, dvdy)``.
        offset_x: Position of the source relative to the stamp's true
            centre, in detector pixels (x).
        offset_y: Same, along y.
        visible_i: Boolean (or 0/1) visibility flag for this source in
            this exposure. Invisible sources are rendered with safe
            fallback inputs and then zeroed out, to keep shapes static
            under ``vmap``.
        stamp_size: Output stamp side length in pixels.
        pixel_scale: Pixel scale in arcsec/pixel (used for the fallback
            WCS of invisible sources).
        gsparams: GalSim FFT parameters for this stamp tier.

    Returns:
        Rendered stamp array of shape ``(stamp_size, stamp_size)``,
        zeroed out where ``visible_i`` is false.
    """
    v = visible_i
    offset_x = jnp.where(v, offset_x, 0.0)
    offset_y = jnp.where(v, offset_y, 0.0)

    gal_shape = galaxy_img.shape[-2:]
    safe_gal = jnp.zeros(gal_shape).at[gal_shape[0] // 2, gal_shape[1] // 2].set(1.0)
    galaxy_img = jnp.where(v, galaxy_img, safe_gal)

    psf_shape = psf_img.shape[-2:]
    safe_psf = jnp.zeros(psf_shape).at[psf_shape[0] // 2, psf_shape[1] // 2].set(1.0)
    psf_img = jnp.where(v, psf_img, safe_psf)
    wcs_params = jnp.where(
        v, wcs_params, jnp.array([pixel_scale, 0.0, 0.0, pixel_scale])
    )

    wcs = galsim.JacobianWCS(
        dudx=wcs_params[0], dudy=wcs_params[1],
        dvdx=wcs_params[2], dvdy=wcs_params[3],
    )

    gal = detector_array_profile(galaxy_img, wcs, gsparams).shear(g1=g1, g2=g2)
    psf = detector_array_profile(psf_img, wcs, gsparams)
    final = galsim.Convolve([gal, psf], gsparams=gsparams)

    stamp = final.drawImage(
        nx=stamp_size, ny=stamp_size, wcs=wcs,
        offset=galsim.PositionD(offset_x, offset_y),
        method="no_pixel",
    ).array

    return stamp * visible_i


def render_learned_galaxy(
    z_i: jnp.ndarray,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    psf_img: jnp.ndarray,
    wcs_params: jnp.ndarray,
    offset_x: jnp.ndarray,
    offset_y: jnp.ndarray,
    visible_i: jnp.ndarray,
    ae: GalaxyAutoEncoder,
    stamp_size: int,
    pixel_scale: float,
    gsparams: galsim.GSParams,
) -> jnp.ndarray:
    """Render one galaxy stamp from an AE latent code.

    Decodes ``z_i`` and hands the image to :func:`render_decoded_galaxy`.
    Intended to be called from inside a ``jax.vmap`` over sources.

    Unlike the parametric path, there is no intrinsic ellipticity to
    apply separately: the decoded image ``ae.decode(z_i)`` already
    encodes the galaxy's morphology (including its intrinsic shape), so
    only the cosmic shear ``(g1, g2)`` is applied on top of it.

    Args:
        z_i: Latent code for this source, shape ``ae``'s expected input
            shape (e.g. ``(latent_channels, h, w)``).
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        psf_img: Residual-PSF postage stamp for this source/exposure (see
            :func:`render_decoded_galaxy`).
        wcs_params: Local WCS Jacobian ``(dudx, dudy, dvdx, dvdy)``.
        offset_x: Position of the source relative to the stamp's true
            centre, in detector pixels (x).
        offset_y: Same, along y.
        visible_i: Boolean (or 0/1) visibility flag for this source in
            this exposure.
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

    # ae.decode is deterministic (dropout disabled by inference_mode at
    # load time); the leading channel dim is squeezed to a plain 2D image.
    galaxy_img = ae.decode(z_i, key=None)[0]
    return render_decoded_galaxy(
        galaxy_img, g1, g2, psf_img, wcs_params, offset_x, offset_y,
        visible_i, stamp_size, pixel_scale, gsparams,
    )
