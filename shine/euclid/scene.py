"""Multi-exposure multi-object scene model for Euclid VIS inference.

Builds a NumPyro forward generative model that samples global shear and
per-galaxy parameters, renders each galaxy through per-exposure PSFs and
WCS, scatter-adds the stamps onto full-sized model images, and evaluates
a per-pixel Gaussian likelihood weighted by the RMS noise map.

Sources are grouped into stamp-size tiers (e.g. 64, 128, 256 px) based
on their catalog half-light radius.  Each tier is rendered with its own
``jax.vmap`` pass and FFT size, preserving full parallelism within each
tier while avoiding expensive FFT convolutions for small galaxies.

Frame and placement conventions (both tiers):

- PSF stamps (and decoded galaxies on the learned tier) are arrays on the
  detector pixel grid; they are mapped onto the sky through the local WCS
  Jacobian (:func:`~shine.morphology.render.detector_array_profile`), so
  the WCS rotation does not rotate them. Shear and intrinsic ellipticity
  are applied on the sky.
- Each stamp is pasted at an integer corner, and the source is drawn at
  its exact catalog position inside it (:func:`stamp_placement`): the
  sub-pixel part of the position and GalSim's half-pixel true centre on
  even stamps are both accounted for, ``dx``/``dy`` being free
  corrections on top.
"""

import logging
import math
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax_galsim as galsim
import numpy as np
import numpyro
import numpyro.distributions as dist

from shine.euclid.config import EuclidInferenceConfig
from shine.morphology.galaxy_ae import GalaxyAutoEncoder
from shine.morphology.loader import load_frozen_autoencoder, load_frozen_flow
from shine.morphology.nn.flow import LatentFlow
from shine.morphology.prior import sample_latent_codes
from shine.morphology.render import detector_array_profile, render_decoded_galaxy
from shine.prior_utils import parse_prior

logger = logging.getLogger(__name__)


def _fft_size_for_stamp(stamp_size: int) -> int:
    """Return the FFT grid size (next power of 2 >= 2 * stamp_size)."""
    return int(2 ** math.ceil(math.log2(2 * stamp_size)))


def _draw_method(psf_includes_pixel: bool) -> str:
    """GalSim ``drawImage`` method for the parametric tiers.

    An empirical PSF sampled at the native pixel scale (the Euclid VIS
    grid is 21x21 at 0.1"/px) already contains the pixel response, so
    ``"no_pixel"`` avoids convolving by the pixel a second time; this is
    also the convention the learned tier's AE was trained with.
    """
    return "no_pixel" if psf_includes_pixel else "auto"


def stamp_placement(
    positions: jnp.ndarray, stamp_size: int, image_nx: int, image_ny: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integer stamp corners and in-stamp offsets for sources.

    The stamp is the ``stamp_size`` square whose corner is
    ``round(position) - stamp_size // 2`` (clipped to the image), i.e. the
    same pixels a plain ``image[y0:y0+n, x0:x0+n]`` cutout centred on the
    source would take. The offset is the source position relative to the
    stamp's GalSim true centre, ``(stamp_size - 1) / 2`` in 0-indexed
    pixels — a half-integer on even stamps — so that drawing with
    ``offset=(offset_x, offset_y)`` puts the profile exactly at
    ``positions``.

    Args:
        positions: 0-indexed pixel positions, shape ``(..., 2)`` as
            ``(x, y)``.
        stamp_size: Stamp side length in pixels.
        image_nx: Image width in pixels (for clipping).
        image_ny: Image height in pixels (for clipping).

    Returns:
        Tuple ``(corner_x, corner_y, offset_x, offset_y)``; corners are
        int32, offsets are in pixels.
    """
    half = stamp_size // 2
    corner_x = jnp.clip(
        jnp.round(positions[..., 0]).astype(jnp.int32) - half,
        0, image_nx - stamp_size,
    )
    corner_y = jnp.clip(
        jnp.round(positions[..., 1]).astype(jnp.int32) - half,
        0, image_ny - stamp_size,
    )
    centre = (stamp_size - 1) / 2.0
    offset_x = positions[..., 0] - (corner_x + centre)
    offset_y = positions[..., 1] - (corner_y + centre)
    return corner_x, corner_y, offset_x, offset_y


def decode_learned_sources(
    ae: GalaxyAutoEncoder, z: jnp.ndarray, learned_indices: jnp.ndarray
) -> jnp.ndarray:
    """Decode the learned tier's latent codes once, for every exposure.

    Args:
        ae: Frozen AutoEncoder.
        z: Per-source latent codes, shape ``(n_sources, *latent_dim)``;
            only the rows in ``learned_indices`` are decoded.
        learned_indices: Indices of the sources on the learned tier.

    Returns:
        Decoded images, shape ``(n_sources, ae.ny, ae.nx)``, zero on rows
        that are not on the learned tier.
    """
    decoded = jnp.zeros((z.shape[0], ae.ny, ae.nx))
    if learned_indices.shape[0] == 0:
        return decoded
    images = jax.vmap(lambda z_i: ae.decode(z_i, key=None)[0])(z[learned_indices])
    return decoded.at[learned_indices].set(images)


def _compute_tier_indices(
    source_stamp_tier: jnp.ndarray, n_tiers: int
) -> list[jnp.ndarray]:
    """Partition source indices by stamp tier.

    Args:
        source_stamp_tier: Per-source tier index, shape ``(n_sources,)``.
        n_tiers: Number of available tiers.

    Returns:
        List of index arrays, one per tier.
    """
    tier_np = np.asarray(source_stamp_tier)
    return [
        jnp.array(np.where(tier_np == t)[0], dtype=jnp.int32)
        for t in range(n_tiers)
    ]


def _render_tier(
    tier_idx: int,
    stamp_size: int,
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    tier_indices: list[jnp.ndarray],
    model_image: jnp.ndarray,
    decoded: Optional[jnp.ndarray] = None,
    learned_tier_idx: Optional[int] = None,
    psf_includes_pixel: bool = True,
) -> jnp.ndarray:
    """Render one stamp-size tier for one exposure and scatter-add.

    Gathers sources belonging to this tier, renders them via
    ``jax.vmap`` at the tier's stamp/FFT size, and scatter-adds the
    resulting stamps onto ``model_image``.

    If ``tier_idx == learned_tier_idx`` (only possible when the scene was
    built with ``learned_morphology.enabled``), sources are rendered from
    their decoded AE images (:func:`render_decoded_galaxy`) instead of the
    parametric ``flux``/``hlr``/``e1``/``e2`` path. Every other tier is
    unaffected.

    Args:
        tier_idx: Index of this tier in the stamp-sizes list.
        stamp_size: Stamp side length in pixels for this tier.
        exp_idx: Exposure index.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position correction, shape ``(n_sources,)``, in
            arcsec along the detector x axis (``dx / pixel_scale`` pixels).
        dy: Same, along the detector y axis.
        data: Packed exposure data.
        pixel_scale: Pixel scale in arcsec/pixel.
        tier_indices: Pre-computed index arrays, one per tier.
        model_image: Accumulated model image to scatter-add onto.
        decoded: Per-source decoded AE images, shape
            ``(n_sources, stamp, stamp)`` (see
            :func:`decode_learned_sources`), only meaningful on rows
            belonging to ``learned_tier_idx``. ``None`` when learned
            morphology is disabled.
        learned_tier_idx: Tier index that the learned-morphology path
            applies to. ``None`` when learned morphology is disabled.
        psf_includes_pixel: Whether the PSF stamps already contain the
            pixel response; selects the parametric tiers' draw method (see
            :func:`_draw_method`). The learned tier always draws with
            ``"no_pixel"``.

    Returns:
        Updated model image with this tier's contributions added.
    """
    indices = tier_indices[tier_idx]
    if indices.shape[0] == 0:
        return model_image

    fft_size = _fft_size_for_stamp(stamp_size)
    gsparams = galsim.GSParams(
        maximum_fft_size=fft_size, minimum_fft_size=fft_size
    )

    # Gather per-source data for this tier and exposure (shared by both
    # the parametric and learned-morphology rendering paths).
    pos_t = data.pixel_positions[indices, exp_idx, :]
    wcs_t = data.wcs_jacobians[indices, exp_idx, :]
    psf_t = data.psf_images[indices, exp_idx, :, :]
    vis_t = data.source_visible[indices, exp_idx]

    # Where each stamp goes, and where the source sits inside it; dx/dy
    # are free corrections on top of the catalog position.
    corner_x, corner_y, sub_x, sub_y = stamp_placement(
        pos_t, stamp_size, data.image_nx, data.image_ny
    )
    off_x = sub_x + dx[indices] / pixel_scale
    off_y = sub_y + dy[indices] / pixel_scale

    use_learned = (
        decoded is not None and learned_tier_idx is not None
        and tier_idx == learned_tier_idx
    )

    if use_learned:
        gal_t = decoded[indices]
        if gal_t.shape[-2:] != (stamp_size, stamp_size):
            raise ValueError(
                f"decoded AE images are {gal_t.shape[-2:]}, but the learned "
                f"tier's stamp is {stamp_size}x{stamp_size}"
            )

        # decode(z) still contains the fixed reference PSF baked in from
        # the AE's own training (only the spatially-varying residual was
        # deconvolved out there -- see shine.morphology.psf_residual).
        # Convolving with the *full* local PSF here would double-convolve
        # with that reference PSF and over-blur the stamp; use the
        # residual-PSF grid instead.
        if data.psf_residual_images is None:
            raise ValueError(
                "learned_morphology is enabled but no residual-PSF grid "
                "was loaded (EuclidInferenceConfig.learned_morphology."
                "psf_residual_path). See shine/morphology/psf_residual.py "
                "and scripts/build_residual_psf.py."
            )
        psf_residual_t = data.psf_residual_images[indices, exp_idx, :, :]

        # Use default-argument capture to bind stamp_size/gsparams/
        # pixel_scale at definition time (Python loop is unrolled by JIT
        # tracer), matching the parametric path's convention below.
        def render_one_learned_galaxy(
            gal_img, psf_img, wcs_params, ox, oy, visible_i,
            _ss=stamp_size, _gsp=gsparams, _ps=pixel_scale,
        ):
            return render_decoded_galaxy(
                gal_img, g1, g2, psf_img, wcs_params, ox, oy, visible_i,
                _ss, _ps, _gsp,
            )

        all_stamps = jax.vmap(render_one_learned_galaxy)(
            gal_t, psf_residual_t, wcs_t, off_x, off_y, vis_t,
        )
    else:
        # Gather per-source data for this tier and exposure
        flux_t = flux[indices]
        hlr_t = hlr[indices]
        e1_t = e1[indices]
        e2_t = e2[indices]

        # Safe PSF fallback for invisible sources
        psf_shape = psf_t.shape[-2:]
        safe_psf = jnp.zeros(psf_shape)
        safe_psf = safe_psf.at[psf_shape[0] // 2, psf_shape[1] // 2].set(1.0)

        method = _draw_method(psf_includes_pixel)

        # Use default-argument capture to bind stamp_size and gsparams
        # at definition time (Python loop is unrolled by JIT tracer).
        def render_one_galaxy(
            flux_i, hlr_i, e1_i, e2_i, ox, oy,
            psf_img, wcs_params, visible_i,
            _ss=stamp_size, _gsp=gsparams, _method=method,
        ):
            v = visible_i
            flux_i = jnp.where(v, flux_i, 1.0)
            hlr_i = jnp.where(v, hlr_i, 0.5)
            e1_i = jnp.where(v, e1_i, 0.0)
            e2_i = jnp.where(v, e2_i, 0.0)
            ox = jnp.where(v, ox, 0.0)
            oy = jnp.where(v, oy, 0.0)
            psf_img = jnp.where(v, psf_img, safe_psf)
            wcs_params = jnp.where(
                v, wcs_params, jnp.array([pixel_scale, 0.0, 0.0, pixel_scale])
            )

            wcs = galsim.JacobianWCS(
                dudx=wcs_params[0], dudy=wcs_params[1],
                dvdx=wcs_params[2], dvdy=wcs_params[3],
            )

            gal = galsim.Exponential(
                flux=flux_i, half_light_radius=hlr_i, gsparams=_gsp
            )
            gal = gal.shear(e1=e1_i, e2=e2_i)
            gal = gal.shear(g1=g1, g2=g2)

            # The PSF stamp is a detector-grid array: map it onto the sky
            # through the WCS, like the learned tier's decoded galaxies.
            psf = detector_array_profile(psf_img, wcs, _gsp)
            final = galsim.Convolve([gal, psf], gsparams=_gsp)

            stamp = final.drawImage(
                nx=_ss, ny=_ss, wcs=wcs,
                offset=galsim.PositionD(ox, oy),
                method=_method,
            ).array

            return stamp * visible_i

        # Vectorise rendering over tier sources
        all_stamps = jax.vmap(render_one_galaxy)(
            flux_t, hlr_t, e1_t, e2_t, off_x, off_y,
            psf_t, wcs_t, vis_t,
        )

    # Scatter-add stamps onto the model image
    def scatter_add(image, inputs, _ss=stamp_size):
        stamp, iy, ix = inputs
        current = jax.lax.dynamic_slice(image, (iy, ix), (_ss, _ss))
        return jax.lax.dynamic_update_slice(
            image, current + stamp, (iy, ix)
        ), None

    model_image, _ = jax.lax.scan(
        scatter_add, model_image, (all_stamps, corner_y, corner_x)
    )
    return model_image


def _render_exposure_image(
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    stamp_sizes: list[int],
    tier_indices: list[jnp.ndarray],
    decoded: Optional[jnp.ndarray] = None,
    learned_tier_idx: Optional[int] = None,
    psf_includes_pixel: bool = True,
) -> jnp.ndarray:
    """Render all galaxies for one exposure into a model image.

    Iterates over stamp-size tiers, rendering each tier's sources via
    ``jax.vmap`` at the appropriate stamp/FFT size and scatter-adding
    them onto a shared model image.

    Args:
        exp_idx: Index of the current exposure.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position correction in arcsec along detector x,
            shape ``(n_sources,)``.
        dy: Same, along detector y.
        data: Packed exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel.
        stamp_sizes: List of stamp side lengths, one per tier.
        tier_indices: Pre-computed index arrays, one per tier.
        decoded: Per-source decoded AE images (see
            :func:`decode_learned_sources`), or ``None`` when learned
            morphology is disabled.
        learned_tier_idx: Tier index the learned-morphology path applies
            to, or ``None`` when disabled.
        psf_includes_pixel: Selects the parametric tiers' draw method,
            see :func:`_render_tier`.

    Returns:
        Model image array of shape ``(image_ny, image_nx)``.
    """
    model_image = jnp.zeros((data.image_ny, data.image_nx))

    for tier_idx, stamp_size in enumerate(stamp_sizes):
        model_image = _render_tier(
            tier_idx, stamp_size, exp_idx,
            g1, g2, flux, hlr, e1, e2, dx, dy,
            data, pixel_scale, tier_indices, model_image,
            decoded=decoded, learned_tier_idx=learned_tier_idx,
            psf_includes_pixel=psf_includes_pixel,
        )

    return model_image


def _render_exposure_likelihood(
    exp_idx: int,
    g1: jnp.ndarray,
    g2: jnp.ndarray,
    flux: jnp.ndarray,
    hlr: jnp.ndarray,
    e1: jnp.ndarray,
    e2: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    data: "ExposureSet",
    pixel_scale: float,
    stamp_sizes: list[int],
    tier_indices: list[jnp.ndarray],
    observed_data: Optional[jnp.ndarray],
    extra_args: dict,
    decoded: Optional[jnp.ndarray] = None,
    learned_tier_idx: Optional[int] = None,
    psf_includes_pixel: bool = True,
) -> None:
    """Render all galaxies for one exposure and evaluate likelihood.

    Delegates rendering to :func:`_render_exposure_image` and registers
    the per-pixel Gaussian likelihood as a NumPyro sample site.

    Args:
        exp_idx: Index of the current exposure.
        g1: Global shear component 1 (scalar).
        g2: Global shear component 2 (scalar).
        flux: Per-source flux in ADU, shape ``(n_sources,)``.
        hlr: Per-source half-light radius in arcsec, shape ``(n_sources,)``.
        e1: Per-source intrinsic ellipticity component 1, shape ``(n_sources,)``.
        e2: Per-source intrinsic ellipticity component 2, shape ``(n_sources,)``.
        dx: Per-source position correction in arcsec along detector x,
            shape ``(n_sources,)``.
        dy: Same, along detector y.
        data: Packed exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel.
        stamp_sizes: List of stamp side lengths, one per tier.
        tier_indices: Pre-computed index arrays, one per tier.
        observed_data: Observed images, shape ``(n_exp, ny, nx)``, or None
            for prior predictive sampling.
        extra_args: Additional keyword arguments forwarded from the model
            call (unused, reserved for future extensions).
        decoded: Per-source decoded AE images (see
            :func:`decode_learned_sources`), or ``None`` when learned
            morphology is disabled.
        learned_tier_idx: Tier index the learned-morphology path applies
            to, or ``None`` when disabled.
        psf_includes_pixel: Selects the parametric tiers' draw method,
            see :func:`_render_tier`.
    """
    model_image = _render_exposure_image(
        exp_idx, g1, g2, flux, hlr, e1, e2, dx, dy,
        data, pixel_scale, stamp_sizes, tier_indices,
        decoded=decoded, learned_tier_idx=learned_tier_idx,
        psf_includes_pixel=psf_includes_pixel,
    )

    # Likelihood: per-pixel Gaussian weighted by the RMS noise map
    noise_sigma_j = data.noise_sigma[exp_idx]

    obs_j = observed_data[exp_idx] if observed_data is not None else None

    numpyro.sample(
        f"obs_{exp_idx}",
        dist.Normal(model_image, noise_sigma_j).to_event(2),
        obs=obs_j,
    )


class MultiExposureScene:
    """Builder for multi-exposure, multi-object NumPyro scene models.

    Constructs a forward generative model that:

    1. Samples global shear ``(g1, g2)`` and per-galaxy parameters
       (flux, half-light radius, ellipticity, position offset).
    2. Renders each galaxy on a small internal stamp via JAX-GalSim,
       convolved with the interpolated per-source PSF.
    3. Scatter-adds all stamps onto the full model image per exposure.
    4. Evaluates per-pixel Gaussian likelihood weighted by the RMS
       noise map.

    Sources are grouped into stamp-size tiers so that small galaxies
    use cheap FFT sizes while large galaxies get bigger stamps.

    Attributes:
        config: Euclid inference configuration.
        data: Packed multi-exposure data (images, PSFs, WCS, noise maps).
        ae: Frozen AutoEncoder used to decode the learned-morphology tier,
            or ``None`` when ``config.learned_morphology`` is disabled.
        flow: Frozen latent-space normalizing flow paired with ``ae``, or
            ``None`` when disabled.
    """

    def __init__(
        self, config: EuclidInferenceConfig, exposure_set: "ExposureSet"
    ) -> None:
        """Initialize the multi-exposure scene builder.

        Args:
            config: Euclid inference configuration including priors,
                stamp sizes, and pixel scale.
            exposure_set: Pre-built data structure holding all exposure
                images, noise maps, PSF stamps, WCS Jacobians, and source
                catalog information.
        """
        self.config = config
        self.data = exposure_set

        # Load the AE/flow once here (not per NumPyro model call) and keep
        # them as plain Python attributes, never as numpyro.sample sites
        # or SVI/NUTS-optimized parameters.
        self.ae: Optional[GalaxyAutoEncoder] = None
        self.flow: Optional[LatentFlow] = None
        self._learned_tier_idx: Optional[int] = None

        lm = config.learned_morphology
        if lm is not None and lm.enabled:
            self.ae = load_frozen_autoencoder(lm.ae_checkpoint_dir, lm.ae_epoch)
            self.flow = load_frozen_flow(lm.flow_checkpoint_dir, lm.flow_epoch)
            self._learned_tier_idx = config.galaxy_stamp_sizes.index(
                lm.apply_to_stamp_size
            )
            if self.ae.nx != lm.apply_to_stamp_size or self.ae.ny != lm.apply_to_stamp_size:
                raise ValueError(
                    f"the AE decodes {self.ae.nx}x{self.ae.ny} stamps, but "
                    f"learned_morphology.apply_to_stamp_size="
                    f"{lm.apply_to_stamp_size}"
                )
            if abs(self.ae.scale - config.data.pixel_scale) > 1e-6:
                raise ValueError(
                    f"the AE was trained at {self.ae.scale}\"/px, but "
                    f"data.pixel_scale={config.data.pixel_scale}"
                )

    def _prepare_tier_indices(self, label: str) -> tuple[list[int], float, list[jnp.ndarray]]:
        """Pre-compute tier indices and log tier summary.

        Args:
            label: Human-readable label for the log message
                (e.g. "multi-exposure" or "single-exposure 0").

        Returns:
            Tuple of ``(stamp_sizes, pixel_scale, tier_indices)``.
        """
        stamp_sizes = self.config.galaxy_stamp_sizes
        pixel_scale = self.config.data.pixel_scale
        tier_indices = _compute_tier_indices(
            self.data.source_stamp_tier, len(stamp_sizes)
        )

        tier_str = ", ".join(
            f"{stamp_sizes[t]}px: {tier_indices[t].shape[0]}"
            for t in range(len(stamp_sizes))
        )
        logger.info(
            "Building %s model: %d sources, tiers=[%s]",
            label,
            self.data.n_sources,
            tier_str,
        )
        return stamp_sizes, pixel_scale, tier_indices

    def _sample_parameters(
        self, tier_indices: Optional[list[jnp.ndarray]] = None
    ) -> tuple:
        """Sample global shear and per-source parameters from config.

        Prior distributions are read from ``self.config.gal`` (a
        :class:`~shine.config.GalaxyConfig`).  Parameters with
        ``center="catalog"`` use the per-source catalog values stored
        in ``self.data`` as the distribution location.

        ``flux``/``hlr``/``e1``/``e2``/``dx``/``dy`` are always sampled
        for every source, exactly as before the learned-morphology
        addition (they are simply unused, not removed, on the learned
        tier's rendering path). When ``self.ae`` is set, an *additional*
        ``z`` latent code is sampled — only for sources belonging to the
        learned tier — via :func:`~shine.morphology.prior.sample_latent_codes`.

        Args:
            tier_indices: Pre-computed per-tier source index arrays (see
                :func:`_compute_tier_indices`), required when learned
                morphology is enabled so the learned tier's sources can be
                identified. Unused otherwise.

        Returns:
            Tuple ``(g1, g2, flux, hlr, e1, e2, dx, dy, z)``. ``z`` is
            ``None`` unless learned morphology is enabled, in which case
            it has shape ``(n_sources, *flow.latent_dim)`` with only the
            learned tier's rows populated (the rest are zero).
        """
        gal_cfg = self.config.gal
        data = self.data

        # Global shear
        g1 = parse_prior("g1", gal_cfg.shear.g1)
        g2 = parse_prior("g2", gal_cfg.shear.g2)

        with numpyro.plate("sources", data.n_sources):
            flux = parse_prior(
                "flux", gal_cfg.flux,
                catalog_values=data.catalog_flux_adu,
            )
            hlr = parse_prior(
                "hlr", gal_cfg.half_light_radius,
                catalog_values=data.catalog_hlr_arcsec,
            )

            # Intrinsic ellipticity
            e1 = 0.0
            e2 = 0.0
            if gal_cfg.ellipticity is not None:
                e1 = parse_prior("e1", gal_cfg.ellipticity.e1)
                e2 = parse_prior("e2", gal_cfg.ellipticity.e2)

            # Position offsets
            dx = 0.0
            dy = 0.0
            if gal_cfg.position is not None and gal_cfg.position.type == "Offset":
                dx = parse_prior("dx", gal_cfg.position.dx)
                dy = parse_prior("dy", gal_cfg.position.dy)

        z = None
        if self.ae is not None:
            learned_indices = tier_indices[self._learned_tier_idx]
            n_learned = int(learned_indices.shape[0])
            latent_shape = tuple(self.flow.latent_dim)
            z = jnp.zeros((data.n_sources, *latent_shape))
            if n_learned > 0:
                with numpyro.plate("learned_sources", n_learned):
                    z_learned = sample_latent_codes("z", self.flow, n_learned)
                z = z.at[learned_indices].set(z_learned)

        return g1, g2, flux, hlr, e1, e2, dx, dy, z

    def _decode(
        self, z: Optional[jnp.ndarray], tier_indices: list[jnp.ndarray]
    ) -> Optional[jnp.ndarray]:
        """Decode the learned tier once per model call (see
        :func:`decode_learned_sources`), or return ``None`` when learned
        morphology is disabled."""
        if self.ae is None or z is None:
            return None
        return decode_learned_sources(
            self.ae, z, tier_indices[self._learned_tier_idx]
        )

    def build_model(self) -> Callable:
        """Build the multi-exposure NumPyro model.

        Returns:
            A NumPyro model function with signature
            ``model(observed_data=None, **extra_args)`` suitable for passing
            to ``numpyro.infer.MCMC`` or ``numpyro.infer.SVI``.
        """
        data = self.data
        stamp_sizes, pixel_scale, tier_indices = self._prepare_tier_indices(
            f"{data.n_exposures}-exposure"
        )
        sample_parameters = self._sample_parameters
        decode = self._decode
        learned_tier_idx = self._learned_tier_idx
        psf_includes_pixel = self.config.psf_includes_pixel

        def model(
            observed_data: Optional[jnp.ndarray] = None, **extra_args
        ) -> None:
            """NumPyro probabilistic model for multi-exposure shear inference.

            Args:
                observed_data: Observed images, shape ``(n_exp, ny, nx)``.
                    Pass ``None`` for prior predictive sampling.
                **extra_args: Reserved for future use.
            """
            g1, g2, flux, hlr, e1, e2, dx, dy, z = sample_parameters(tier_indices)
            decoded = decode(z, tier_indices)

            for j in range(data.n_exposures):
                _render_exposure_likelihood(
                    j, g1, g2, flux, hlr, e1, e2, dx, dy,
                    data, pixel_scale, stamp_sizes, tier_indices,
                    observed_data, extra_args,
                    decoded=decoded, learned_tier_idx=learned_tier_idx,
                    psf_includes_pixel=psf_includes_pixel,
                )

        return model

    def build_single_exposure_model(
        self, exposure_idx: int = 0
    ) -> Callable:
        """Build a model for a single exposure (useful for debugging).

        Identical to :meth:`build_model` but only renders and evaluates
        the likelihood for one exposure, which makes JIT compilation and
        gradient evaluation much faster during development.

        Args:
            exposure_idx: Zero-based index of the exposure to model.

        Returns:
            A NumPyro model function with the same signature as
            :meth:`build_model`.

        Raises:
            IndexError: If ``exposure_idx`` is out of range.
        """
        if exposure_idx < 0 or exposure_idx >= self.data.n_exposures:
            raise IndexError(
                f"exposure_idx {exposure_idx} out of range for "
                f"{self.data.n_exposures} exposures"
            )

        data = self.data
        stamp_sizes, pixel_scale, tier_indices = self._prepare_tier_indices(
            f"single-exposure {exposure_idx}"
        )
        sample_parameters = self._sample_parameters
        decode = self._decode
        learned_tier_idx = self._learned_tier_idx
        psf_includes_pixel = self.config.psf_includes_pixel

        def model(
            observed_data: Optional[jnp.ndarray] = None, **extra_args
        ) -> None:
            """NumPyro model for single-exposure shear inference.

            Args:
                observed_data: Observed images, shape ``(n_exp, ny, nx)``.
                    Pass ``None`` for prior predictive sampling.
                **extra_args: Reserved for future use.
            """
            g1, g2, flux, hlr, e1, e2, dx, dy, z = sample_parameters(tier_indices)
            decoded = decode(z, tier_indices)

            _render_exposure_likelihood(
                exposure_idx, g1, g2, flux, hlr, e1, e2, dx, dy,
                data, pixel_scale, stamp_sizes, tier_indices,
                observed_data, extra_args,
                decoded=decoded, learned_tier_idx=learned_tier_idx,
                psf_includes_pixel=psf_includes_pixel,
            )

        return model


def render_model_images(
    params: dict,
    data: "ExposureSet",
    pixel_scale: float = 0.1,
    stamp_sizes: Optional[list[int]] = None,
    ae: Optional[GalaxyAutoEncoder] = None,
    learned_tier_idx: Optional[int] = None,
    psf_includes_pixel: bool = True,
) -> jnp.ndarray:
    """Render model images for all exposures from parameter values.

    Takes a dictionary of MAP (or sampled) parameters and produces the
    corresponding forward-model images for every exposure by reusing the
    same tiered vmap + scan rendering pipeline as the NumPyro model.

    Args:
        params: Parameter dictionary with keys ``"g1"``, ``"g2"``,
            ``"flux"``, ``"hlr"``, ``"e1"``, ``"e2"``, ``"dx"``, ``"dy"``.
            Scalars for shear, arrays of shape ``(n_sources,)`` for the
            rest. If learned morphology was used, also pass ``"z"``
            (shape ``(n_sources, *ae's latent shape)``, e.g. the
            ``"z"`` deterministic site from the model trace) along with
            ``ae``/``learned_tier_idx`` to render that tier's sources
            through the decoder instead of the parametric path.
        data: Packed multi-exposure data (images, PSFs, WCS, etc.).
        pixel_scale: Pixel scale in arcsec/pixel (default 0.1).
        stamp_sizes: Stamp tier sizes.  If ``None``, defaults to
            ``[64, 128, 256]``.
        ae: Frozen AutoEncoder to decode the learned tier, or ``None`` to
            render every tier parametrically (default).
        learned_tier_idx: Tier index the learned-morphology path applies
            to. Required (together with ``ae`` and ``params["z"]``) to
            render the learned tier.
        psf_includes_pixel: Selects the parametric tiers' draw method;
            pass the ``EuclidInferenceConfig.psf_includes_pixel`` the
            model was fitted with.

    Returns:
        Model images array of shape ``(n_exp, image_ny, image_nx)``.
    """
    if stamp_sizes is None:
        stamp_sizes = [64, 128, 256]

    tier_indices = _compute_tier_indices(
        data.source_stamp_tier, len(stamp_sizes)
    )

    param_names = ("g1", "g2", "flux", "hlr", "e1", "e2", "dx", "dy")
    g1, g2, flux, hlr, e1, e2, dx, dy = (
        jnp.asarray(params[k]) for k in param_names
    )
    decoded = None
    if ae is not None and learned_tier_idx is not None and "z" in params:
        decoded = decode_learned_sources(
            ae, jnp.asarray(params["z"]), tier_indices[learned_tier_idx]
        )

    images = [
        _render_exposure_image(
            j, g1, g2, flux, hlr, e1, e2, dx, dy,
            data, pixel_scale, stamp_sizes, tier_indices,
            decoded=decoded, learned_tier_idx=learned_tier_idx,
            psf_includes_pixel=psf_includes_pixel,
        )
        for j in range(data.n_exposures)
    ]

    return jnp.stack(images, axis=0)
