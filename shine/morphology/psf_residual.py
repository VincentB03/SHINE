r"""Residual-PSF kernels for the learned-morphology (AE + Flow) tier.

The AE checkpoint used for ``shine.morphology`` was trained against
``psf_residual`` (see ``VincentB03/euclid-Q1-VF`` on Hugging Face), not the
full local Euclid PSF. Each per-object PSF stamp is modelled as
``psf_ref (*) kernel``, where ``psf_ref`` is a fixed isotropic reference PSF
and ``(*)`` is a 2-D convolution; ``kernel`` (the *residual*) is recovered
by dividing the two in Fourier space. Concretely, the autoencoder's decoder
output is trained to satisfy::

    sci_subtracted = G_true (*) psf_stamp
                    = G_true (*) (psf_ref (*) kernel)
                    = (G_true (*) psf_ref) (*) kernel
    decode(z) (*) psf_residual ~= sci_subtracted
    => decode(z) ~= G_true (*) psf_ref

so ``decode(z)`` still contains the reference PSF ``psf_ref`` baked in.
Rendering it through the *full* local PSF (as the parametric tiers do)
double-convolves with ``psf_ref`` and over-blurs the result. The learned
tier must instead be convolved with the residual kernel computed here.

Ported (unregularized division, matching how the AE's own training data was
built — see ``add_psf_residual`` in the ``Euclid-Q1-postage-stamps``
dataset-construction repository) from a module carrying the docstring
"Code from SHINE repository https://github.com/CosmoStat/SHINE" for
:class:`~shine.euclid.data_loader.EuclidPSFModel` itself — i.e. this is the
inverse operation of that same tiled-PSF-grid convention, applied back onto
SHINE's own PSF product.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy.io import fits


def centered_fft2(image: np.ndarray) -> np.ndarray:
    """Forward FFT of a centred PSF (peak in the middle of the array).

    ``ifftshift`` first moves the central pixel to index ``(0, 0)`` so the
    transform carries no linear phase ramp. Operates on the last two axes,
    so ``image`` may be 2-D ``(ny, nx)`` or a stack ``(n, ny, nx)``.
    """
    return np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))


def centered_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of :func:`centered_fft2`: return a centred real-valued image."""
    return np.fft.fftshift(np.fft.ifft2(spectrum, axes=(-2, -1)).real, axes=(-2, -1))


def load_reference_psf(path: Union[str, Path], normalize: bool = True) -> np.ndarray:
    """Load the isotropic reference PSF (``psf_ref``) used to build residual kernels.

    Args:
        path: Path to the reference PSF FITS file. The first HDU holding a
            2-D array is used.
        normalize: If True, rescale so the PSF sums to 1.

    Returns:
        The reference PSF as a 2-D float64 array.
    """
    with fits.open(path) as hdul:
        data = next(h.data for h in hdul if h.data is not None and np.ndim(h.data) == 2)

    psf_ref = np.asarray(data, dtype=np.float64)
    if normalize:
        total = psf_ref.sum()
        if total > 0:
            psf_ref = psf_ref / total
    return psf_ref


def compute_psf_residual(
    psf_stamp: np.ndarray,
    psf_ref: np.ndarray,
    ref_fft: Optional[np.ndarray] = None,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Compute the residual PSF kernel(s) of ``psf_stamp`` relative to ``psf_ref``.

    Solves ``psf_stamp = psf_ref (*) kernel`` for ``kernel`` by dividing the
    two in Fourier space.

    Args:
        psf_stamp: A single PSF stamp ``(ny, nx)`` or a stack ``(n, ny, nx)``.
        psf_ref: The reference PSF, matching the ``(ny, nx)`` of each stamp.
        ref_fft: Optional pre-computed ``centered_fft2(psf_ref)``; pass it to
            avoid recomputing the reference transform on every call/batch.
        epsilon: Tikhonov regularisation on the (power-normalised) division.
            ``0.0`` (the default) reproduces the plain Fourier division used
            to build the AE's own training data — keep it at ``0.0`` unless
            deliberately deviating from that convention.

    Returns:
        The residual kernel(s), a real array centred (peak in the middle)
        with the same shape as ``psf_stamp``.
    """
    psf_stamp = np.asarray(psf_stamp, dtype=np.float64)
    psf_ref = np.asarray(psf_ref, dtype=np.float64)

    if psf_stamp.shape[-2:] != psf_ref.shape:
        raise ValueError(
            f"PSF stamp shape {psf_stamp.shape[-2:]} and reference shape "
            f"{psf_ref.shape} are incompatible."
        )

    b = centered_fft2(psf_ref) if ref_fft is None else ref_fft
    x = centered_fft2(psf_stamp)

    if epsilon > 0:
        kernel_fft = x * np.conj(b) / (np.abs(b) ** 2 + epsilon)
    else:
        kernel_fft = x / b

    return centered_ifft2(kernel_fft)


def reconvolve_psf(
    residual_kernel: np.ndarray,
    psf_ref: np.ndarray,
    ref_fft: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rebuild PSF stamp(s) from residual kernel(s): ``psf_ref (*) residual_kernel``.

    Inverse of :func:`compute_psf_residual` with ``epsilon=0``. Use it to
    check that a residual kernel round-trips back to the original stamp.

    Args:
        residual_kernel: A kernel ``(ny, nx)`` or a stack ``(n, ny, nx)``.
        psf_ref: The reference PSF, matching the ``(ny, nx)`` of each kernel.
        ref_fft: Optional pre-computed ``centered_fft2(psf_ref)``.

    Returns:
        The reconstructed PSF stamp(s), centred, same shape as
        ``residual_kernel``.
    """
    residual_kernel = np.asarray(residual_kernel, dtype=np.float64)
    psf_ref = np.asarray(psf_ref, dtype=np.float64)

    b = centered_fft2(psf_ref) if ref_fft is None else ref_fft
    k = centered_fft2(residual_kernel)
    return centered_ifft2(b * k)


class ResidualPSFGridResult(NamedTuple):
    """Outcome of :func:`build_residual_psf_grid`.

    Attributes:
        output_path: Where the residual-PSF grid FITS was written.
        max_abs_error: Largest ``|reconvolve(kernel) - original_stamp|``
            over all tiles (round-trip validation).
        max_rel_error: ``max_abs_error`` divided by the original stamp's
            peak value, worst tile.
        n_tiles: Number of PSF stamps processed (``grid_nx * grid_ny``).
    """

    output_path: Path
    max_abs_error: float
    max_rel_error: float
    n_tiles: int


def build_residual_psf_grid(
    psf_grid_path: Union[str, Path],
    psf_ref_path: Union[str, Path],
    output_path: Union[str, Path],
    stamp_size: int = 21,
    grid_nx: int = 9,
    grid_ny: int = 9,
    epsilon: float = 0.0,
) -> ResidualPSFGridResult:
    """Build a residual-PSF grid FITS file matching an existing PSF-grid FITS.

    Reads a tiled PSF-grid FITS file in the same format as
    ``EuclidPSFModel`` expects (a single 2-D image tiling
    ``grid_ny x grid_nx`` stamps of ``stamp_size x stamp_size``, e.g.
    ``PSF_3-4-F.fits.gz``), replaces every stamp with its residual kernel
    relative to ``psf_ref`` (see :func:`compute_psf_residual`), and writes
    the result to ``output_path`` with the identical HDU/header layout —
    so :class:`~shine.euclid.data_loader.EuclidPSFModel` can load it
    unchanged, just pointed at a different file.

    Args:
        psf_grid_path: Path to the existing PSF-grid FITS (e.g.
            ``PSF_3-4-F.fits.gz``).
        psf_ref_path: Path to the isotropic reference PSF FITS
            (``psf_ref``, matching ``stamp_size``).
        output_path: Where to write the residual-PSF grid FITS.
        stamp_size: Side length of each individual PSF stamp (21 for the
            Euclid VIS PSF grid convention used throughout SHINE).
        grid_nx: Number of stamps along the x (column) axis.
        grid_ny: Number of stamps along the y (row) axis.
        epsilon: Forwarded to :func:`compute_psf_residual`. Keep at ``0.0``
            to match the AE's own training data (see module docstring).

    Returns:
        A :class:`ResidualPSFGridResult` with the output path and
        round-trip validation error statistics.

    Raises:
        ValueError: If no 2-D image HDU is found in ``psf_grid_path``, or
            the tile grid shape doesn't match ``stamp_size``/``grid_nx``/
            ``grid_ny``.
    """
    psf_ref = load_reference_psf(psf_ref_path, normalize=True)
    if psf_ref.shape != (stamp_size, stamp_size):
        raise ValueError(
            f"Reference PSF shape {psf_ref.shape} != expected "
            f"({stamp_size}, {stamp_size})"
        )
    ref_fft = centered_fft2(psf_ref)

    with fits.open(psf_grid_path) as hdul:
        image_hdu = next(
            h for h in hdul if h.data is not None and np.ndim(h.data) == 2
        )
        grid_data = np.asarray(image_hdu.data, dtype=np.float64)

        expected_shape = (grid_ny * stamp_size, grid_nx * stamp_size)
        if grid_data.shape != expected_shape:
            raise ValueError(
                f"PSF grid shape {grid_data.shape} != expected "
                f"{expected_shape} for stamp_size={stamp_size}, "
                f"grid_nx={grid_nx}, grid_ny={grid_ny}"
            )

        # (grid_ny*stamp_size, grid_nx*stamp_size)
        #   -> (grid_ny, stamp_size, grid_nx, stamp_size)
        #   -> (grid_ny, grid_nx, stamp_size, stamp_size)
        # Same convention as EuclidPSFModel.__init__.
        stamps = grid_data.reshape(grid_ny, stamp_size, grid_nx, stamp_size).transpose(
            0, 2, 1, 3
        )
        flat_stamps = stamps.reshape(grid_ny * grid_nx, stamp_size, stamp_size)

        kernels = compute_psf_residual(flat_stamps, psf_ref, ref_fft=ref_fft, epsilon=epsilon)

        # Round-trip validation: psf_ref (*) kernel should reconstruct the
        # original stamp.
        reconstructed = reconvolve_psf(kernels, psf_ref, ref_fft=ref_fft)
        abs_error = np.abs(reconstructed - flat_stamps)
        peak = flat_stamps.reshape(flat_stamps.shape[0], -1).max(axis=-1)
        rel_error = abs_error.reshape(abs_error.shape[0], -1).max(axis=-1) / np.where(
            peak > 0, peak, 1.0
        )
        max_abs_error = float(abs_error.max())
        max_rel_error = float(rel_error.max())

        # Reassemble into the flat tiled-grid layout, inverse of the
        # reshape/transpose above.
        kernel_grid = (
            kernels.reshape(grid_ny, grid_nx, stamp_size, stamp_size)
            .transpose(0, 2, 1, 3)
            .reshape(grid_ny * stamp_size, grid_nx * stamp_size)
        )

        out_hdul = hdul.copy()
        out_index = hdul.index(image_hdu)
        out_hdul[out_index].data = kernel_grid.astype(np.float32)
        out_hdul[out_index].header["HISTORY"] = (
            "Residual PSF kernel = IFFT[FFT(psf_stamp)/FFT(psf_ref)], "
            f"epsilon={epsilon}, ref={Path(psf_ref_path).name} "
            "(shine.morphology.psf_residual.build_residual_psf_grid)"
        )

        output_path = Path(output_path)
        out_hdul.writeto(output_path, overwrite=True)

    return ResidualPSFGridResult(
        output_path=output_path,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        n_tiles=grid_nx * grid_ny,
    )
