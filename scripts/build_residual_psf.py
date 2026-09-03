#!/usr/bin/env python
"""Build a residual-PSF grid FITS file for the learned-morphology tier.

Replaces every stamp in an existing Euclid PSF-grid FITS (e.g.
``PSF_3-4-F.fits.gz``) with its residual kernel relative to a fixed
isotropic reference PSF, so it can be used as the PSF fed to
``shine.morphology.render.render_learned_galaxy`` instead of the full local
PSF. See ``shine/morphology/psf_residual.py`` for why this is needed (the
AE was trained against ``psf_residual``, not the full PSF) and
``data/LEARNED_MORPHOLOGY_NOTES.md`` for the full investigation.

Usage:
    python scripts/build_residual_psf.py \\
        --psf-grid data/EUC_VIS_SWL/PSF_3-4-F.fits.gz \\
        --reference-psf data/EUC_VIS_SWL/euclid_vis_isotropic_min_psf.fits \\
        --output data/EUC_VIS_SWL/PSF_3-4-F_residual.fits.gz
"""

import argparse
import logging
import sys

from shine.morphology.psf_residual import build_residual_psf_grid

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psf-grid", required=True, help="Existing PSF-grid FITS (e.g. PSF_3-4-F.fits.gz)")
    parser.add_argument("--reference-psf", required=True, help="Isotropic reference PSF FITS (psf_ref)")
    parser.add_argument("--output", required=True, help="Output path for the residual-PSF grid FITS")
    parser.add_argument("--stamp-size", type=int, default=21)
    parser.add_argument("--grid-nx", type=int, default=9)
    parser.add_argument("--grid-ny", type=int, default=9)
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="Tikhonov regularization for the Fourier division. Keep at 0.0 "
             "(default) to match the AE training data's own (unregularized) "
             "construction.",
    )
    parser.add_argument(
        "--max-rel-error", type=float, default=1e-3,
        help="Fail if the round-trip validation's worst relative error "
             "exceeds this (default 1e-3).",
    )
    args = parser.parse_args()

    result = build_residual_psf_grid(
        psf_grid_path=args.psf_grid,
        psf_ref_path=args.reference_psf,
        output_path=args.output,
        stamp_size=args.stamp_size,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        epsilon=args.epsilon,
    )

    logger.info("Wrote %s (%d tiles)", result.output_path, result.n_tiles)
    logger.info(
        "Round-trip validation: max abs error = %.3e, max rel error = %.3e",
        result.max_abs_error,
        result.max_rel_error,
    )

    if result.max_rel_error > args.max_rel_error:
        logger.error(
            "Round-trip relative error (%.3e) exceeds threshold (%.3e) -- "
            "residual-PSF file may be unreliable.",
            result.max_rel_error,
            args.max_rel_error,
        )
        sys.exit(1)

    logger.info("Validation passed.")


if __name__ == "__main__":
    main()
