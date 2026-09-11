"""Tests for shine.euclid.scene module.

Verifies the NumPyro generative model structure produced by
MultiExposureScene, checking that all expected sample sites exist
with the correct shapes for both multi-exposure and single-exposure
model variants.
"""

from pathlib import Path

import jax
import numpy as np
import numpyro.handlers as handlers
import pytest

from tests.test_euclid.conftest import _data_config
from tests.test_morphology.conftest import (
    AE_CHECKPOINT_DIR,
    AE_EPOCH,
    FLOW_CHECKPOINT_DIR,
    FLOW_EPOCH,
    requires_checkpoints,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "EUC_VIS_SWL"

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="Euclid test data not available"
)

# small_config and exposure_set fixtures are provided by conftest.py


# ---------------------------------------------------------------------------
# Model trace tests
# ---------------------------------------------------------------------------


class TestMultiExposureModel:
    """Test the multi-exposure NumPyro model structure."""

    def test_model_trace(self, small_config, exposure_set):
        """The model trace should contain g1, g2, flux, hlr, e1, e2, dx, dy
        sample sites and one obs_j site per exposure, all with correct
        shapes."""
        from shine.euclid.scene import MultiExposureScene

        scene = MultiExposureScene(small_config, exposure_set)
        model = scene.build_model()

        rng = jax.random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace(
            observed_data=exposure_set.images,
        )

        # Scalar shear sites
        assert "g1" in trace
        assert "g2" in trace
        assert trace["g1"]["value"].shape == ()
        assert trace["g2"]["value"].shape == ()

        # Per-source parameter sites
        for name in ("flux", "hlr", "e1", "e2", "dx", "dy"):
            assert name in trace, f"Missing sample site: {name}"
            assert trace[name]["value"].shape == (exposure_set.n_sources,), (
                f"{name} shape mismatch"
            )

        # Observation sites for each exposure
        for j in range(exposure_set.n_exposures):
            site = f"obs_{j}"
            assert site in trace, f"Missing observation site: {site}"

    def test_single_exposure_model(self, small_config, exposure_set):
        """A single-exposure model should have only obs_0 and not obs_1
        or obs_2."""
        from shine.euclid.scene import MultiExposureScene

        scene = MultiExposureScene(small_config, exposure_set)
        model = scene.build_single_exposure_model(exposure_idx=0)

        rng = jax.random.PRNGKey(1)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace(
            observed_data=exposure_set.images,
        )

        assert "obs_0" in trace
        assert "obs_1" not in trace
        assert "obs_2" not in trace


# ---------------------------------------------------------------------------
# Stamp placement
# ---------------------------------------------------------------------------


class TestStampPlacement:
    """Sources must be drawn at their catalog position, not at the
    nearest pixel (nor half a pixel off on even stamps)."""

    def test_offsets_point_back_to_the_position(self):
        from shine.euclid.scene import stamp_placement

        pos = np.array([[100.3, 200.7], [50.5, 60.49], [1000.0, 10.0]])
        for stamp in (64, 65):
            cx, cy, ox, oy = (np.asarray(a) for a in stamp_placement(pos, stamp, 2048, 2066))
            centre = (stamp - 1) / 2
            np.testing.assert_allclose(cx + centre + ox, pos[:, 0], atol=1e-4)
            np.testing.assert_allclose(cy + centre + oy, pos[:, 1], atol=1e-4)
            # Away from the edges the stamp is the plain rounded cutout.
            np.testing.assert_array_equal(cx[:2], np.round(pos[:2, 0]).astype(int) - stamp // 2)
            assert np.all(np.abs(ox[:2]) <= 1.0) and np.all(np.abs(oy[:2]) <= 1.0)

    def test_edge_clipping_keeps_the_position(self):
        from shine.euclid.scene import stamp_placement

        pos = np.array([[10.2, 2060.8]])
        cx, cy, ox, oy = (np.asarray(a) for a in stamp_placement(pos, 64, 2048, 2066))
        assert cx[0] == 0 and cy[0] == 2066 - 64
        np.testing.assert_allclose(cx + 31.5 + ox, pos[:, 0], atol=1e-4)
        np.testing.assert_allclose(cy + 31.5 + oy, pos[:, 1], atol=1e-4)

    def test_model_centroid_matches_catalog_position(self, exposure_set):
        """Round, unsheared galaxies with dx = dy = 0 must be centred on
        their catalog pixel positions (up to the PSF's own centroid)."""
        from shine.euclid.scene import render_model_images, stamp_placement

        n = exposure_set.n_sources
        stamp_sizes = [64, 128, 256]
        params = {
            "g1": 0.0, "g2": 0.0,
            "flux": np.full(n, 1000.0), "hlr": np.full(n, 0.2),
            "e1": np.zeros(n), "e2": np.zeros(n),
            "dx": np.zeros(n), "dy": np.zeros(n),
        }
        images = np.asarray(render_model_images(params, exposure_set, stamp_sizes=stamp_sizes))
        tiers = np.asarray(exposure_set.source_stamp_tier)

        checked = 0
        for i in range(n):
            if not bool(exposure_set.source_visible[i, 0]):
                continue
            ss = stamp_sizes[tiers[i]]
            pos = np.asarray(exposure_set.pixel_positions[i, 0])
            cx, cy = (int(a) for a in stamp_placement(
                pos, ss, exposure_set.image_nx, exposure_set.image_ny)[:2])
            stamp = images[0, cy:cy + ss, cx:cx + ss]
            y, x = np.mgrid[0:ss, 0:ss]
            got = np.array([(stamp * x).sum(), (stamp * y).sum()]) / stamp.sum()

            psf = np.asarray(exposure_set.psf_images[i, 0])
            py, px = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
            psf_shift = np.array([(psf * px).sum(), (psf * py).sum()]) / psf.sum()
            psf_shift -= (np.array(psf.shape[::-1]) - 1) / 2

            expected = pos - np.array([cx, cy]) + psf_shift
            np.testing.assert_allclose(got, expected, atol=0.05)
            checked += 1
        assert checked > 0


# ---------------------------------------------------------------------------
# Learned-morphology (AE + Flow) tier — gated on real checkpoints
# ---------------------------------------------------------------------------


@requires_checkpoints
class TestLearnedMorphologyModel:
    """Non-regression for the ``learned_morphology.enabled=True`` path.

    Forces ``galaxy_stamp_sizes=[64]`` so every selected source lands in
    the (sole) learned tier, regardless of its actual catalog size —
    simpler and more deterministic than hand-picking small sources from
    the bundled test data.
    """

    @pytest.fixture(scope="class")
    def learned_config(self):
        from shine.euclid.config import EuclidInferenceConfig, SourceSelectionConfig
        from shine.morphology.config import LearnedMorphologyConfig

        return EuclidInferenceConfig(
            data=_data_config(),
            sources=SourceSelectionConfig(
                max_sources=3, min_snr=50.0, exclude_point_sources=False
            ),
            galaxy_stamp_sizes=[64],
            learned_morphology=LearnedMorphologyConfig(
                enabled=True,
                ae_checkpoint_dir=str(AE_CHECKPOINT_DIR),
                ae_epoch=AE_EPOCH,
                flow_checkpoint_dir=str(FLOW_CHECKPOINT_DIR),
                flow_epoch=FLOW_EPOCH,
                apply_to_stamp_size=64,
                psf_residual_path=str(DATA_DIR / "PSF_3-4-F_residual.fits.gz"),
            ),
        )

    @pytest.fixture(scope="class")
    def learned_exposure_set(self, learned_config):
        from shine.euclid.data_loader import EuclidDataLoader

        return EuclidDataLoader(learned_config).load()

    def test_model_trace_has_z_site(self, learned_config, learned_exposure_set):
        """The trace should contain the flow base sample and the z
        deterministic site, with every source on the learned tier."""
        from shine.euclid.scene import MultiExposureScene

        scene = MultiExposureScene(learned_config, learned_exposure_set)
        assert scene.ae is not None and scene.flow is not None

        model = scene.build_model()
        rng = jax.random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model, rng)).get_trace(
            observed_data=learned_exposure_set.images,
        )

        n_sources = learned_exposure_set.n_sources
        latent_dim_flat = int(np.prod(scene.flow.latent_dim))
        assert "z_base" in trace
        assert trace["z_base"]["value"].shape == (n_sources, latent_dim_flat)
        assert "z" in trace and trace["z"]["type"] == "deterministic"
        assert trace["z"]["value"].shape == (n_sources, *scene.flow.latent_dim)

        for j in range(learned_exposure_set.n_exposures):
            assert f"obs_{j}" in trace

    def test_map_step_produces_no_nan(self, learned_config, learned_exposure_set):
        """A couple of MAP/SVI steps on the learned model should not
        produce NaN parameter estimates."""
        from shine.config import MAPConfig
        from shine.euclid.scene import MultiExposureScene
        from shine.inference import Inference

        scene = MultiExposureScene(learned_config, learned_exposure_set)
        model = scene.build_model()

        inference = Inference(model, learned_config.inference)
        estimates = inference.run_map(
            jax.random.PRNGKey(0),
            observed_data=learned_exposure_set.images,
            map_config=MAPConfig(num_steps=3, learning_rate=0.01),
        )

        for name, value in estimates.items():
            assert np.all(np.isfinite(np.asarray(value))), (
                f"Non-finite MAP estimate for {name}"
            )
