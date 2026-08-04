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
