"""Tests for shine.morphology.prior: NumPyro latent-flow prior sampling.

Uses a freshly initialized (untrained) LatentFlow via make_latent_flow
rather than a real checkpoint: sample_latent_codes only depends on the
flow's shape/API (forward, latent_dim, unflatten_latent), not on trained
weights, so these tests run without any checkpoint files on disk.
"""

import jax
import numpy as np
import numpyro
from numpyro.handlers import seed, trace

from shine.morphology.nn.flow import make_latent_flow
from shine.morphology.prior import sample_latent_codes


def _flow():
    return make_latent_flow(key=jax.random.key(0))


def _model(flow, n_sources):
    def model():
        with numpyro.plate("sources", n_sources):
            return sample_latent_codes("z", flow, n_sources)

    return model


class TestSampleLatentCodes:
    def test_trace_sites_and_shapes(self):
        flow = _flow()
        n_sources = 5
        latent_dim_flat = int(np.prod(flow.latent_dim))

        tr = trace(seed(_model(flow, n_sources), jax.random.PRNGKey(0))).get_trace()

        assert "z_base" in tr and tr["z_base"]["type"] == "sample"
        assert tr["z_base"]["value"].shape == (n_sources, latent_dim_flat)

        assert "z" in tr and tr["z"]["type"] == "deterministic"
        assert tr["z"]["value"].shape == (n_sources, *flow.latent_dim)

    def test_return_value_matches_deterministic_site(self):
        flow = _flow()
        model = _model(flow, 4)

        z_returned = seed(model, jax.random.PRNGKey(7))()
        tr = trace(seed(model, jax.random.PRNGKey(7))).get_trace()

        np.testing.assert_array_equal(
            np.asarray(z_returned), np.asarray(tr["z"]["value"])
        )

    def test_deterministic_given_same_seed(self):
        flow = _flow()
        model = _model(flow, 3)

        z1 = seed(model, jax.random.PRNGKey(42))()
        z2 = seed(model, jax.random.PRNGKey(42))()

        np.testing.assert_array_equal(np.asarray(z1), np.asarray(z2))

    def test_differs_across_seeds(self):
        flow = _flow()
        model = _model(flow, 3)

        z1 = seed(model, jax.random.PRNGKey(0))()
        z2 = seed(model, jax.random.PRNGKey(1))()

        assert not np.allclose(np.asarray(z1), np.asarray(z2))
