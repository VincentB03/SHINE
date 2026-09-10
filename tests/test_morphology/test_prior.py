"""Tests for shine.morphology.prior: NumPyro latent-flow prior sampling.

Uses a freshly initialized (untrained) LatentFlow via make_latent_flow
rather than a real checkpoint: sample_latent_codes only depends on the
flow's shape/API (forward, latent_dim, unflatten_latent), not on trained
weights, so these tests run without any checkpoint files on disk.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import paramax
from flowjax.distributions import Normal as FlowjaxNormal
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

    def test_matches_flow_own_samples_with_non_standard_base(self):
        """The prior must reproduce ``flow.sample()``, base params included.

        flowjax keeps the base distribution's ``loc``/``scale`` trainable, so
        a trained checkpoint's base is not guaranteed to be ``N(0, 1)``: the
        RealNVP flow previously shipped here (``2815kuay``) had drifted to a
        ``loc`` component of ``-1.65``. The MAF flow now in
        ``wandb_weights/9i28jqsm`` happens to have stayed close to standard
        (max ``|loc|`` 0.02, scales 0.90-1.04), so this test injects a
        deliberately non-standard base rather than relying on the shipped
        checkpoint -- it must keep guarding the behaviour for whatever
        checkpoint comes next. Pushing a standard normal straight through the
        bijection would sample a different prior than the one that was
        trained.
        """
        flow = _flow()
        n_dims = int(np.prod(flow.latent_dim))

        # Give the base distribution a deliberately non-standard loc/scale,
        # mimicking what training does to a real checkpoint.
        loc = jnp.linspace(-2.0, 1.0, n_dims)
        scale = jnp.linspace(0.5, 1.5, n_dims)
        # Replace the whole base distribution rather than its leaves: flowjax
        # stores `scale` behind a paramax positivity wrapper.
        flow = eqx.tree_at(
            lambda f: f.flow.base_dist, flow, FlowjaxNormal(loc, scale)
        )
        assert np.allclose(np.asarray(paramax.unwrap(flow.flow).base_dist.loc), loc)

        n_draws = 4000
        reference = np.asarray(flow.sample(key=jax.random.key(0), sample_shape=(n_draws,)))
        model = _model(flow, n_draws)
        sampled = np.asarray(seed(model, jax.random.PRNGKey(1))()).reshape(n_draws, n_dims)

        # Distributional agreement, well inside Monte-Carlo noise for 4k draws
        # (the un-fixed implementation misses per-dimension means by ~2).
        np.testing.assert_allclose(
            sampled.mean(axis=0), reference.mean(axis=0), atol=0.15
        )
        np.testing.assert_allclose(
            sampled.std(axis=0), reference.std(axis=0), rtol=0.15
        )
