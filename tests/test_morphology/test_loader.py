"""Tests for shine.morphology.loader: real AE + Flow checkpoints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shine.morphology.loader import load_frozen_autoencoder, load_frozen_flow

from .conftest import (
    AE_CHECKPOINT_DIR,
    AE_EPOCH,
    FLOW_CHECKPOINT_DIR,
    FLOW_EPOCH,
    requires_checkpoints,
)


@requires_checkpoints
class TestLoadFrozenAutoencoder:
    def test_loads_and_matches_trained_config(self):
        ae = load_frozen_autoencoder(AE_CHECKPOINT_DIR, AE_EPOCH)
        assert ae.nx == 64
        assert ae.ny == 64
        assert ae.scale == pytest.approx(0.1)

    def test_decode_is_deterministic_across_keys(self):
        """inference_mode must disable dropout: two different PRNG keys
        should give bit-identical decoder output."""
        ae = load_frozen_autoencoder(AE_CHECKPOINT_DIR, AE_EPOCH)
        z = jnp.zeros((1, 4, 4))

        g1 = ae.decode(z, key=jax.random.key(0))
        g2 = ae.decode(z, key=jax.random.key(1))

        np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))
        assert g1.shape == (1, 64, 64)
        assert np.all(np.asarray(g1) >= 0)  # softplus output


@requires_checkpoints
class TestLoadFrozenFlow:
    def test_loads_and_matches_trained_config(self):
        flow = load_frozen_flow(FLOW_CHECKPOINT_DIR, FLOW_EPOCH)
        assert list(flow.latent_dim) == [1, 4, 4]
        assert flow.cond_dim is None

    def test_sample_and_unflatten_matches_ae_latent_shape(self):
        flow = load_frozen_flow(FLOW_CHECKPOINT_DIR, FLOW_EPOCH)
        z_flat = flow.sample(key=jax.random.key(0), sample_shape=(3,))
        z = flow.unflatten_latent(z_flat)
        assert z.shape == (3, 1, 4, 4)


@requires_checkpoints
def test_ae_and_flow_compose_end_to_end():
    """flow.sample -> ae.decode, the exact hand-off used by
    shine.morphology.prior.sample_latent_codes downstream."""
    ae = load_frozen_autoencoder(AE_CHECKPOINT_DIR, AE_EPOCH)
    flow = load_frozen_flow(FLOW_CHECKPOINT_DIR, FLOW_EPOCH)

    z_flat = flow.sample(key=jax.random.key(0), sample_shape=(2,))
    z = flow.unflatten_latent(z_flat)

    images = jax.vmap(lambda zi: ae.decode(zi, key=None))(z)
    assert images.shape == (2, 1, 64, 64)
    assert np.all(np.isfinite(np.asarray(images)))
