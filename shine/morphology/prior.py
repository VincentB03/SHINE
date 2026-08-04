"""NumPyro-compatible latent prior sampling via a frozen normalizing flow.

A :class:`~shine.morphology.nn.flow.LatentFlow` is not a
``numpyro.distributions.Distribution``: it has no ``log_prob``-friendly
sample site of its own. The standard trick for using an external
normalizing flow as a prior is to sample its Gaussian *base* distribution
as the actual NumPyro site, then push those samples through the flow's
forward bijection deterministically (a fixed, frozen transform — no extra
randomness, no gradient into the flow's own training loss).
"""

import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .nn.flow import LatentFlow


def sample_latent_codes(name: str, flow: LatentFlow, n_sources: int) -> jnp.ndarray:
    """Sample per-source AE latent codes from the flow prior.

    Must be called inside a ``numpyro.plate(..., n_sources)`` context so
    the base-distribution sample site is correctly batched over sources.

    Args:
        name: Base name for the NumPyro sites. Produces a sample site
            ``f"{name}_base"`` (the standard-normal base draw) and a
            deterministic site ``name`` (the flow-transformed latent,
            reshaped to the AE's latent shape) for diagnostics/tracing.
        flow: Frozen :class:`~shine.morphology.nn.flow.LatentFlow`, as
            returned by :func:`shine.morphology.loader.load_frozen_flow`.
            Must be unconditional (``flow.cond_dim is None``); conditional
            flows need catalog covariates threaded through ``flow.forward``,
            which is not implemented here.
        n_sources: Number of sources in the enclosing plate.

    Returns:
        Latent codes with shape ``(n_sources, *flow.latent_dim)``, ready
        to pass to ``ae.decode`` (one slice per source under ``vmap``).
    """
    latent_dim_flat = math.prod(flow.latent_dim)
    z_base = numpyro.sample(
        f"{name}_base",
        dist.Normal(jnp.zeros(latent_dim_flat), 1.0).to_event(1),
    )
    # flowjax bijections transform a single (unbatched) event; vmap
    # explicitly over the source batch rather than relying on the flow
    # to broadcast internally.
    z_flat = jax.vmap(flow.forward)(z_base)
    z = flow.unflatten_latent(z_flat)
    numpyro.deterministic(name, z)
    return z
