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
import paramax

from .nn.flow import LatentFlow


def _base_loc_scale(flow: LatentFlow) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the flow's own base-distribution ``(loc, scale)``.

    ``LatentFlow`` builds its flowjax base distribution as
    ``Normal(jnp.zeros(prod(latent_dim)))``, but flowjax stores that
    distribution's ``loc``/``scale`` as ordinary inexact arrays, so they are
    *trained along with the bijection* unless the training script freezes
    them (``paramax.NonTrainable``). The checkpoints used here did not:
    ``wandb_weights/4q23te9a/epoch_420`` carries a base ``loc`` whose worst
    component is ``-1.97`` and scales spanning ``0.95-1.15``, not ``N(0, 1)``.

    Pushing a standard normal through :meth:`LatentFlow.forward` would
    therefore *not* reproduce ``flow.sample()`` — measured on that
    checkpoint, per-dimension means drift by up to 1.96 and percentiles by
    up to 3.2, i.e. the inference would run under a prior that is not the
    learned one. Reading the base parameters back makes the two agree to
    Monte-Carlo noise (max mean drift 0.03).

    Args:
        flow: Frozen :class:`~shine.morphology.nn.flow.LatentFlow`.

    Returns:
        Tuple ``(loc, scale)`` of shape ``(prod(latent_dim),)`` each.

    Raises:
        AttributeError: If the flow's base distribution exposes no
            ``loc``/``scale`` (i.e. is not a location-scale family), in
            which case this reparameterization does not apply.
    """
    base = paramax.unwrap(flow.flow).base_dist
    try:
        return jnp.asarray(base.loc), jnp.asarray(base.scale)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"Flow base distribution {type(base).__name__} has no loc/scale; "
            "sample_latent_codes only supports location-scale base "
            "distributions."
        ) from exc


def sample_latent_codes(name: str, flow: LatentFlow, n_sources: int) -> jnp.ndarray:
    """Sample per-source AE latent codes from the flow prior.

    Must be called inside a ``numpyro.plate(..., n_sources)`` context so
    the base-distribution sample site is correctly batched over sources.

    The sample site stays a *standard* normal — good geometry for MAP and
    HMC, and ``0`` remains a sensible initial value — and the flow's own
    base ``loc``/``scale`` are applied deterministically on top of it (see
    :func:`_base_loc_scale`, which explains why they are not ``0``/``1``).
    Drawing this site and pushing it through the flow therefore reproduces
    ``flow.sample()``.

    Args:
        name: Base name for the NumPyro sites. Produces a sample site
            ``f"{name}_base"`` (a standard-normal draw, rescaled to the
            flow's base distribution before the bijection) and a
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
    base_loc, base_scale = _base_loc_scale(flow)
    z_std = numpyro.sample(
        f"{name}_base",
        dist.Normal(jnp.zeros(latent_dim_flat), 1.0).to_event(1),
    )
    z_base = base_loc + base_scale * z_std
    # flowjax bijections transform a single (unbatched) event; vmap
    # explicitly over the source batch rather than relying on the flow
    # to broadcast internally.
    z_flat = jax.vmap(flow.forward)(z_base)
    z = flow.unflatten_latent(z_flat)
    numpyro.deterministic(name, z)
    return z
