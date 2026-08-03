r"""Loading of frozen, pretrained AutoEncoder + Flow checkpoints.

A checkpoint is a directory containing a ``config.yaml`` (architecture
hyperparameters, as dumped by the ``pshear`` training scripts or exported
from a wandb run) and a ``model_checkpoint_<epoch>.eqx`` file (weights,
serialized with ``eqx.tree_serialise_leaves``). Both the checkpoint
directory and the epoch are ordinary arguments here (and, in turn, plain
config fields on :class:`~shine.morphology.config.LearnedMorphologyConfig`)
so that swapping to a different trained model is a config change, not a
code change.
"""

from pathlib import Path
from typing import Union

import equinox as eqx
import yaml
from jax.random import key as jax_key

from .galaxy_ae import GalaxyAutoEncoder, make_galaxy_autoencoder
from .nn.flow import LatentFlow, make_latent_flow


def _checkpoint_path(checkpoint_dir: Union[str, Path], epoch: int) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    return checkpoint_dir / f"model_checkpoint_{epoch}.eqx"


def load_frozen_autoencoder(
    checkpoint_dir: Union[str, Path], epoch: int
) -> GalaxyAutoEncoder:
    """Load a pretrained :class:`GalaxyAutoEncoder` in inference mode.

    Args:
        checkpoint_dir: Directory containing ``config.yaml`` and
            ``model_checkpoint_<epoch>.eqx``.
        epoch: Checkpoint epoch to load.

    Returns:
        The autoencoder with weights restored and dropout disabled
        (``eqx.nn.inference_mode``), so ``decode`` is purely deterministic.
    """
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = make_galaxy_autoencoder(key=jax_key(0), **config)
    model = eqx.tree_deserialise_leaves(_checkpoint_path(checkpoint_dir, epoch), model)
    return eqx.nn.inference_mode(model, value=True)


def load_frozen_flow(checkpoint_dir: Union[str, Path], epoch: int) -> LatentFlow:
    """Load a pretrained :class:`LatentFlow` in inference mode.

    Args:
        checkpoint_dir: Directory containing ``config.yaml`` and
            ``model_checkpoint_<epoch>.eqx``.
        epoch: Checkpoint epoch to load.

    Returns:
        The flow with weights restored and dropout disabled (if any;
        the flowjax MLP conditioners used here have none by default).
    """
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = make_latent_flow(key=jax_key(0), **config)
    model = eqx.tree_deserialise_leaves(_checkpoint_path(checkpoint_dir, epoch), model)
    return eqx.nn.inference_mode(model, value=True)
