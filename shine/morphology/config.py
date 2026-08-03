"""Configuration for the learned (AutoEncoder + Flow) galaxy morphology.

Every field that identifies a trained model (checkpoint directory, epoch)
lives here as plain config, never hard-coded in ``shine/morphology`` or
``shine/euclid``. Swapping to a newly trained AE/flow pair is a YAML edit,
not a code change.
"""

from pathlib import Path
from typing import Union

from pydantic import BaseModel, field_validator


class LearnedMorphologyConfig(BaseModel):
    """Points to a trained AutoEncoder + Flow pair for one stamp tier.

    Attributes:
        enabled: Whether the learned-morphology path is active at all.
            When False, ``EuclidInferenceConfig`` falls back strictly to
            the existing parametric Sersic/Exponential renderer.
        ae_checkpoint_dir: Directory holding the AutoEncoder's
            ``config.yaml`` + ``model_checkpoint_<ae_epoch>.eqx``.
        ae_epoch: Epoch of the AutoEncoder checkpoint to load.
        flow_checkpoint_dir: Directory holding the LatentFlow's
            ``config.yaml`` + ``model_checkpoint_<flow_epoch>.eqx``. Must
            be a flow trained on latents from the *same* AE checkpoint
            (the flow's own config.yaml records ``ae_run_dir``/
            ``ae_epoch`` for traceability, but this is not cross-checked
            here).
        flow_epoch: Epoch of the flow checkpoint to load.
        apply_to_stamp_size: Stamp tier (in pixels) this AE/flow pair was
            trained for and should replace the parametric renderer on.
            Must match one of the values in
            ``EuclidInferenceConfig.galaxy_stamp_sizes``.
    """

    enabled: bool = False
    ae_checkpoint_dir: str
    ae_epoch: int
    flow_checkpoint_dir: str
    flow_epoch: int
    apply_to_stamp_size: int = 64

    @field_validator("ae_checkpoint_dir", "flow_checkpoint_dir")
    @classmethod
    def validate_checkpoint_dir_exists(cls, v: str) -> str:
        """Validate that the checkpoint directory exists on disk.

        Args:
            v: Checkpoint directory path to validate.

        Returns:
            The validated path.

        Raises:
            ValueError: If the directory does not exist.
        """
        if not Path(v).is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {v}")
        return v

    @field_validator("ae_epoch", "flow_epoch")
    @classmethod
    def validate_epoch_positive(cls, v: int) -> int:
        """Validate that the checkpoint epoch is positive.

        Args:
            v: Epoch value to validate.

        Returns:
            The validated epoch.

        Raises:
            ValueError: If epoch is not positive.
        """
        if v <= 0:
            raise ValueError(f"Checkpoint epoch must be positive, got {v}")
        return v

    @property
    def ae_checkpoint_path(self) -> Path:
        """Path to the AutoEncoder's ``.eqx`` weight file."""
        return Path(self.ae_checkpoint_dir) / f"model_checkpoint_{self.ae_epoch}.eqx"

    @property
    def flow_checkpoint_path(self) -> Path:
        """Path to the flow's ``.eqx`` weight file."""
        return Path(self.flow_checkpoint_dir) / f"model_checkpoint_{self.flow_epoch}.eqx"
