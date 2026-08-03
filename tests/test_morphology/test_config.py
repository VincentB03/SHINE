"""Tests for LearnedMorphologyConfig and its wiring into EuclidInferenceConfig.

These only exercise validation logic, so they use throwaway directories
instead of real checkpoints (kept in test_loader.py, which needs the
actual weights).
"""

import pytest
from pydantic import ValidationError

from shine.euclid.config import EuclidDataConfig, EuclidInferenceConfig
from shine.morphology.config import LearnedMorphologyConfig


def _morphology_config(tmp_path, **overrides):
    kwargs = dict(
        enabled=True,
        ae_checkpoint_dir=str(tmp_path),
        ae_epoch=2000,
        flow_checkpoint_dir=str(tmp_path),
        flow_epoch=50,
        apply_to_stamp_size=64,
    )
    kwargs.update(overrides)
    return LearnedMorphologyConfig(**kwargs)


def _data_config():
    return EuclidDataConfig(
        exposure_paths=["exp.fits"],
        psf_path="psf.fits",
        catalog_path="catalog.fits",
    )


class TestLearnedMorphologyConfig:
    def test_disabled_by_default_requires_no_checkpoint(self):
        # enabled defaults to False; checkpoint fields are still required
        # (no defaults) so the section can't be half-configured silently.
        with pytest.raises(ValidationError):
            LearnedMorphologyConfig()

    def test_rejects_missing_checkpoint_dir(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ValidationError, match="does not exist"):
            _morphology_config(tmp_path, ae_checkpoint_dir=str(missing))

    def test_rejects_nonpositive_epoch(self, tmp_path):
        with pytest.raises(ValidationError):
            _morphology_config(tmp_path, ae_epoch=0)

    def test_checkpoint_path_properties(self, tmp_path):
        cfg = _morphology_config(tmp_path, ae_epoch=2000, flow_epoch=50)
        assert cfg.ae_checkpoint_path == tmp_path / "model_checkpoint_2000.eqx"
        assert cfg.flow_checkpoint_path == tmp_path / "model_checkpoint_50.eqx"

    def test_swapping_checkpoint_dir_and_epoch_is_a_pure_config_change(
        self, tmp_path
    ):
        """The whole point of this config is that pointing at a different
        trained model touches no code — only these two fields."""
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        run_a.mkdir()
        run_b.mkdir()

        cfg_a = _morphology_config(
            tmp_path, ae_checkpoint_dir=str(run_a), ae_epoch=100
        )
        cfg_b = _morphology_config(
            tmp_path, ae_checkpoint_dir=str(run_b), ae_epoch=200
        )

        assert cfg_a.ae_checkpoint_path != cfg_b.ae_checkpoint_path


class TestEuclidInferenceConfigWiring:
    def test_defaults_to_no_learned_morphology(self):
        cfg = EuclidInferenceConfig(data=_data_config())
        assert cfg.learned_morphology is None

    def test_accepts_learned_morphology_matching_a_stamp_tier(self, tmp_path):
        cfg = EuclidInferenceConfig(
            data=_data_config(),
            galaxy_stamp_sizes=[64, 128, 256],
            learned_morphology=_morphology_config(tmp_path, apply_to_stamp_size=64),
        )
        assert cfg.learned_morphology.apply_to_stamp_size == 64

    def test_rejects_stamp_size_not_in_galaxy_stamp_sizes(self, tmp_path):
        with pytest.raises(ValidationError, match="galaxy_stamp_sizes"):
            EuclidInferenceConfig(
                data=_data_config(),
                galaxy_stamp_sizes=[128, 256],
                learned_morphology=_morphology_config(
                    tmp_path, apply_to_stamp_size=64
                ),
            )

    def test_disabled_learned_morphology_skips_stamp_size_check(self, tmp_path):
        # enabled=False must fall back strictly to the parametric renderer,
        # so a stale/mismatched stamp size shouldn't block config loading.
        cfg = EuclidInferenceConfig(
            data=_data_config(),
            galaxy_stamp_sizes=[128, 256],
            learned_morphology=_morphology_config(
                tmp_path, enabled=False, apply_to_stamp_size=64
            ),
        )
        assert cfg.learned_morphology.enabled is False
