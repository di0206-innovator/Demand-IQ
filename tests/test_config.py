"""Tests for DemandIQ configuration."""

from demandiq.config import DEFAULT_CONFIG, AppConfig, PathConfig


def test_default_config_instantiation():
    config = AppConfig()
    assert config.paths.data_raw.name == "raw"
    assert config.paths.data_processed.name == "processed"
    assert config.split.train_end_week < config.split.val_end_week < config.split.total_weeks
    assert 0.0 < config.inventory.service_level < 1.0


def test_path_config_resolution():
    paths = PathConfig()
    assert paths.project_root.exists()
    assert (paths.project_root / "pyproject.toml").exists()
    assert DEFAULT_CONFIG.random_seed == 42
