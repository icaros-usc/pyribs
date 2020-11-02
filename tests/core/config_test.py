"""Tests for ribs config functions."""

from ribs.archives import GridArchiveConfig
from ribs.config import create_config, load_configs, save_configs
from ribs.emitters import GaussianEmitterConfig
from ribs.optimizers import OptimizerConfig


def test_create_config_none():
    config = create_config(None, GridArchiveConfig)
    expected = GridArchiveConfig()
    assert config.__dict__ == expected.__dict__


def test_create_config_dict():
    config = create_config({"seed": 42}, GridArchiveConfig)
    expected = GridArchiveConfig(seed=42)
    assert config.__dict__ == expected.__dict__


def test_create_config_object():
    config = create_config(GridArchiveConfig(seed=42), GridArchiveConfig)
    expected = GridArchiveConfig(seed=42)
    assert config.__dict__ == expected.__dict__


def test_save_and_reload_config(tmp_path):
    filename = str(tmp_path / "config.json")
    optimizer_config = OptimizerConfig()
    archive_config = GridArchiveConfig(seed=42)
    emitter_configs = [GaussianEmitterConfig(seed=42, batch_size=32)]

    save_configs(optimizer_config, archive_config, emitter_configs, filename)
    (optimizer_config_2, archive_config_2,
     emitter_configs_2) = load_configs(filename)

    assert optimizer_config.__dict__ == optimizer_config_2.__dict__
    assert archive_config.__dict__ == archive_config_2.__dict__
    assert all(c.__dict__ == c2.__dict__
               for c, c2 in zip(emitter_configs, emitter_configs_2))
