from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.model_path_resolver import ModelPathResolutionError, resolve_model_path


class ModelPathResolverTests(unittest.TestCase):
    def test_cli_overrides_env_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cli_dir = root / "cli model"
            env_dir = root / "env_model"
            cfg_dir = root / "cfg_model"
            cli_dir.mkdir()
            env_dir.mkdir()
            cfg_dir.mkdir()
            os.environ["TEST_FLORENCE_MODEL_PATH"] = str(env_dir)
            resolved = resolve_model_path(
                cli_value=cli_dir,
                environment_variable="TEST_FLORENCE_MODEL_PATH",
                config_value=cfg_dir,
                project_root=root,
                required=True,
                expect_directory=True,
            )
            self.assertEqual(resolved, cli_dir.resolve())
        os.environ.pop("TEST_FLORENCE_MODEL_PATH", None)

    def test_project_relative_path_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "models" / "florence"
            model_dir.mkdir(parents=True)
            resolved = resolve_model_path(
                cli_value=None,
                environment_variable=None,
                config_value=Path("models") / "florence",
                project_root=root,
                required=True,
                expect_directory=True,
            )
            self.assertEqual(resolved, model_dir.resolve())

    def test_missing_required_path_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ModelPathResolutionError):
                resolve_model_path(
                    cli_value=None,
                    environment_variable=None,
                    config_value="missing",
                    project_root=Path(tmpdir),
                    required=True,
                    expect_directory=True,
                )


if __name__ == "__main__":
    unittest.main()
