from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.models.model_path_resolver import (
    ModelPathResolutionError,
    resolve_model_path,
    resolve_model_path_with_source,
)


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

    def test_env_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_dir = root / "env_model"
            cfg_dir = root / "cfg_model"
            env_dir.mkdir()
            cfg_dir.mkdir()
            os.environ["TEST_FLORENCE_MODEL_PATH"] = str(env_dir)
            resolved = resolve_model_path_with_source(
                cli_value=None,
                environment_variable="TEST_FLORENCE_MODEL_PATH",
                config_value=cfg_dir,
                project_root=root,
                required=True,
                expect_directory=True,
            )
            self.assertEqual(resolved.path, env_dir.resolve())
            self.assertEqual(resolved.source, "env:TEST_FLORENCE_MODEL_PATH")
        os.environ.pop("TEST_FLORENCE_MODEL_PATH", None)

    def test_default_is_used_after_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            default_dir = root / "models" / "florence"
            default_dir.mkdir(parents=True)
            resolved = resolve_model_path_with_source(
                cli_value=None,
                environment_variable=None,
                config_value=None,
                default_value=Path("models") / "florence",
                project_root=root,
                required=True,
                expect_directory=True,
            )
            self.assertEqual(resolved.path, default_dir.resolve())
            self.assertEqual(resolved.source, "default")

    def test_path_with_spaces_resolves(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex model ") as tmpdir:
            root = Path(tmpdir)
            model_file = root / "models" / "plate detection" / "license_plate_weights.pt"
            model_file.parent.mkdir(parents=True)
            model_file.write_bytes(b"x")
            resolved = resolve_model_path(
                cli_value=model_file,
                environment_variable=None,
                config_value=None,
                project_root=root,
                required=True,
                expect_directory=False,
            )
            self.assertEqual(resolved, model_file.resolve())

    def test_missing_required_path_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ModelPathResolutionError):
                resolve_model_path(
                    cli_value=None,
                    environment_variable=None,
                    config_value="missing",
                    default_value=Path("models") / "missing",
                    project_root=Path(tmpdir),
                    required=True,
                    expect_directory=True,
                )


if __name__ == "__main__":
    unittest.main()
