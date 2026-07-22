from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..database.client import create_backend_client
from ..database.config import DatabaseConfig, DatabaseConfigError
from ..ingestion.camera_config import CameraConfig, load_camera_configs
from ..ingestion.frame_packet import FramePacket
from ..ingestion.multi_camera_reader import MultiCameraReader

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OrchestratorRunResult:
    report: dict[str, object]


class MultiCameraOrchestrator:
    def __init__(self, config_path: str | Path, *, mode: str = "round_robin", max_frames_per_camera: int | None = None) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.mode = mode
        self.max_frames_per_camera = max_frames_per_camera

    def load_enabled_cameras(self) -> list[CameraConfig]:
        return load_camera_configs(self.config_path, include_disabled=False, validate_paths=True)

    def sync_cameras_to_db(self, camera_configs: list[CameraConfig], *, insert_missing: bool) -> None:
        try:
            config = DatabaseConfig.from_env(require_backend_credentials=True)
        except DatabaseConfigError as exc:
            raise RuntimeError("--sync-cameras-to-db was requested but required database settings are missing.") from exc
        client = create_backend_client(config)
        try:
            existing_response = client.table("cameras").select("camera_code").execute()
            existing_codes = {row["camera_code"] for row in getattr(existing_response, "data", []) or []}
            missing_rows = [
                {
                    "camera_code": camera.camera_code,
                    "camera_name": camera.camera_name,
                    "source_path": str(camera.source_path),
                    "enabled": camera.enabled,
                }
                for camera in camera_configs
                if camera.camera_code not in existing_codes
            ]
            if missing_rows and insert_missing:
                client.table("cameras").insert(missing_rows).execute()
                LOGGER.info("Inserted %s camera rows into Supabase.", len(missing_rows))
            elif missing_rows:
                raise RuntimeError(f"Cameras missing from database: {', '.join(item['camera_code'] for item in missing_rows)}")
        except Exception as exc:
            raise RuntimeError(f"Failed to sync camera metadata to Supabase: {exc}") from exc

    def run(
        self,
        *,
        preview_callback: Callable[[FramePacket], bool] | None = None,
        sync_cameras_to_db: bool = False,
    ) -> OrchestratorRunResult:
        camera_configs = self.load_enabled_cameras()
        if sync_cameras_to_db:
            self.sync_cameras_to_db(camera_configs, insert_missing=True)

        camera_stats: dict[str, dict[str, object]] = {
            config.camera_code: {
                "camera_name": config.camera_name,
                "source_path": str(config.source_path),
                "frames_read": 0,
                "first_frame_number": None,
                "last_frame_number": None,
                "first_video_time_seconds": None,
                "last_video_time_seconds": None,
                "source_fps": None,
                "video_duration": None,
                "read_errors": 0,
                "errors": [],
                "end_of_stream_reached": False,
            }
            for config in camera_configs
        }
        total_frames = 0

        reader = MultiCameraReader(camera_configs, mode=self.mode, max_frames_per_camera=self.max_frames_per_camera)
        try:
            reader.open()
            for source in reader.sources:
                metadata = source.metadata()
                camera_stats[metadata.camera_code]["source_fps"] = metadata.source_fps
                camera_stats[metadata.camera_code]["video_duration"] = metadata.duration_seconds
            for packet in reader:
                total_frames += 1
                stats = camera_stats[packet.camera_code]
                stats["frames_read"] = int(stats["frames_read"]) + 1
                if stats["first_frame_number"] is None:
                    stats["first_frame_number"] = packet.frame_number
                    stats["first_video_time_seconds"] = packet.video_time_seconds
                stats["last_frame_number"] = packet.frame_number
                stats["last_video_time_seconds"] = packet.video_time_seconds
                if preview_callback is not None and not preview_callback(packet):
                    break
        finally:
            for source in reader.sources:
                camera_stats[source.config.camera_code]["end_of_stream_reached"] = True
            reader.close()

        report = {
            "mode": self.mode,
            "camera_count": len(camera_configs),
            "total_frames": total_frames,
            "cameras": camera_stats,
            "generated_at": datetime.now().isoformat(),
        }
        LOGGER.info("Multi-camera input validation complete mode=%s camera_count=%s total_frames=%s", self.mode, len(camera_configs), total_frames)
        return OrchestratorRunResult(report=report)
