from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    from .manual_review_export import export_object_reviews_csv, export_summary_markdown, write_json
    from .manual_review_schema import (
        MergeReview,
        ObjectReview,
        PossibleMergeReview,
        WARNING_GROUPS,
        normalize_int_list,
    )
    from .manual_review_summary import build_manual_ground_truth, build_progress, build_summary
    from .video_reader import read_video_metadata
except ImportError:  # pragma: no cover
    from manual_review_export import export_object_reviews_csv, export_summary_markdown, write_json
    from manual_review_schema import (
        MergeReview,
        ObjectReview,
        PossibleMergeReview,
        WARNING_GROUPS,
        normalize_int_list,
    )
    from manual_review_summary import build_manual_ground_truth, build_progress, build_summary
    from video_reader import read_video_metadata


INPUT_FILE_NAMES = [
    "04d2_reconciled_tracks.json",
    "04d2_track_merge_events.json",
    "04d2_reconciliation_candidates.json",
    "04d2_rejected_merge_candidates.json",
    "04d2_track_quality_report.json",
    "05v2_representative_frames.json",
    "05v2_local_identity_packages.json",
    "05v2_local_identity_packages_flat.json",
    "05v2_local_identity_package_report.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(from_track_id: int, to_track_id: int) -> str:
    return f"{int(from_track_id)}->{int(to_track_id)}"


@dataclass
class ReviewPaths:
    root: Path
    screenshots_dir: Path
    debug_clips_dir: Path
    object_reviews_json: Path
    object_reviews_csv: Path
    merge_reviews_json: Path
    possible_merge_reviews_json: Path
    ground_truth_json: Path
    progress_json: Path
    summary_json: Path
    summary_md: Path
    errors_json: Path


class ManualReviewRepository:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        post_tracking_dir: str | Path,
        video_path: str | Path | None = None,
        camera_id: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.post_tracking_dir = Path(post_tracking_dir)
        self._errors: list[dict[str, Any]] = []
        self.inputs = {name: self.post_tracking_dir / name for name in INPUT_FILE_NAMES}
        self._ensure_inputs()
        self.paths = self._build_paths()
        self._ensure_dirs()
        self.reconciled_tracks_payload = _read_json(self.inputs["04d2_reconciled_tracks.json"])
        self.reconciled_tracks = list(self.reconciled_tracks_payload.get("tracks", []))
        self.merge_events_payload = _read_json(self.inputs["04d2_track_merge_events.json"])
        self.accepted_merge_events = list(self.merge_events_payload.get("events", []))
        self.candidates_payload = _read_json(self.inputs["04d2_reconciliation_candidates.json"])
        self.reconciliation_candidates = list(self.candidates_payload.get("candidates", []))
        self.rejected_candidates_payload = _read_json(self.inputs["04d2_rejected_merge_candidates.json"])
        self.rejected_merge_candidates = list(self.rejected_candidates_payload.get("candidates", []))
        self.track_quality_payload = _read_json(self.inputs["04d2_track_quality_report.json"])
        self.track_quality_rows = list(self.track_quality_payload.get("tracks", []))
        self.representative_frames_payload = _read_json(self.inputs["05v2_representative_frames.json"])
        self.representative_frames = list(self.representative_frames_payload.get("objects", []))
        self.packages_payload = _read_json(self.inputs["05v2_local_identity_packages.json"])
        self.packages = list(self.packages_payload.get("packages", []))
        self.package_report = _read_json(self.inputs["05v2_local_identity_package_report.json"])

        self.packages_by_id = {int(item["local_object_id"]): item for item in self.packages}
        self.tracks_by_local_object_id = {int(item["local_object_id"]): item for item in self.reconciled_tracks}
        self.representatives_by_id = {int(item["local_object_id"]): item for item in self.representative_frames}
        self.track_quality_by_track_id = {int(item["track_id"]): item for item in self.track_quality_rows}
        self.accepted_merges_by_local_object_id = {
            int(item["local_object_id"]): item for item in self.accepted_merge_events
        }
        self.raw_track_to_local_object_ids: dict[int, list[int]] = {}
        for package in self.packages:
            local_object_id = int(package["local_object_id"])
            for track_id in package.get("source_raw_track_ids", []):
                self.raw_track_to_local_object_ids.setdefault(int(track_id), []).append(local_object_id)

        self.hybrid_config = _read_json(self.run_dir / "hybrid_tracking_test" / "04c_hybrid_config.json")
        self.hybrid_report = _read_json(self.run_dir / "hybrid_tracking_test" / "04c_hybrid_tracking_report.json")
        detected_video_path = (
            video_path
            or self.hybrid_report.get("video_metadata", {}).get("input_video_path")
            or self.hybrid_config.get("video_path")
        )
        self.video_path = Path(detected_video_path) if detected_video_path else None
        self.camera_id = camera_id or self.packages[0]["camera_id"]
        self.video_metadata = read_video_metadata(self.video_path) if self.video_path and self.video_path.exists() else None

        self.object_reviews = self._load_object_reviews()
        self.merge_reviews = self._load_merge_reviews()
        self.possible_merge_reviews = self._load_possible_merge_reviews()
        self.persist_outputs()

    def _ensure_inputs(self) -> None:
        missing = [str(path) for path in self.inputs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required review inputs: {missing}")

    def _build_paths(self) -> ReviewPaths:
        root = self.post_tracking_dir / "manual_review"
        return ReviewPaths(
            root=root,
            screenshots_dir=root / "screenshots",
            debug_clips_dir=root / "debug_clips",
            object_reviews_json=root / "manual_object_reviews.json",
            object_reviews_csv=root / "manual_object_reviews.csv",
            merge_reviews_json=root / "manual_merge_reviews.json",
            possible_merge_reviews_json=root / "manual_possible_merge_reviews.json",
            ground_truth_json=root / "manual_ground_truth_objects.json",
            progress_json=root / "manual_review_progress.json",
            summary_json=root / "manual_review_summary.json",
            summary_md=root / "manual_review_summary.md",
            errors_json=root / "manual_review_errors.json",
        )

    def _ensure_dirs(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.paths.debug_clips_dir.mkdir(parents=True, exist_ok=True)

    def _load_object_reviews(self) -> dict[int, ObjectReview]:
        if not self.paths.object_reviews_json.exists():
            return {}
        payload = _read_json(self.paths.object_reviews_json)
        return {
            int(item["local_object_id"]): ObjectReview.from_dict(item)
            for item in payload.get("reviews", [])
        }

    def _load_merge_reviews(self) -> dict[int, MergeReview]:
        if not self.paths.merge_reviews_json.exists():
            return {}
        payload = _read_json(self.paths.merge_reviews_json)
        return {
            int(item["local_object_id"]): MergeReview.from_dict(item)
            for item in payload.get("reviews", [])
        }

    def _load_possible_merge_reviews(self) -> dict[str, PossibleMergeReview]:
        if not self.paths.possible_merge_reviews_json.exists():
            return {}
        payload = _read_json(self.paths.possible_merge_reviews_json)
        return {
            str(item["candidate_key"]): PossibleMergeReview.from_dict(item)
            for item in payload.get("reviews", [])
        }

    def available_input_files(self) -> list[str]:
        return [str(path) for path in self.inputs.values() if path.exists()]

    def classify_warnings(self, warnings: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {name: [] for name in WARNING_GROUPS}
        grouped["other"] = []
        for warning in warnings:
            matched = False
            for level, values in WARNING_GROUPS.items():
                if warning in values:
                    grouped[level].append(warning)
                    matched = True
                    break
            if not matched:
                grouped["other"].append(warning)
        return grouped

    def get_all_object_ids(self) -> list[int]:
        return sorted(self.packages_by_id)

    def get_object_record(self, local_object_id: int) -> dict[str, Any]:
        package = dict(self.packages_by_id[int(local_object_id)])
        package["representative_frames_detail"] = self.representatives_by_id.get(int(local_object_id), {}).get(
            "representative_frames",
            {},
        )
        package["track_detail"] = self.tracks_by_local_object_id.get(int(local_object_id), {})
        package["accepted_merge_event"] = self.accepted_merges_by_local_object_id.get(int(local_object_id))
        return package

    def get_object_review(self, local_object_id: int) -> ObjectReview | None:
        return self.object_reviews.get(int(local_object_id))

    def save_object_review(self, payload: dict[str, Any]) -> ObjectReview:
        review = ObjectReview.from_dict(payload)
        self.object_reviews[int(review.local_object_id)] = review
        self.persist_outputs()
        return review

    def save_merge_review(self, payload: dict[str, Any]) -> MergeReview:
        review = MergeReview.from_dict(payload)
        self.merge_reviews[int(review.local_object_id)] = review
        self.persist_outputs()
        return review

    def save_possible_merge_review(self, payload: dict[str, Any]) -> PossibleMergeReview:
        review = PossibleMergeReview.from_dict(payload)
        self.possible_merge_reviews[str(review.candidate_key)] = review
        self.persist_outputs()
        return review

    def build_csv_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for local_object_id in sorted(self.object_reviews):
            review = self.object_reviews[local_object_id]
            package = self.packages_by_id.get(local_object_id, {})
            rows.append(
                {
                    "local_object_id": local_object_id,
                    "manual_real_object_id": review.manual_real_object_id,
                    "object_family": package.get("object_family", ""),
                    "automated_class": package.get("final_class", ""),
                    "manual_class": review.manual_class,
                    "start_timestamp": package.get("start_timestamp_seconds", ""),
                    "end_timestamp": package.get("end_timestamp_seconds", ""),
                    "object_review_status": review.object_review_status,
                    "crop_review_status": review.crop_review_status,
                    "timeline_review_status": review.timeline_review_status,
                    "class_review_status": review.class_review_status,
                    "downstream_decision": review.downstream_decision,
                    "same_real_object_as": ",".join(str(item) for item in review.same_real_object_as_local_object_ids),
                    "reviewer_notes": review.reviewer_notes,
                    "reviewed_at": review.reviewed_at,
                }
            )
        return rows

    def filter_object_ids(
        self,
        *,
        object_family: str = "all",
        final_class: str = "all",
        downstream_status: str = "all",
        warning: str = "all",
        only_unreviewed: bool = False,
        only_merged: bool = False,
        only_possible_merges: bool = False,
    ) -> list[int]:
        possible_merge_local_ids = {
            local_object_id
            for candidate in self.get_possible_merge_candidates()
            for local_object_id in candidate.get("related_local_object_ids", [])
        }
        result: list[int] = []
        for package in self.packages:
            local_object_id = int(package["local_object_id"])
            if object_family != "all" and package.get("object_family") != object_family:
                continue
            if final_class != "all" and package.get("final_class") != final_class:
                continue
            if downstream_status != "all" and package.get("downstream_status") != downstream_status:
                continue
            if warning != "all" and warning not in list(package.get("warnings", [])):
                continue
            if only_unreviewed and local_object_id in self.object_reviews:
                continue
            if only_merged and package.get("reconciliation_status") != "merged":
                continue
            if only_possible_merges and local_object_id not in possible_merge_local_ids:
                continue
            result.append(local_object_id)
        return result

    def get_accepted_merge_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.accepted_merge_events:
            local_object_id = int(event["local_object_id"])
            package = self.packages_by_id.get(local_object_id, {})
            rows.append(
                {
                    **event,
                    "package": package,
                    "review": self.merge_reviews.get(local_object_id),
                }
            )
        return rows

    def _build_track_reference(self, track_id: int) -> dict[str, Any]:
        quality = self.track_quality_by_track_id.get(int(track_id), {})
        related_local_ids = list(self.raw_track_to_local_object_ids.get(int(track_id), []))
        local_object_id = related_local_ids[0] if related_local_ids else None
        package = self.packages_by_id.get(local_object_id, {}) if local_object_id is not None else {}
        representative = self.representatives_by_id.get(local_object_id, {}) if local_object_id is not None else {}
        frames = representative.get("representative_frames", {})
        primary = frames.get("primary", {}) if isinstance(frames, dict) else {}
        return {
            "track_id": int(track_id),
            "local_object_id": local_object_id,
            "object_family": quality.get("object_family") or package.get("object_family"),
            "final_class": quality.get("final_class") or package.get("final_class"),
            "crop_path": primary.get("crop_path"),
            "full_frame_path": primary.get("full_frame_path"),
            "start_timestamp_seconds": package.get("start_timestamp_seconds"),
            "end_timestamp_seconds": package.get("end_timestamp_seconds"),
            "integrity_status": quality.get("integrity_status"),
            "quality_level": quality.get("quality_level"),
            "quality_score": quality.get("quality_score"),
        }

    def get_possible_merge_candidates(self) -> list[dict[str, Any]]:
        accepted_pairs = {
            _candidate_key(item["source_track_ids"][0], item["source_track_ids"][1])
            for item in self.accepted_merge_events
            if len(item.get("source_track_ids", [])) >= 2
        }
        rows: list[dict[str, Any]] = []
        for candidate in self.reconciliation_candidates:
            candidate_key = _candidate_key(candidate["from_track_id"], candidate["to_track_id"])
            if candidate_key in accepted_pairs:
                continue
            raw_local_ids = [
                *self.raw_track_to_local_object_ids.get(int(candidate["from_track_id"]), []),
                *self.raw_track_to_local_object_ids.get(int(candidate["to_track_id"]), []),
            ]
            rows.append(
                {
                    **candidate,
                    "candidate_key": candidate_key,
                    "from_track": self._build_track_reference(int(candidate["from_track_id"])),
                    "to_track": self._build_track_reference(int(candidate["to_track_id"])),
                    "related_local_object_ids": sorted(set(int(item) for item in raw_local_ids)),
                    "review": self.possible_merge_reviews.get(candidate_key),
                }
            )
        return rows

    def default_object_review_payload(self, local_object_id: int) -> dict[str, Any]:
        package = self.packages_by_id[int(local_object_id)]
        existing = self.object_reviews.get(int(local_object_id))
        suggested_group = f"{package['object_family']}_{int(local_object_id):04d}"
        payload = existing.to_dict() if existing else {}
        payload.setdefault("local_object_id", int(local_object_id))
        payload.setdefault("camera_id", str(package["camera_id"]))
        payload.setdefault("manual_real_object_id", suggested_group)
        payload.setdefault("object_review_status", "uncertain")
        payload.setdefault("crop_review_status", "crop_uncertain")
        payload.setdefault("timeline_review_status", "timeline_uncertain")
        payload.setdefault("class_review_status", "class_uncertain")
        payload.setdefault("downstream_decision", "manual_review")
        payload.setdefault("manual_class", package.get("final_class", ""))
        payload.setdefault("same_real_object_as_local_object_ids", [])
        payload.setdefault("suggested_real_object_group", suggested_group)
        payload.setdefault("false_detection_reason", "")
        payload.setdefault("switch_timestamp_seconds", None)
        payload.setdefault("switch_original_object", "")
        payload.setdefault("switch_new_object", "")
        payload.setdefault("track_should_be_split", None)
        payload.setdefault("reviewer_notes", "")
        return payload

    def persist_outputs(self) -> None:
        object_review_rows = [self.object_reviews[key].to_dict() for key in sorted(self.object_reviews)]
        merge_review_rows = [self.merge_reviews[key].to_dict() for key in sorted(self.merge_reviews)]
        possible_merge_rows = [
            self.possible_merge_reviews[key].to_dict()
            for key in sorted(self.possible_merge_reviews)
        ]
        manual_ground_truth = build_manual_ground_truth(
            camera_id=str(self.camera_id),
            video_name=self.video_path.name if self.video_path else "",
            packages=self.packages,
            object_reviews=object_review_rows,
        )
        progress = build_progress(
            total_objects=len(self.packages),
            packages=self.packages,
            object_reviews=object_review_rows,
            merge_reviews=merge_review_rows,
            possible_merge_reviews=possible_merge_rows,
            accepted_merge_count=len(self.accepted_merge_events),
            possible_merge_count=len(self.get_possible_merge_candidates()),
        )
        summary = build_summary(
            packages=self.packages,
            object_reviews=object_review_rows,
            merge_reviews=merge_review_rows,
            possible_merge_reviews=possible_merge_rows,
            accepted_merge_count=len(self.accepted_merge_events),
            possible_merge_count=len(self.get_possible_merge_candidates()),
            manual_ground_truth=manual_ground_truth,
        )
        write_json(self.paths.object_reviews_json, {"status": "success", "reviews": object_review_rows})
        write_json(self.paths.merge_reviews_json, {"status": "success", "reviews": merge_review_rows})
        write_json(
            self.paths.possible_merge_reviews_json,
            {"status": "success", "reviews": possible_merge_rows},
        )
        write_json(self.paths.ground_truth_json, manual_ground_truth)
        write_json(self.paths.progress_json, progress)
        write_json(self.paths.summary_json, summary)
        write_json(self.paths.errors_json, {"status": "success", "errors": self._errors})
        export_object_reviews_csv(self.build_csv_rows(), self.paths.object_reviews_csv)
        export_summary_markdown(summary, self.paths.summary_md)

    def get_progress(self) -> dict[str, Any]:
        return _read_json(self.paths.progress_json) if self.paths.progress_json.exists() else {}

    def get_summary(self) -> dict[str, Any]:
        return _read_json(self.paths.summary_json) if self.paths.summary_json.exists() else {}

    def get_clip_window(self, local_object_id: int) -> tuple[float, float]:
        package = self.packages_by_id[int(local_object_id)]
        clip_start = max(0.0, float(package["start_timestamp_seconds"]) - 2.0)
        if self.video_metadata is None:
            clip_end = float(package["end_timestamp_seconds"]) + 2.0
        else:
            clip_end = min(self.video_metadata.duration_seconds, float(package["end_timestamp_seconds"]) + 2.0)
        return clip_start, clip_end

    def ensure_review_clip(self, local_object_id: int) -> dict[str, Any]:
        if self.video_path is None or not self.video_path.exists():
            return {
                "status": "missing_video",
                "video_path": str(self.video_path) if self.video_path else "",
                "clip_path": None,
                "message": "Source video is unavailable.",
            }
        if self.video_metadata is None:
            return {
                "status": "missing_video_metadata",
                "video_path": str(self.video_path),
                "clip_path": None,
                "message": "Video metadata could not be read.",
            }
        clip_start, clip_end = self.get_clip_window(local_object_id)
        clip_name = f"object_{int(local_object_id):06d}_review_{clip_start:07.1f}_{clip_end:07.1f}.mp4"
        clip_path = self.paths.debug_clips_dir / clip_name
        if clip_path.exists():
            return {
                "status": "cached",
                "video_path": str(self.video_path),
                "clip_path": str(clip_path),
                "clip_start": clip_start,
                "clip_end": clip_end,
            }
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            message = f"Failed to open source video: {self.video_path}"
            self._errors.append({"local_object_id": local_object_id, "error": message})
            self.persist_outputs()
            return {
                "status": "error",
                "video_path": str(self.video_path),
                "clip_path": None,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "message": message,
            }
        fps = self.video_metadata.fps
        width = self.video_metadata.width
        height = self.video_metadata.height
        start_frame = max(0, int(clip_start * fps))
        end_frame = min(self.video_metadata.frame_count - 1, int(clip_end * fps))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame
        while frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index += 1
        writer.release()
        capture.release()
        if not clip_path.exists():
            message = f"Clip generation did not create output for object {local_object_id}"
            self._errors.append({"local_object_id": local_object_id, "error": message})
            self.persist_outputs()
            return {
                "status": "error",
                "video_path": str(self.video_path),
                "clip_path": None,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "message": message,
            }
        return {
            "status": "created",
            "video_path": str(self.video_path),
            "clip_path": str(clip_path),
            "clip_start": clip_start,
            "clip_end": clip_end,
        }
