from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from ..persistence.cross_camera_match_repository import CrossCameraMatchRepository
from ..persistence.global_vehicle_object_repository import GlobalVehicleObjectRepository
from .global_match_config import CameraRouteRule, GlobalMatchConfig
from .global_match_models import CrossCameraMatchResult, GlobalObjectMembership, GlobalVehicleObjectProposal, TrackIdentityFeatures
from .global_match_scoring import evaluate_track_pair


@dataclass(slots=True)
class GlobalMatchBuildReport:
    run_code: str
    mode: str
    rule_version: str
    tracks_loaded: int
    candidate_count: int
    decisions: dict[str, int]
    global_objects: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    writes_performed: int = 0
    tracks_with_verified_plates: int = 0
    tracks_with_colour: int = 0
    single_track_objects: int = 0
    multi_camera_objects: int = 0
    memberships_proposed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_code": self.run_code,
            "mode": self.mode,
            "rule_version": self.rule_version,
            "tracks_loaded": self.tracks_loaded,
            "tracks_with_verified_plates": self.tracks_with_verified_plates,
            "tracks_with_colour": self.tracks_with_colour,
            "candidate_count": self.candidate_count,
            "decisions": self.decisions,
            "global_objects": self.global_objects,
            "matches": self.matches,
            "errors": self.errors,
            "writes_performed": self.writes_performed,
            "single_track_objects": self.single_track_objects,
            "multi_camera_objects": self.multi_camera_objects,
            "memberships_proposed": self.memberships_proposed,
        }


class GlobalMatchService:
    def __init__(
        self,
        config: GlobalMatchConfig,
        match_repository: CrossCameraMatchRepository,
        global_object_repository: GlobalVehicleObjectRepository,
    ) -> None:
        self._config = config
        self._match_repository = match_repository
        self._global_object_repository = global_object_repository

    def build_for_run(self, run_code: str, *, persist: bool = False) -> GlobalMatchBuildReport:
        run_row = self._match_repository.find_run_by_code(run_code)
        if run_row is None:
            raise RuntimeError(f"Processing run not found: {run_code}")
        processing_run_id = str(run_row["id"])
        tracks = self._match_repository.find_tracks_for_run(processing_run_id)
        route_by_pair = self._route_rules_for(tracks)
        candidate_pairs = self._generate_candidate_pairs(tracks, route_by_pair)
        results: list[CrossCameraMatchResult] = []
        decisions = {"confirmed": 0, "possible": 0, "rejected": 0, "insufficient_evidence": 0, "review_required": 0}
        for left, right in candidate_pairs:
            route_rule = route_by_pair.get((left.camera_code, right.camera_code))
            evaluation = evaluate_track_pair(left, right, self._config, route_rule)
            results.append(evaluation.result)
            decisions[evaluation.result.decision.lower()] = decisions.get(evaluation.result.decision.lower(), 0) + 1
        proposals = self._build_object_proposals(run_code, processing_run_id, tracks, results)
        report = GlobalMatchBuildReport(
            run_code=run_code,
            mode="persist" if persist else "dry_run",
            rule_version=self._config.rule_version,
            tracks_loaded=len(tracks),
            candidate_count=len(candidate_pairs),
            decisions=decisions,
            tracks_with_verified_plates=len([track for track in tracks if str(track.plate_status or "").upper() == "VERIFIED"]),
            tracks_with_colour=len([track for track in tracks if str(track.canonical_colour or "").strip()]),
            single_track_objects=len([item for item in proposals if item.track_count == 1]),
            multi_camera_objects=len([item for item in proposals if item.camera_count > 1]),
            memberships_proposed=sum(len(item.members) for item in proposals),
        )
        report.matches = [self._serialize_match(result) for result in results]
        report.global_objects = [self._serialize_object(item) for item in proposals]
        if not persist:
            return report
        for proposal in proposals:
            object_row = self._global_object_repository.create_or_get_global_object(proposal)
            global_vehicle_id = str(object_row.get("id") or "")
            for membership in proposal.members:
                self._global_object_repository.add_or_update_member(global_vehicle_id, membership)
                report.writes_performed += 1
            for result in results:
                member_ids = {member.vehicle_track_id for member in proposal.members}
                if {result.left_vehicle_track_id, result.right_vehicle_track_id}.issubset(member_ids):
                    self._match_repository.upsert_match(processing_run_id, result, global_vehicle_id=global_vehicle_id)
                    report.writes_performed += 1
        return report

    def _route_rules_for(self, tracks: list[TrackIdentityFeatures]) -> dict[tuple[str, str], CameraRouteRule]:
        routes: dict[tuple[str, str], CameraRouteRule] = {}
        for left, right in combinations(sorted({track.camera_code for track in tracks}), 2):
            rule = self._config.route_for(left, right)
            if rule is not None:
                routes[(left, right)] = rule
                routes[(right, left)] = rule
        return routes

    def _generate_candidate_pairs(self, tracks: list[TrackIdentityFeatures], route_by_pair: dict[tuple[str, str], CameraRouteRule]) -> list[tuple[TrackIdentityFeatures, TrackIdentityFeatures]]:
        pairs: dict[tuple[str, str], tuple[TrackIdentityFeatures, TrackIdentityFeatures]] = {}

        def add_pair(left: TrackIdentityFeatures, right: TrackIdentityFeatures) -> None:
            ordered = (left, right) if left.vehicle_track_id <= right.vehicle_track_id else (right, left)
            if ordered[0].vehicle_track_id == ordered[1].vehicle_track_id:
                return
            if not self._config.same_camera_matching and ordered[0].camera_id == ordered[1].camera_id:
                return
            pairs[(ordered[0].vehicle_track_id, ordered[1].vehicle_track_id)] = ordered

        verified_by_plate: dict[str, list[TrackIdentityFeatures]] = {}
        verified_tracks: list[TrackIdentityFeatures] = []
        no_plate_tracks: list[TrackIdentityFeatures] = []
        supportive_tracks: list[TrackIdentityFeatures] = []
        for track in tracks:
            status = str(track.plate_status or "").upper()
            if status == "VERIFIED" and track.normalized_plate:
                verified_tracks.append(track)
                verified_by_plate.setdefault(str(track.normalized_plate).upper(), []).append(track)
            elif not track.normalized_plate:
                no_plate_tracks.append(track)
            supportive_tracks.append(track)
        for group in verified_by_plate.values():
            for left, right in combinations(group, 2):
                add_pair(left, right)
        for verified_track in verified_tracks:
            for candidate in no_plate_tracks:
                if verified_track.vehicle_track_id == candidate.vehicle_track_id:
                    continue
                if self._class_or_colour_support(verified_track, candidate):
                    add_pair(verified_track, candidate)
        for left, right in combinations(supportive_tracks, 2):
            if self._class_or_colour_support(left, right):
                route_rule = route_by_pair.get((left.camera_code, right.camera_code))
                if route_rule is None or route_rule.allowed:
                    add_pair(left, right)
        return list(pairs.values())

    def _class_or_colour_support(self, left: TrackIdentityFeatures, right: TrackIdentityFeatures) -> bool:
        same_class = bool(left.canonical_class and right.canonical_class and str(left.canonical_class).upper() == str(right.canonical_class).upper())
        same_colour = bool(left.canonical_colour and right.canonical_colour and str(left.canonical_colour).upper() == str(right.canonical_colour).upper())
        return same_class or same_colour

    def _build_object_proposals(
        self,
        run_code: str,
        processing_run_id: str,
        tracks: list[TrackIdentityFeatures],
        results: list[CrossCameraMatchResult],
    ) -> list[GlobalVehicleObjectProposal]:
        confirmed_edges = [(result.left_vehicle_track_id, result.right_vehicle_track_id, result) for result in results if result.decision == "CONFIRMED"]
        track_by_id = {track.vehicle_track_id: track for track in tracks}
        adjacency: dict[str, set[str]] = {track.vehicle_track_id: set() for track in tracks}
        result_by_pair: dict[frozenset[str], CrossCameraMatchResult] = {}
        for left_id, right_id, result in confirmed_edges:
            adjacency.setdefault(left_id, set()).add(right_id)
            adjacency.setdefault(right_id, set()).add(left_id)
            result_by_pair[frozenset((left_id, right_id))] = result
        visited: set[str] = set()
        proposals: list[GlobalVehicleObjectProposal] = []
        for track in tracks:
            if track.vehicle_track_id in visited:
                continue
            component: list[TrackIdentityFeatures] = []
            stack = [track.vehicle_track_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(track_by_id[current])
                stack.extend(sorted(adjacency.get(current, set()) - visited))
            if len(component) == 1 and not self._config.create_single_track_global_objects:
                continue
            proposals.append(self._proposal_from_component(run_code, processing_run_id, component, result_by_pair))
        proposals.sort(key=lambda item: item.global_object_code)
        return proposals

    def _proposal_from_component(
        self,
        run_code: str,
        processing_run_id: str,
        component: list[TrackIdentityFeatures],
        result_by_pair: dict[frozenset[str], CrossCameraMatchResult],
    ) -> GlobalVehicleObjectProposal:
        component_sorted = sorted(component, key=lambda item: item.track_uuid)
        track_ids = {item.vehicle_track_id for item in component_sorted}
        track_uuids = [item.track_uuid for item in component_sorted]
        digest = hashlib.sha1("|".join(track_uuids).encode("utf-8")).hexdigest()[:12].upper()
        matching_results = [result for pair, result in result_by_pair.items() if pair.issubset(track_ids)]
        confirmed = any(result.decision == "CONFIRMED" for result in matching_results)
        canonical_plate = self._canonical_verified_plate(component_sorted)
        canonical_colour = next((track.canonical_colour for track in component_sorted if track.canonical_colour), None)
        canonical_class = next((track.canonical_class for track in component_sorted if track.canonical_class), None)
        confidence = max((result.score for result in matching_results), default=0.60 if len(component_sorted) == 1 else 0.85)
        members = tuple(
            GlobalObjectMembership(
                vehicle_track_id=track.vehicle_track_id,
                track_uuid=track.track_uuid,
                membership_status="CONFIRMED" if confirmed or len(component_sorted) == 1 else "POSSIBLE",
                membership_confidence=confidence,
                match_method="VERIFIED_PLATE" if canonical_plate and confirmed else "SINGLE_TRACK" if len(component_sorted) == 1 else "RULE_BASED",
                metadata={"camera_code": track.camera_code},
            )
            for track in component_sorted
        )
        return GlobalVehicleObjectProposal(
            processing_run_id=processing_run_id,
            global_object_code=f"GVO:{run_code}:{digest}",
            status="CONFIRMED" if confirmed else "ACTIVE" if len(component_sorted) == 1 else "POSSIBLE",
            confidence=confidence,
            canonical_plate=canonical_plate,
            canonical_colour=canonical_colour,
            canonical_vehicle_class=canonical_class,
            first_seen_at=min((track.first_seen_at for track in component_sorted if track.first_seen_at is not None), default=None),
            last_seen_at=max((track.last_seen_at for track in component_sorted if track.last_seen_at is not None), default=None),
            creation_method="VERIFIED_PLATE" if canonical_plate and confirmed else "SINGLE_TRACK" if len(component_sorted) == 1 else "RULE_BASED",
            camera_count=len({track.camera_id for track in component_sorted}),
            track_count=len(component_sorted),
            members=members,
            metadata={"member_track_uuids": track_uuids},
        )

    def _canonical_verified_plate(self, tracks: list[TrackIdentityFeatures]) -> str | None:
        verified_plates = {
            str(track.normalized_plate).upper()
            for track in tracks
            if str(track.plate_status or "").upper() == "VERIFIED" and str(track.normalized_plate or "").strip()
        }
        if len(verified_plates) == 1:
            return next(iter(verified_plates))
        return None

    def _serialize_match(self, result: CrossCameraMatchResult) -> dict[str, Any]:
        return {
            "left_track_uuid": result.left_track_uuid,
            "right_track_uuid": result.right_track_uuid,
            "decision": result.decision,
            "score": result.score,
            "plate_score": result.plate_score,
            "time_score": result.time_score,
            "camera_route_score": result.camera_route_score,
            "class_score": result.class_score,
            "colour_score": result.colour_score,
            "visual_score": result.visual_score,
            "reasons": list(result.reasons),
            "rule_version": result.rule_version,
        }

    def _serialize_object(self, proposal: GlobalVehicleObjectProposal) -> dict[str, Any]:
        return {
            "global_object_code": proposal.global_object_code,
            "status": proposal.status,
            "confidence": proposal.confidence,
            "canonical_plate": proposal.canonical_plate,
            "canonical_colour": proposal.canonical_colour,
            "canonical_vehicle_class": proposal.canonical_vehicle_class,
            "camera_count": proposal.camera_count,
            "track_count": proposal.track_count,
            "creation_method": proposal.creation_method,
            "members": [
                {
                    "vehicle_track_id": member.vehicle_track_id,
                    "track_uuid": member.track_uuid,
                    "membership_status": member.membership_status,
                    "membership_confidence": member.membership_confidence,
                    "match_method": member.match_method,
                }
                for member in proposal.members
            ],
        }
