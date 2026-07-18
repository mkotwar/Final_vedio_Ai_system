from __future__ import annotations

from collections import Counter
from typing import Any

from .search_result_card_schemas import VehicleResultCardPackage


def build_result_card_metrics(packages: list[VehicleResultCardPackage], *, packaging_runtime: float) -> dict[str, Any]:
    cards = [card for package in packages for card in package.cards]
    duplicate_ids: list[str] = []
    for package in packages:
        package_ids = [card.record_id for card in package.cards]
        duplicate_ids.extend(
            f"{package.raw_query}:{record_id}"
            for record_id, count in Counter(package_ids).items()
            if count > 1
        )
    status_counts = Counter(card.plate_status for card in cards)
    return {
        "queries_packaged": len(packages),
        "cards_created": len(cards),
        "cards_with_vehicle_image": sum(1 for card in cards if card.thumbnail_path),
        "cards_with_plate_image": sum(1 for card in cards if card.secondary_image_path),
        "cards_missing_vehicle_image": sum(1 for card in cards if not card.thumbnail_path),
        "cards_missing_plate_image": sum(1 for card in cards if not card.secondary_image_path),
        "verified_plate_cards": status_counts["verified"],
        "weak_plate_cards": status_counts["weak"],
        "no_plate_cards": status_counts["no_plate_detected"],
        "invalid_plate_cards": status_counts["invalid"],
        "cards_by_class": dict(sorted(Counter(str(card.object_class or "unknown") for card in cards).items())),
        "cards_by_colour": dict(sorted(Counter(str(card.colour or "unknown") for card in cards).items())),
        "cards_by_status": dict(sorted(status_counts.items())),
        "duplicate_card_ids": duplicate_ids,
        "packaging_runtime": round(packaging_runtime, 6),
    }
