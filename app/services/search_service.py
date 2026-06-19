"""Search Service for indexing events in Qdrant and executing semantic searches.
"""

import uuid
from typing import List, Dict, Any, Optional
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny, MatchValue, Range

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.status_service import JobStatusService

class SearchService:
    """Service to connect to Qdrant, configure collections, index events, and query similar events."""
    
    _client: Optional[QdrantClient] = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        """Retrieve or initialize the Qdrant client."""
        if cls._client is not None:
            return cls._client

        # Use local persistent Qdrant or in-memory fallback
        if settings.ENV == "testing" or settings.MOCK_MODEL:
            logger.info("Initializing local in-memory Qdrant client fallback.")
            cls._client = QdrantClient(location=":memory:")
        elif settings.USE_LOCAL_QDRANT:
            db_path = settings.DATA_DIR / "qdrant_db"
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing local persistent Qdrant client at {db_path}")
            cls._client = QdrantClient(path=str(db_path))
        else:
            logger.info(f"Connecting to remote Qdrant server at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            cls._client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        cls.initialize_collection()
        return cls._client

    @classmethod
    def initialize_collection(cls):
        """Creates the collection in Qdrant if it doesn't already exist."""
        client = cls._client
        if client is None:
            return

        collection_name = settings.QDRANT_COLLECTION
        model_id = settings.EMBEDDING_MODEL_ID.lower()
        vector_size = 1024 if "bge-m3" in model_id else 384

        try:
            if not client.collection_exists(collection_name):
                logger.info(f"Creating Qdrant collection: {collection_name} (size: {vector_size}, distance: COSINE)")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            else:
                logger.debug(f"Qdrant collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    @classmethod
    def index_events(cls, video_id: str, events: List[Dict[str, Any]]) -> bool:
        """Generates embeddings for a list of events and indexes them into Qdrant."""
        if not events:
            logger.warning(f"No events provided to index for video {video_id}.")
            return False

        try:
            client = cls.get_client()
            
            JobStatusService.update(video_id, current_step="Generating embeddings...", progress_percent=90.0)

            # NEW: Generate and include macro-incidents
            from app.services.incident_engine import IncidentEngine
            from app.schemas.summary import AggregatedEvent
            try:
                agg_events = [AggregatedEvent(**e) for e in events]
                chains = IncidentEngine.correlate_events(agg_events)
                for chain in chains:
                    sev_val = 100 if chain.severity == "CRITICAL" else (80 if chain.severity == "HIGH" else (50 if chain.severity == "MEDIUM" else 20))
                    events.append({
                        "event_id": chain.incident_id,
                        "video_id": video_id,
                        "event_type": chain.incident_type,
                        "event_severity": sev_val,
                        "summary": chain.description,
                        "narrative_sentence": " ".join(chain.timeline),
                        "start_time": chain.start_time,
                        "end_time": chain.end_time,
                        "duration_seconds": 0.0,
                        "objects": ["macro_incident"],
                        "activities": [chain.incident_type],
                        "primary_actor": "Macro Incident"
                    })
            except Exception as exc:
                logger.warning(f"Failed to correlate incident chains for search indexing: {exc}")

            # Prepare pristine textual description strings to be embedded (Search Document)
            descriptions = []
            for e in events:
                actor = str(e.get("primary_actor", "Unknown"))
                activities = " ".join(e.get("activities", []))
                
                objs = []
                for obj in e.get("objects", []):
                    if isinstance(obj, dict):
                        objs.append(str(obj.get("subtype", obj.get("type", ""))))
                    else:
                        objs.append(str(obj))
                objects_str = " ".join(objs)
                
                severity_val = str(e.get("event_severity", "unknown"))
                narrative_val = str(e.get("narrative_sentence", e.get("summary", "")))
                event_type_val = str(e.get("event_type", "unknown")).replace("_", " ")

                # Clean deduplicated search document string
                search_doc = f"{actor} {event_type_val} {activities} {objects_str} severity {severity_val} {narrative_val}".replace("  ", " ").strip()
                descriptions.append(search_doc)
            
            # Generate embeddings vectors
            embeddings = EmbeddingService.generate_embeddings(descriptions)
            
            JobStatusService.update(video_id, current_step="Indexing vectors...", progress_percent=95.0)
            
            points = []
            for idx, (event, vector) in enumerate(zip(events, embeddings)):
                # Generate stable UUID based on video_id + event_id to prevent duplicates on re-ingestion
                stable_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{event['event_id']}"))
                
                # Align event parameters to the index payloads
                payload = {
                    "event_id": event["event_id"],
                    "video_id": video_id,
                    "event_type": event["event_type"],
                    "event_severity": event.get("event_severity", 15),
                    "description": event.get("summary", event.get("description", "")),
                    "narrative": event.get("narrative_sentence", event.get("summary", "")),
                    "start_time": event.get("timestamp_start_human", event.get("start_time", "00:00:00")),
                    "end_time": event.get("timestamp_end_human", event.get("end_time", "00:00:00")),
                    "duration_seconds": float(event.get("duration_seconds", 0.0)),
                    "objects": event.get("objects", []),
                    "activities": event.get("activities", []),
                    "primary_actor": event.get("primary_actor", "Unknown"),
                    "thumbnail_path": event.get("thumbnail_path")
                }
                
                points.append(
                    PointStruct(
                        id=stable_id,
                        vector=vector,
                        payload=payload
                    )
                )

            client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points
            )
            logger.info(f"Successfully indexed {len(points)} events in Qdrant for video {video_id}.")
            JobStatusService.update(video_id, status="complete", current_step="Processing complete", progress_percent=100.0)
            return True
            
        except Exception as e:
            logger.error(f"Failed to index events in Qdrant for video {video_id}: {e}")
            JobStatusService.update(video_id, status="failed", current_step="Failed during vector indexing")
            return False

    @classmethod
    def auto_index_existing_events(cls) -> None:
        """Scan the metadata directory for any existing events files and index them into Qdrant."""
        import json
        try:
            logger.info("Scanning metadata directory to auto-index existing events in Qdrant...")
            indexed_count = 0
            for path in settings.METADATA_DIR.glob("*_events.json"):
                stem = path.name
                if stem == "mock-video-id_events.json":
                    video_id = "mock-video-id"
                else:
                    video_id = stem.replace("_events.json", "")
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                    
                    if events:
                        cls.index_events(video_id, events)
                        indexed_count += 1
                except Exception as e:
                    logger.error(f"Failed to auto-index events for file {path.name}: {e}")
            logger.info(f"Auto-indexing complete. Successfully indexed events for {indexed_count} videos.")
        except Exception as e:
            logger.error(f"Error during auto-indexing of existing events: {e}")

    @staticmethod
    def normalize_score(raw: float) -> float:
        """Scales raw BGE-M3 cosine similarity scores (~0.35-0.70) to a user-friendly 0.0-1.0 range."""
        return max(0.0, min(1.0, (raw - 0.35) * 2.0))

    @classmethod
    def search_events(
        cls,
        query: str,
        limit: int = 10,
        video_ids: Optional[List[str]] = None,
        start_after: Optional[str] = None,  # ISO datetime string, inclusive
        end_before: Optional[str] = None,   # ISO datetime string, inclusive
        score_threshold: float = 0.0        # Minimum similarity score (0-1)
    ) -> List[Dict[str, Any]]:
        """Executes a semantic similarity search across video events.

        Optional filters:
        * `video_ids` – restrict to specific videos.
        * `start_after` – only events with `start_time` >= this value.
        * `end_before` – only events with `end_time` <= this value.
        """
        try:
            client = cls.get_client()

            # Vectorize query
            query_vector = EmbeddingService.generate_embeddings(query)

            # Build filter conditions dynamically
            conditions: List[FieldCondition] = []
            if video_ids:
                # Match any of the provided video IDs
                conditions.append(
                    FieldCondition(
                        key="video_id",
                        match=MatchAny(any=video_ids)
                    )
                )
            if start_after:
                conditions.append(
                    FieldCondition(
                        key="start_time",
                        match=Range(gte=start_after)
                    )
                )
            if end_before:
                conditions.append(
                    FieldCondition(
                        key="end_time",
                        match=Range(lte=end_before)
                    )
                )

            search_filter = Filter(must=conditions) if conditions else None

            raw_results = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=search_filter,
                limit=limit * 3,  # fetch extra to allow threshold filtering
                with_payload=True
            ).points

            # Apply score threshold and limit
            filtered = []
            for hit in raw_results:
                normalized = cls.normalize_score(float(hit.score))
                
                event_type = hit.payload.get("event_type", "unknown")
                severity_num = hit.payload.get("event_severity", 0)
                
                # Search Relevance Boosting
                # Apply boost only if base similarity is >= 0.40 to prevent irrelevant events leaping to the top
                if normalized >= 0.40:
                    if event_type == "weapon_drawn":
                        normalized += 1.00
                    elif event_type == "fire_incident":
                        normalized += 0.75
                    elif event_type in ["fall_incident", "medical_emergency", "collision_or_accident", "intrusion", "robbery_incident"]:
                        normalized += 0.50
                
                normalized = min(1.0, normalized)
                
                # Explainability logic
                match_reasons = []
                if event_type not in ["unknown", "normal_activity"]:
                    match_reasons.append(f"{event_type.replace('_', ' ')} detected")
                
                actor = hit.payload.get("primary_actor", "")
                if actor and actor != "Unknown":
                    match_reasons.append(f"actor '{actor}' identified")
                    
                if severity_num >= 70:
                    match_reasons.append("high severity")

                if normalized >= score_threshold:
                    severity_str = "High" if severity_num >= 70 else ("Medium" if severity_num >= 40 else "Low")
                    filtered.append({
                        "score": round(normalized, 3),
                        "event_id": hit.payload.get("event_id"),
                        "video_id": hit.payload.get("video_id"),
                        "event_type": event_type,
                        "severity": severity_str,
                        "description": hit.payload.get("description", ""),
                        "narrative": hit.payload.get("narrative", ""),
                        "start_time": hit.payload.get("start_time"),
                        "end_time": hit.payload.get("end_time"),
                        "duration_seconds": hit.payload.get("duration_seconds"),
                        "objects": hit.payload.get("objects", []),
                        "activities": hit.payload.get("activities", []),
                        "thumbnail_path": hit.payload.get("thumbnail_path"),
                        "match_reasons": match_reasons
                    })
                if len(filtered) >= limit:
                    break

            logger.info(f"Semantic search query '{query}' retrieved {len(filtered)} results after threshold filtering.")
            return filtered

        except Exception as e:
            logger.error(f"Failed to search events in Qdrant: {e}")
            return []
