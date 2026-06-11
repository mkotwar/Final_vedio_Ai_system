import sys
import json
from pathlib import Path
from pprint import pprint

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService

print("Initializing backend services...")
EmbeddingService.initialize()
SearchService.get_client()

print("Re-indexing events to apply new pristine search documents...")
SearchService.auto_index_existing_events()

queries = [
    "fall",
    "person fell",
    "medical emergency",
    "vehicle crash",
    "accident",
    "fire",
    "intruder",
    "person lying on floor"
]

with open(ROOT / "search_results.txt", "w", encoding="utf-8") as f:
    f.write("RUNNING SEMANTIC SEARCH QUALITY VALIDATION\n")
    f.write("="*50 + "\n")
    for query in queries:
        f.write(f"\n[QUERY]: '{query}'\n")
        hits = SearchService.search_events(query=query, limit=3, score_threshold=0.30)
        
        if not hits:
            f.write("  -> No results found.\n")
            continue
            
        for i, hit in enumerate(hits):
            f.write(f"  {i+1}. Score: {hit['score']:.3f} | Type: {hit['event_type']} | Severity: {hit.get('severity', 'Low')} | Actor: {hit.get('actor', '')}\n")
            f.write(f"     Narrative: {hit.get('narrative', hit.get('description', ''))}\n")
            f.write(f"     Explainability: {', '.join(hit.get('match_reasons', []))}\n\n")
