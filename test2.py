import sys
import traceback
from pathlib import Path

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

try:
    from app.services.embedding_service import EmbeddingService
    from app.services.search_service import SearchService
    
    out = []
    out.append("Initializing backend services...")
    EmbeddingService.initialize()
    SearchService.get_client()
    
    out.append("Re-indexing events to apply new pristine search documents...")
    SearchService.auto_index_existing_events()
    
    queries = ["fall", "medical emergency", "vehicle crash", "fire", "person lying on floor"]
    
    out.append("\n" + "="*50)
    out.append("RUNNING SEMANTIC SEARCH QUALITY VALIDATION")
    out.append("="*50)
    
    for query in queries:
        out.append(f"\n[QUERY]: '{query}'")
        hits = SearchService.search_events(query=query, limit=3, score_threshold=0.30)
        
        if not hits:
            out.append("  -> No results found.")
            continue
            
        for i, hit in enumerate(hits):
            out.append(f"  {i+1}. Score: {hit['score']:.3f} | Type: {hit['event_type']} | Severity: {hit.get('severity', 'Low')}")
            out.append(f"     Narrative: {hit.get('narrative', hit.get('description', ''))}")
            out.append(f"     Explainability: {', '.join(hit.get('match_reasons', []))}")
            out.append("")
    
    with open(ROOT / "search_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
except Exception as e:
    with open(ROOT / "search_results.txt", "w", encoding="utf-8") as f:
        f.write("FATAL ERROR:\n")
        f.write(traceback.format_exc())
