import sys
import os
sys.path.append("c:\\Mukul K\\vinfo1\\video-search-engine")
from app.services.search_service import SearchService

queries = ["red car", "person running", "accident", "vehicle collision"]
out_path = "c:\\Mukul K\\vinfo1\\video-search-engine\\scratch\\search_out.txt"

with open(out_path, "w", encoding="utf-8") as f:
    for q in queries:
        f.write(f"--- QUERY: {q} ---\n")
        try:
            results = SearchService.search_events(query=q, limit=3)
            if not results:
                f.write("No results found.\n")
            for i, r in enumerate(results):
                desc = r.get("event", {}).get("description", "No description")
                f.write(f"Result {i+1} (Score: {r['score']}): {desc}\n")
        except Exception as e:
            f.write(f"Error searching {q}: {e}\n")
        f.write("\n")
