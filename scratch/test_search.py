import asyncio
from app.services.search_service import SearchService
import json

queries = [
    "red car",
    "person running",
    "accident",
    "vehicle collision"
]

def run_searches():
    for q in queries:
        print(f"--- QUERY: {q} ---")
        try:
            results = SearchService.search_events(query=q, limit=3)
            for i, r in enumerate(results):
                print(f"Result {i+1} (Score: {r['score']}): {r['event']['description']}")
        except Exception as e:
            print(f"Error searching {q}: {e}")
        print("\n")

if __name__ == "__main__":
    run_searches()
