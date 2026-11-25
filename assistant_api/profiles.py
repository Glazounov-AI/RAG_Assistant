"""Load trusted, local JSON profile manifests; never accept paths from HTTP clients."""
import json
import re
from pathlib import Path


def load_profiles(directory, existing):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Profile directory does not exist")
    result = {}
    collections = {p["collection"] for p in existing.values()}
    for manifest in sorted(root.glob("*.json")):
        p = json.loads(manifest.read_text(encoding="utf-8"))
        mode = p.pop("mode")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", mode) or mode in existing or mode in result:
            raise ValueError("Invalid or duplicate profile mode")
        collection = p["collection"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{3,63}", collection) or collection in collections:
            raise ValueError("Invalid or duplicate collection")
        collections.add(collection)
        for key, low, high in (("top_k",1,20),("max_tokens",1,4096),("max_query_length",1,16000),("rate_limit_per_minute",1,120)):
            if type(p.get(key)) is not int or not low <= p[key] <= high:
                raise ValueError(f"Invalid {key}")
        if type(p.get("temperature")) not in (int,float) or not 0 <= p["temperature"] <= 2:
            raise ValueError("Invalid temperature")
        if type(p.get("cache_enabled")) is not bool or type(p.get("log_content")) is not bool:
            raise ValueError("Explicit cache and logging policy required")
        if not isinstance(p.get("model"),str) or not p["model"].strip():
            raise ValueError("Model required")
        p["data_dir"] = (manifest.parent / p["data_dir"]).resolve()
        if not p["data_dir"].is_dir() or not any(x.is_file() and x.suffix.lower() in {".md",".txt",".markdown"} for x in p["data_dir"].iterdir()):
            raise ValueError("Empty or missing knowledge base")
        p["system_prompt"] = (manifest.parent / p.pop("prompt_file")).read_text(encoding="utf-8").strip()
        if not p["system_prompt"]:
            raise ValueError("Empty system prompt")
        result[mode] = p
    return result
