"""Profile-scoped structured catalog; JSON header is the source of truth."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

STRATEGIES = {"semantic", "list", "count", "filter"}


def validate_catalog(settings):
    c = settings.get("catalog")
    if c is None:
        return
    if not isinstance(c, dict):
        raise ValueError("Invalid catalog")
    enabled = c.get("strategies", [])
    if not isinstance(enabled, list) or not all(isinstance(x, str) for x in enabled) or len(set(enabled)) != len(enabled) or not set(enabled) <= STRATEGIES or "semantic" not in enabled:
        raise ValueError("Invalid catalog strategies")
    for key in ("noun", "object_type"):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f"Invalid catalog {key}")
    fields = c.get("filter_fields", [])
    if not isinstance(fields, list) or not all(isinstance(x, str) and x.isidentifier() for x in fields):
        raise ValueError("Invalid filter fields")
    if type(c.get("page_size", 20)) is not int or not 1 <= c.get("page_size", 20) <= 100:
        raise ValueError("Invalid page size")

    aliases = c.get("query_aliases", {})
    def strings(value):
        return isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
    if not isinstance(aliases, dict) or set(aliases) - {"count", "list", "filter", "values"}:
        raise ValueError("Invalid query aliases")
    for key in ("count", "list"):
        if not strings(aliases.get(key, [])):
            raise ValueError("Invalid query aliases")
    for key in ("filter", "values"):
        mapping = aliases.get(key, {})
        if not isinstance(mapping, dict) or any(field not in fields or not strings(items) for field, items in mapping.items()):
            raise ValueError("Invalid filter aliases")


def read_objects(directory, catalog):
    objects = []
    ids = set()
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8-sig")
        if not content.startswith("<!-- object\n"):
            continue
        header, sep, _ = content[len("<!-- object\n"):].partition("\n-->")
        if not sep:
            raise ValueError(f"Unclosed object header: {path.name}")
        obj = json.loads(header)
        if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) and obj[k].strip() for k in ("object_id", "object_type", "title", "summary")):
            raise ValueError(f"Invalid object: {path.name}")
        if obj["object_type"] != catalog["object_type"]:
            continue
        if obj["object_id"] in ids:
            raise ValueError("Duplicate object ID")
        for field in catalog.get("filter_fields", []):
            v = obj.get(field)
            if not (isinstance(v, str) or isinstance(v, list) and all(isinstance(x, str) for x in v)):
                raise ValueError(f"Invalid filter field {field}: {path.name}")
        ids.add(obj["object_id"])
        objects.append((obj, path.name))
    if not objects:
        raise ValueError("Empty structured catalog")
    return objects


class ObjectRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS objects (mode TEXT, id TEXT, payload TEXT, source TEXT, PRIMARY KEY(mode,id))")

    def sync(self, mode, directory, catalog):
        rows = read_objects(directory, catalog)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM objects WHERE mode=?", (mode,))
            db.executemany("INSERT INTO objects VALUES (?,?,?,?)", [(mode, o["object_id"], json.dumps(o, ensure_ascii=False), s) for o,s in rows])

    def objects(self, mode):
        with closing(sqlite3.connect(self.path)) as db:
            return [(json.loads(p), s) for p,s in db.execute("SELECT payload,source FROM objects WHERE mode=? ORDER BY id", (mode,))]
