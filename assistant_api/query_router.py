"""Conservative deterministic routing; ambiguous requests stay semantic."""
import re


def route(query, catalog):
    q = query.strip().casefold().rstrip("?.!")
    noun = re.escape(catalog["noun"].casefold())
    plural = re.escape(catalog.get("plural", catalog["noun"]).casefold())
    enabled = catalog["strategies"]
    aliases = catalog.get("query_aliases", {})
    for strategy in ("count", "list"):
        if strategy in enabled and q in [x.casefold() for x in aliases.get(strategy, [])]:
            return strategy, {"page": 1}
    if "filter" in enabled:
        for field, prefixes in aliases.get("filter", {}).items():
            for prefix in prefixes:
                match = re.fullmatch(re.escape(prefix.casefold()) + r" (.+?)(?: страница ([1-9][0-9]{0,5}))?", q)
                if match and field in catalog.get("filter_fields", []):
                    value = match.group(1)
                    # Only known single values are resolved by natural-language aliases.
                    if value in [v.casefold() for v in aliases.get("values", {}).get(field, [])]:
                        return "filter", {"field": field, "value": value, "page": int(match.group(2) or 1)}
    for name, pattern in (("count", rf"(?:сколько|количество) {noun}"), ("list", rf"(?:(?:список|перечень) {noun}|(?:все|покажи все|перечисли все) {plural})(?: страница ([1-9][0-9]{{0,5}}))?")):
        match = re.fullmatch(pattern, q)
        if name in enabled and match:
            return name, {"page": int(match.group(1) or 1) if name == "list" else 1}
    match = re.fullmatch(rf"{plural} по ([a-z_]+): (.+?)(?: страница ([1-9][0-9]{{0,5}}))?", q)
    if "filter" in enabled and match and match.group(1) in catalog.get("filter_fields", []):
        return "filter", {"field": match.group(1), "value": match.group(2), "page": int(match.group(3) or 1)}
    return "semantic", {}


def list_objects(rows, args):
    return rows


def filter_objects(rows, args):
    def matches(obj):
        v = obj[args["field"]]
        return args["value"] in [x.casefold() for x in (v if isinstance(v, list) else [v])]
    return [(o,s) for o,s in rows if matches(o)]


def count_objects(rows, args):
    return rows


HANDLERS = {"list": list_objects, "count": count_objects, "filter": filter_objects}


def execute(registry, mode, query, catalog, strategy, args):
    rows = HANDLERS[strategy](registry.objects(mode), args)
    total = len(rows)
    size = catalog.get("page_size", 20)
    page = args.get("page", 1)
    visible = [] if strategy == "count" else rows[(page-1)*size:page*size]
    answer = f"Объектов в реестре по запросу: {total}."
    if strategy != "count":
        answer += f" Страница {page}; показано {len(visible)} из {total}."
        if not visible:
            answer += " Нет совпадений в реестре." if total == 0 else " Нет объектов на этой странице."
        for obj, _ in visible:
            answer += f"\n- {obj['title']} — {obj['summary']}"
    return {"query": query, "answer": answer, "from_cache": False, "model": "structured-registry", "strategy": strategy, "total": total, "page": page, "retrieved_sources": [{"source": s} for _,s in visible], "usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}}
