import sys
from pathlib import Path
import json
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assistant_api"))
from object_registry import ObjectRegistry, validate_catalog
from query_router import route, execute

C = {"noun":"проектов", "plural":"проекты", "object_type":"project", "strategies":["semantic","list","count","filter"], "filter_fields":["technologies"], "page_size":2}

@pytest.fixture
def registry(tmp_path):
    d = tmp_path / "docs"; d.mkdir()
    for i in range(5):
        obj = dict(object_id=str(i), object_type="project", title=str(i), summary="demo", technologies=["Python"] if i < 3 else ["PHP"])
        (d / f"{i}.md").write_text("<!-- object\n"+json.dumps(obj)+"\n-->\n# Demo", encoding="utf-8")
    reg = ObjectRegistry(tmp_path / "registry.db")
    reg.sync("a", d, C)
    return reg,d

@pytest.mark.parametrize("query,expected", [("сколько проектов?","count"),("все проекты","list"),("список проектов страница 2","list"),("проекты по technologies: Python","filter"),("как работает проект?","semantic")])
def test_route(query, expected):
    assert route(query,C)[0] == expected

def test_disabled():
    assert route("все проекты",dict(C,strategies=["semantic"]))[0] == "semantic"

def test_invalid():
    with pytest.raises(ValueError): validate_catalog({"catalog":dict(C,strategies=["semantic","unknown"])})

def test_count(registry):
    reg,_ = registry
    result = execute(reg,"a","q",C,"count",{})
    assert result["total"] == 5 and result["usage"]["input_tokens"] == 0

def test_page(registry):
    reg,_ = registry
    result = execute(reg,"a","q",C,"list",{"page":2})
    assert result["total"] == 5 and len(result["retrieved_sources"]) == 2

def test_filter(registry):
    reg,_ = registry
    result = execute(reg,"a","q",C,"filter",{"field":"technologies","value":"python"})
    assert result["total"] == 3

def test_isolation_and_delete(registry):
    reg,d = registry
    reg.sync("b",d,C)
    (d / "0.md").unlink()
    reg.sync("a",d,C)
    assert len(reg.objects("a")) == 4 and len(reg.objects("b")) == 5

def test_failed_sync_preserves_snapshot(registry):
    reg,d = registry
    (d / "bad.md").write_text("<!-- object\n{}\n-->",encoding="utf-8")
    with pytest.raises(ValueError): reg.sync("a",d,C)
    assert len(reg.objects("a")) == 5


ALIASES = {"count": ["сколько всего проектов"], "list": ["какие проекты есть"],
           "filter": {"technologies": ["покажи проекты на"]},
           "values": {"technologies": ["Python", "COBOL"]}}

@pytest.mark.parametrize("query,expected", [
    ("Сколько всего проектов?", "count"), ("Какие проекты есть?", "list"),
    ("Покажи проекты на Python", "filter"), ("Покажи проекты на COBOL", "filter"),
    ("Покажи проекты на Python и объясни их", "semantic"),
])
def test_aliases(query, expected):
    assert route(query, dict(C, query_aliases=ALIASES))[0] == expected


def test_aliases_disabled():
    assert route("Какие проекты есть?", dict(C, query_aliases=ALIASES, strategies=["semantic"]))[0] == "semantic"


def test_invalid_aliases():
    with pytest.raises(ValueError):
        validate_catalog({"catalog": dict(C, query_aliases={"filter": {"unknown": ["prefix"]}})})


def test_portfolio_metadata():
    from object_registry import read_objects
    folder = Path(__file__).resolve().parents[1] / "assistant_api" / "data" / "portfolio"
    rows = read_objects(folder, C)
    assert len(rows) >= 5
    assert all(o["title"] != o["summary"] for o, _ in rows)
    assert {o["object_id"] for o, _ in rows if "Python" in o["technologies"]} >= {"rag-assistant", "client-request-normalizer"}
