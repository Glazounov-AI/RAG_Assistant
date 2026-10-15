import importlib.util
import sys
from pathlib import Path

import chromadb


def test_gigachat_initial_sync(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    giga_dir = root / "assistant_giga"

    # Загружаем GigaChat-модули из нужной директории.
    spec = importlib.util.spec_from_file_location(
        "giga_config_test", giga_dir / "config.py"
    )
    giga_config = importlib.util.module_from_spec(spec)

    # config.py использует локальный paths.py.
    sys.path.insert(0, str(giga_dir))
    try:
        spec.loader.exec_module(giga_config)
    finally:
        sys.path.pop(0)

    # Проверяем, что конфигурация содержит обе KB.
    assert "support" in giga_config.KNOWLEDGE_BASES
    assert "portfolio" in giga_config.KNOWLEDGE_BASES
    assert (
        giga_config.KNOWLEDGE_BASES["support"]["collection"]
        == "gigachat_rag_collection"
    )
import os
import subprocess
import sys
from pathlib import Path


def test_gigachat_vector_sync(tmp_path):
    root = Path(__file__).resolve().parents[1]
    giga_dir = root / "assistant_giga"

    script = r"""
from pathlib import Path
import hashlib
import sys
import chromadb

sys.path.insert(0, sys.argv[1])

from vector_store import VectorStore

base = Path(sys.argv[2])
data_dir = base / "portfolio"
data_dir.mkdir()

document = data_dir / "project.md"
content = "# GigaChat Portfolio\n\nТестовый проект."
document.write_text(content, encoding="utf-8")

collection = chromadb.PersistentClient(
    path=str(base / "chromadb")
).get_or_create_collection("gigachat_portfolio_kb")

store = object.__new__(VectorStore)
store.collection_name = "gigachat_portfolio_kb"
store.collection = collection
store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

assert store.sync_data_directory(data_dir) == 1

first_version = store.get_index_version()
assert first_version

metadata = collection.get(include=["metadatas"])["metadatas"]
expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
assert all(m["content_hash"] == expected_hash for m in metadata)

assert store.sync_data_directory(data_dir) == 0
assert store.get_index_version() == first_version

document.write_text(content + "\nОбновление.", encoding="utf-8")
assert store.sync_data_directory(data_dir) == 1
assert store.get_index_version() != first_version
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(giga_dir), str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )

    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_gigachat_cache_isolation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    giga_dir = root / "assistant_giga"

    script = r"""
import sys

sys.path.insert(0, sys.argv[1])

from cache import RAGCache

cache = RAGCache(db_path=sys.argv[2])
query = "Расскажи о проекте"

cache.set(
    query, "Ответ поддержки", ["support context"],
    mode="support", kb_version="revision-1",
)

cache.set(
    query, "Ответ портфолио", ["portfolio context"],
    mode="portfolio", kb_version="revision-1",
)

support = cache.get(
    query, mode="support", kb_version="revision-1"
)
portfolio = cache.get(
    query, mode="portfolio", kb_version="revision-1"
)

assert support is not None
assert portfolio is not None
assert support["answer"] == "Ответ поддержки"
assert portfolio["answer"] == "Ответ портфолио"

assert cache.get(
    query, mode="portfolio", kb_version="revision-2"
) is None

assert cache.get(
    query, mode="portfolio", kb_version="revision-1"
)["answer"] == "Ответ портфолио"
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(giga_dir),
            str(tmp_path / "cache.db"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_gigachat_pipeline_routing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    giga_dir = root / "assistant_giga"

    script = r"""
import sys
sys.path.insert(0, sys.argv[1])

from rag_pipeline import RAGPipeline
from cache import RAGCache


class FakeStore:
    def __init__(self, mode):
        self.mode = mode
        self.search_count = 0

    def get_index_version(self):
        return "revision-1"

    def search(self, query, top_k=5):
        self.search_count += 1
        return [{
            "text": f"Контекст {self.mode}",
            "metadata": {"source": f"{self.mode}.md"},
        }]


pipeline = object.__new__(RAGPipeline)
pipeline.default_mode = "support"
pipeline.knowledge_bases = {
    "support": {},
    "portfolio": {},
}
pipeline.vector_stores = {
    "support": FakeStore("support"),
    "portfolio": FakeStore("portfolio"),
}
pipeline.vector_store = pipeline.vector_stores["support"]
pipeline.cache = RAGCache(db_path=sys.argv[2])
pipeline.model = "GigaChat"

generation_count = [0]

pipeline._create_prompt = lambda query, docs: docs[0]["text"]

def generate(prompt):
    generation_count[0] += 1
    return f"Ответ: {prompt}"

pipeline._generate_answer = generate

query = "Расскажи о проекте"

support = pipeline.query(query)
portfolio = pipeline.query(query, mode="portfolio")

assert support["answer"] == "Ответ: Контекст support"
assert portfolio["answer"] == "Ответ: Контекст portfolio"
assert not support["from_cache"]
assert not portfolio["from_cache"]
assert generation_count[0] == 2

cached = pipeline.query(query, mode="portfolio")

assert cached["from_cache"]
assert cached["answer"] == portfolio["answer"]
assert generation_count[0] == 2
assert pipeline.vector_stores["portfolio"].search_count == 1

try:
    pipeline.query(query, mode="unknown")
except ValueError:
    pass
else:
    raise AssertionError("Неизвестный режим должен вызывать ValueError")
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(giga_dir),
            str(tmp_path / "pipeline_cache.db"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
