import hashlib
import sys
import pytest
from pathlib import Path

import chromadb

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "assistant_api"),
)

from vector_store import VectorStore


def test_portfolio_initial_sync_and_unchanged(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    content = "# Тестовый проект\n\nОписание проекта."
    document.write_text(content, encoding="utf-8")

    # Временная ChromaDB — без обращения к рабочей базе.
    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    # Создаём объект без __init__, чтобы не подключаться
    # к реальной ChromaDB и OpenAI.
    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    assert store.sync_data_directory(data_dir) == 1

    first = collection.get(include=["metadatas"])
    assert len(first["ids"]) > 0

    expected_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    assert all(
        metadata["content_hash"] == expected_hash
        for metadata in first["metadatas"]
    )

    # Повторный запуск не должен переиндексировать документ.
    assert store.sync_data_directory(data_dir) == 0

    second = collection.get(include=["metadatas"])
    assert second["ids"] == first["ids"]

def test_portfolio_document_update(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    document.write_text("Первая версия проекта.", encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    # Первая индексация
    assert store.sync_data_directory(data_dir) == 1

    first = collection.get(include=["documents", "metadatas"])
    first_ids = set(first["ids"])

    # Изменяем исходный документ
    document.write_text("Вторая версия проекта.", encoding="utf-8")

    # Повторная индексация
    assert store.sync_data_directory(data_dir) == 1

    second = collection.get(include=["documents", "metadatas"])
    second_ids = set(second["ids"])

    # Старые чанки удалены, новые созданы
    assert first_ids.isdisjoint(second_ids)

    # В базе осталось только актуальное содержимое
    assert second["documents"] == ["Вторая версия проекта."]

    # Хеш соответствует новой версии
    expected_hash = hashlib.sha256(
        "Вторая версия проекта.".encode("utf-8")
    ).hexdigest()

    assert all(
        metadata["content_hash"] == expected_hash
        for metadata in second["metadatas"]
    )

def test_portfolio_embedding_failure_preserves_old_version(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    document.write_text("Рабочая версия проекта.", encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    # Создаём рабочую версию индекса.
    assert store.sync_data_directory(data_dir) == 1

    before = collection.get(include=["documents", "metadatas"])

    # Изменяем файл.
    document.write_text("Новая версия проекта.", encoding="utf-8")

    # Имитируем недоступность OpenAI.
    def failing_embedding(text):
        raise RuntimeError("OpenAI embeddings unavailable")

    store._create_embedding = failing_embedding

    with pytest.raises(RuntimeError, match="OpenAI embeddings unavailable"):
        store.sync_data_directory(data_dir)

    # После ошибки содержимое индекса не должно измениться.
    after = collection.get(include=["documents", "metadatas"])

    assert after["ids"] == before["ids"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]

def test_portfolio_index_version_changes(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    document.write_text("Первая версия проекта.", encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    # Индексируем первую версию документа.
    assert store.sync_data_directory(data_dir) == 1
    version_1 = store.get_index_version()

    # Без изменений версия индекса должна сохраняться.
    assert store.sync_data_directory(data_dir) == 0
    assert store.get_index_version() == version_1

    # Изменяем документ и переиндексируем.
    document.write_text("Вторая версия проекта.", encoding="utf-8")

    assert store.sync_data_directory(data_dir) == 1
    version_2 = store.get_index_version()

    assert version_2 != version_1

    # Повторная синхронизация не должна менять версию.
    assert store.sync_data_directory(data_dir) == 0
    assert store.get_index_version() == version_2

def test_portfolio_partial_write_failure_preserves_old_version(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    document.write_text("Рабочая версия проекта.", encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    # Создаём исходную рабочую ревизию.
    assert store.sync_data_directory(data_dir) == 1

    before = collection.get(include=["documents", "metadatas"])

    # Подготавливаем обновление.
    document.write_text("Обновлённая версия проекта.", encoding="utf-8")

    original_add = collection.add

    def failing_add(*args, **kwargs):
        # Имитируем запись новых чанков перед ошибкой.
        original_add(*args, **kwargs)
        raise RuntimeError("Simulated ChromaDB write failure")

    monkeypatch.setattr(collection, "add", failing_add)

    with pytest.raises(
        RuntimeError,
        match="Simulated ChromaDB write failure",
    ):
        store.sync_data_directory(data_dir)

    after = collection.get(include=["documents", "metadatas"])

    # В базе должна остаться только исходная ревизия.
    assert set(after["ids"]) == set(before["ids"])
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]

def test_portfolio_incomplete_revision_recovery(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    document.write_text("Первая версия проекта.", encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection

    embedding_calls = []

    def fake_embedding(text):
        embedding_calls.append(text)
        return [float(len(text)), 1.0, 0.5]

    store._create_embedding = fake_embedding

    # Индексируем исходный документ.
    assert store.sync_data_directory(data_dir) == 1

    # Изменяем исходный файл.
    new_content = "Вторая версия проекта."
    document.write_text(new_content, encoding="utf-8")

    # Имитируем незавершённую запись новой ревизии:
    # сохранён один чанк, хотя ожидалось два.
    collection.add(
        ids=["project.md_incomplete_0"],
        documents=[new_content],
        embeddings=[[float(len(new_content)), 1.0, 0.5]],
        metadatas=[{
            "source": "project.md",
            "content_hash": hashlib.sha256(
                new_content.encode("utf-8")
            ).hexdigest(),
            "revision": "incomplete",
            "chunk_count": 2,
        }],
    )

    calls_before = len(embedding_calls)

    # Неполная ревизия не должна приниматься за готовую.
    assert store.sync_data_directory(data_dir) == 1

    # Должна произойти повторная генерация embeddings.
    assert len(embedding_calls) > calls_before

    after = collection.get(include=["documents", "metadatas"])

    assert after["documents"] == [new_content]
    assert len(after["ids"]) == 1
    assert after["metadatas"][0]["chunk_count"] == 1

def test_portfolio_complete_revision_recovery(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    old_content = "Первая версия проекта."
    new_content = "Вторая версия проекта."

    document.write_text(old_content, encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    # Индексируем старую версию.
    assert store.sync_data_directory(data_dir) == 1

    old_ids = set(collection.get()["ids"])

    # Изменяем исходный документ.
    document.write_text(new_content, encoding="utf-8")

    # Имитируем полную запись новой ревизии,
    # после которой приложение аварийно завершилось.
    collection.add(
        ids=["project.md_recovered_0"],
        documents=[new_content],
        embeddings=[[float(len(new_content)), 1.0, 0.5]],
        metadatas=[{
            "source": "project.md",
            "content_hash": hashlib.sha256(
                new_content.encode("utf-8")
            ).hexdigest(),
            "config_hash": store._index_config_hash(),
            "revision": "recovered",
            "chunk_count": 1,
        }],
    )

    # При восстановлении embeddings создаваться не должны.
    def unexpected_embedding(text):
        raise AssertionError("Повторная генерация embeddings")

    store._create_embedding = unexpected_embedding

    assert store.sync_data_directory(data_dir) == 0

    after = collection.get(include=["documents", "metadatas"])

    assert after["ids"] == ["project.md_recovered_0"]
    assert after["documents"] == [new_content]
    assert old_ids.isdisjoint(set(after["ids"]))

def test_portfolio_invalid_chunk_ids_recovery(tmp_path):
    data_dir = tmp_path / "portfolio"
    data_dir.mkdir()

    document = data_dir / "project.md"
    old_content = "Первая версия проекта."
    new_content = "Вторая версия проекта."

    document.write_text(old_content, encoding="utf-8")

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("portfolio_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "portfolio_kb"
    store.collection = collection

    embedding_calls = []

    def fake_embedding(text):
        embedding_calls.append(text)
        return [float(len(text)), 1.0, 0.5]

    store._create_embedding = fake_embedding

    assert store.sync_data_directory(data_dir) == 1

    document.write_text(new_content, encoding="utf-8")

    # Количество чанков правильное, но индекс неправильный.
    collection.add(
        ids=["project.md_invalid_99"],
        documents=[new_content],
        embeddings=[[float(len(new_content)), 1.0, 0.5]],
        metadatas=[{
            "source": "project.md",
            "content_hash": hashlib.sha256(
                new_content.encode("utf-8")
            ).hexdigest(),
            "revision": "invalid",
            "chunk_count": 1,
        }],
    )

    calls_before = len(embedding_calls)

    # Неверная ревизия должна быть отвергнута.
    assert store.sync_data_directory(data_dir) == 1
    assert len(embedding_calls) > calls_before

    after = collection.get(include=["documents", "metadatas"])

    assert len(after["ids"]) == 1
    assert after["documents"] == [new_content]
    assert after["ids"][0].endswith("_0")
    assert after["ids"][0] != "project.md_invalid_99"


def test_support_sync_and_document_deletion(tmp_path, monkeypatch):
    from vector_store import VectorStore

    data_dir = tmp_path / "support"
    data_dir.mkdir()
    document = data_dir / "test.md"
    document.write_text("Тестовый документ базы поддержки. " * 20, encoding="utf-8")

    store = VectorStore(
        collection_name="api_rag_collection",
        persist_directory=tmp_path / "chroma",
    )
    monkeypatch.setattr(store, "_create_embedding", lambda text: [0.1, 0.2, 0.3])

    assert store.sync_data_directory(data_dir) == 1
    assert store.sync_data_directory(data_dir) == 0

    second = data_dir / "remaining.md"
    second.write_text("Документ, который должен остаться. " * 20, encoding="utf-8")
    assert store.sync_data_directory(data_dir) == 1

    document.unlink()
    assert store.sync_data_directory(data_dir) == 0

    sources = store._get_indexed_sources()
    assert sources == {"remaining.md"}


def test_config_change_triggers_reindex(tmp_path, monkeypatch):
    import config

    data_dir = tmp_path / "support"
    data_dir.mkdir()
    (data_dir / "test.md").write_text(
        "Проверка изменения конфигурации. " * 20,
        encoding="utf-8",
    )

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("support_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "support_kb"
    store.collection = collection

    calls = []
    def fake_embedding(text):
        calls.append(text)
        return [float(len(text)), 1.0, 0.5]

    store._create_embedding = fake_embedding

    assert store.sync_data_directory(data_dir) == 1
    initial_ids = set(collection.get()["ids"])
    initial_calls = len(calls)

    assert store.sync_data_directory(data_dir) == 0
    assert len(calls) == initial_calls

    monkeypatch.setattr(config, "CHUNK_SIZE", config.CHUNK_SIZE + 100)

    assert store.sync_data_directory(data_dir) == 1
    assert len(calls) > initial_calls
    assert set(collection.get()["ids"]).isdisjoint(initial_ids)

    calls_after_reindex = len(calls)
    assert store.sync_data_directory(data_dir) == 0
    assert len(calls) == calls_after_reindex


def test_index_version_changes_with_config(tmp_path, monkeypatch):
    import config

    data_dir = tmp_path / "support"
    data_dir.mkdir()
    (data_dir / "test.md").write_text(
        "Проверка версии индекса. " * 20,
        encoding="utf-8",
    )

    collection = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb")
    ).get_or_create_collection("support_kb")

    store = object.__new__(VectorStore)
    store.collection_name = "support_kb"
    store.collection = collection
    store._create_embedding = lambda text: [float(len(text)), 1.0, 0.5]

    assert store.sync_data_directory(data_dir) == 1
    original_version = store.get_index_version()

    monkeypatch.setattr(
        config, "CHUNK_OVERLAP", config.CHUNK_OVERLAP + 10
    )

    # Изменение настроек само по себе не меняет индекс.
    assert store.get_index_version() == original_version

    # Версия меняется после успешной переиндексации.
    assert store.sync_data_directory(data_dir) == 1
    new_version = store.get_index_version()
    assert new_version != original_version

    # Повторная синхронизация ничего не меняет.
    assert store.sync_data_directory(data_dir) == 0
    assert store.get_index_version() == new_version
