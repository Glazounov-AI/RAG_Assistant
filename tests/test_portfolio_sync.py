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

    assert store.sync_portfolio_directory(data_dir) == 1

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
    assert store.sync_portfolio_directory(data_dir) == 0

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
    assert store.sync_portfolio_directory(data_dir) == 1

    first = collection.get(include=["documents", "metadatas"])
    first_ids = set(first["ids"])

    # Изменяем исходный документ
    document.write_text("Вторая версия проекта.", encoding="utf-8")

    # Повторная индексация
    assert store.sync_portfolio_directory(data_dir) == 1

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
    assert store.sync_portfolio_directory(data_dir) == 1

    before = collection.get(include=["documents", "metadatas"])

    # Изменяем файл.
    document.write_text("Новая версия проекта.", encoding="utf-8")

    # Имитируем недоступность OpenAI.
    def failing_embedding(text):
        raise RuntimeError("OpenAI embeddings unavailable")

    store._create_embedding = failing_embedding

    with pytest.raises(RuntimeError, match="OpenAI embeddings unavailable"):
        store.sync_portfolio_directory(data_dir)

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
    assert store.sync_portfolio_directory(data_dir) == 1
    version_1 = store.get_portfolio_index_version()

    # Без изменений версия индекса должна сохраняться.
    assert store.sync_portfolio_directory(data_dir) == 0
    assert store.get_portfolio_index_version() == version_1

    # Изменяем документ и переиндексируем.
    document.write_text("Вторая версия проекта.", encoding="utf-8")

    assert store.sync_portfolio_directory(data_dir) == 1
    version_2 = store.get_portfolio_index_version()

    assert version_2 != version_1

    # Повторная синхронизация не должна менять версию.
    assert store.sync_portfolio_directory(data_dir) == 0
    assert store.get_portfolio_index_version() == version_2

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
    assert store.sync_portfolio_directory(data_dir) == 1

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
        store.sync_portfolio_directory(data_dir)

    after = collection.get(include=["documents", "metadatas"])

    # В базе должна остаться только исходная ревизия.
    assert set(after["ids"]) == set(before["ids"])
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]