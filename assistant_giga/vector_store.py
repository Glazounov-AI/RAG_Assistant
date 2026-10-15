import uuid
import hashlib
"""
Модуль работы с векторным хранилищем ChromaDB для GigaChat.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from pathlib import Path
from gigachat_client import GigaChatClient
from paths import ensure_db_dirs
import config


class VectorStore:
    """Векторное хранилище на основе ChromaDB с GigaChat embeddings."""
    
    def __init__(self, collection_name: str = config.COLLECTION_NAME, persist_directory: str | Path | None = None):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        ensure_db_dirs()
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory or config.CHROMADB_PATH)
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # GigaChat клиент для создания embeddings
        self.gigachat_client = GigaChatClient()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap (config.CHUNK_OVERLAP)
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            
        Returns:
            список чанков
        """
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Сохраняем исходные чанки до фильтрации.
        original_chunks = chunks

        # Пост-обработка: фильтруем слишком короткие чанки.
        chunks = [
            chunk for chunk in original_chunks
            if len(chunk) >= config.MIN_CHUNK_SIZE
        ]

        # Не теряем содержимое небольшого документа целиком.
        if not chunks and original_chunks:
            chunks = original_chunks

        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int | None = None) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap (по умолчанию config.CHUNK_OVERLAP)
            
        Returns:
            текст overlap
        """
        overlap_size = overlap_size or config.CHUNK_OVERLAP
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            
        Returns:
            список чанков
        """
        overlap = config.CHUNK_OVERLAP
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_indexed_sources(self) -> set[str]:
        """Имена файлов, уже присутствующие в коллекции."""
        if self.collection.count() == 0:
            return set()

        result = self.collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        sources = {meta["source"] for meta in metadatas if meta and "source" in meta}

        # Совместимость с индексом без metadata (один файл docs.txt)
        if not sources and self.collection.count() > 0:
            sources.add("docs.txt")

        return sources

    def _list_data_files(self, data_dir: Path) -> List[Path]:
        """Список файлов в папке data с поддерживаемыми расширениями."""
        return sorted(
            path for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.DATA_EXTENSIONS
        )

    @staticmethod
    def _index_config_hash() -> str:
        """SHA-256 параметров построения векторного индекса."""
        settings = (
            config.EMBEDDING_MODEL,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
            config.MIN_CHUNK_SIZE,
        )
        return hashlib.sha256(repr(settings).encode("utf-8")).hexdigest()

    def get_index_version(self) -> str:
        """Возвращает версию по фактически сохранённым чанкам портфолио."""

        indexed = self.collection.get(include=["metadatas"])

        entries = sorted(
            (
                chunk_id,
                (metadata or {}).get("content_hash", ""),
                (metadata or {}).get("config_hash", ""),
            )
            for chunk_id, metadata in zip(
                indexed["ids"],
                indexed["metadatas"] or [],
            )
        )

        signature = "\n".join(
            f"{chunk_id}:{content_hash}:{config_hash}"
            for chunk_id, content_hash, config_hash in entries
        )

        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def sync_data_directory(self, data_dir: str | Path) -> int:
        """Индексирует новые и изменённые документы портфолио."""

        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Каталог не найден: {data_dir}")

        updated = 0
        config_hash = self._index_config_hash()

        for file_path in self._list_data_files(data_dir):
            source = file_path.name
            text = file_path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            existing = self.collection.get(
                where={"source": source},
                include=["metadatas"],
            )
            old_ids = existing["ids"]
            old_metadatas = existing.get("metadatas") or []

            # Группируем проиндексированные чанки по ревизиям.
            revisions = {}

            for chunk_id, metadata in zip(old_ids, old_metadatas):
                if not metadata:
                    continue

                revision = metadata.get("revision")
                indexed_hash = metadata.get("content_hash")
                indexed_config = metadata.get("config_hash")

                if revision and indexed_hash and indexed_config == config_hash:
                    revisions.setdefault(
                        (revision, indexed_hash), []
                    ).append(chunk_id)

            # Ищем ревизию, соответствующую текущему содержимому файла.
            matching_revisions = []

            for (revision, indexed_hash), ids in revisions.items():
                if indexed_hash != content_hash:
                    continue

                revision_metadatas = [
                    metadata
                    for metadata in old_metadatas
                    if metadata
                    and metadata.get("revision") == revision
                    and metadata.get("content_hash") == indexed_hash
                ]

                expected_counts = {
                    metadata.get("chunk_count")
                    for metadata in revision_metadatas
                }

                # Ревизия считается полной, только если все её чанки
                # имеют одинаковое ожидаемое количество.
                if len(expected_counts) != 1:
                    continue

                expected_count = next(iter(expected_counts))

                if type(expected_count) is not int or expected_count <= 0:
                    continue

                expected_ids = {
                    f"{source}_{revision}_{i}"
                    for i in range(expected_count)
                }

                if set(ids) == expected_ids:
                    matching_revisions.append((revision, ids))

            if matching_revisions:
                # При наличии нескольких совпадений сохраняем одно.
                matching_revisions.sort(key=lambda item: item[0])
                keep_revision, keep_ids = matching_revisions[-1]

                redundant_ids = [
                    chunk_id
                    for chunk_id in old_ids
                    if chunk_id not in set(keep_ids)
                ]

                if redundant_ids:
                    self.collection.delete(ids=redundant_ids)
                    print(f"Удалены лишние ревизии документа: {source}")

                continue

            chunks = self._chunk_text(text)
            if not chunks:
                raise ValueError(f"Документ пуст после разбиения: {source}")

            revision = uuid.uuid4().hex
            new_ids = [f"{source}_{revision}_{i}" for i in range(len(chunks))]
            embeddings = [self._create_embedding(chunk) for chunk in chunks]
            metadatas = [
                {
                    "source": source,
                    "content_hash": content_hash,
                    "config_hash": config_hash,
                    "revision": revision,
                    "chunk_count": len(chunks),
                }
                for _ in chunks
            ]

            # Сначала записываем новую ревизию
            try:
                self.collection.add(
                    ids=new_ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

            except Exception:
                # При частичной записи пытаемся удалить новые чанки.
                # Старую ревизию не затрагиваем.
                try:
                    self.collection.delete(ids=new_ids)
                except Exception as cleanup_error:
                    print(
                        f"ВНИМАНИЕ: не удалось очистить "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )

                raise

            # Только после успешной записи удаляем старую
            if old_ids:
                self.collection.delete(ids=old_ids)

            updated += 1
            print(f"Обновлён документ портфолио: {source}")

        # Удаляем документы, которых больше нет в исходном каталоге.
        current_sources = {
            path.name for path in self._list_data_files(data_dir)
        }
        indexed_sources = self._get_indexed_sources()

        if not current_sources and indexed_sources:
            raise RuntimeError(
                f"Каталог {data_dir} пуст. Удаление всей KB заблокировано."
            )

        for source in indexed_sources - current_sources:
            self.collection.delete(where={"source": source})
            print(f"Удалён из индекса: {source}")

        return updated

    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через GigaChat.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        embeddings = self.gigachat_client.get_embeddings([text], model=config.EMBEDDING_MODEL)
        return embeddings[0]
    
    def search(self, query: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
        
        # Форматирование результатов
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                meta = None
                if results.get('metadatas') and results['metadatas'][0]:
                    meta = results['metadatas'][0][i]
                documents.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'source': meta.get('source') if meta else None,
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    vector_store.sync_data_directory(config.DATA_DIR)
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

