"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List
import os
import hashlib
import json
import time
from contextvars import ContextVar
from vector_store import embedding_tokens

_generation_usage = ContextVar("generation_usage", default=None)
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache
import config
from paths import DB_DIR
from object_registry import ObjectRegistry, validate_catalog
from query_router import route, execute


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self, 
                 collection_name: str = config.COLLECTION_NAME,
                 cache_db_path: str | None = None,
                 data_dir: str | None = None,
                 model: str = config.MODEL):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_dir: папка с документами для индексации
            model: модель OpenAI для генерации ответов
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=15.0, max_retries=0)
        data_dir = str(data_dir or config.DATA_DIR)

        # Настройки режимов и коллекций
        self.default_mode = config.DEFAULT_MODE
        self.knowledge_bases = config.KNOWLEDGE_BASES
        
        if collection_name != self.knowledge_bases[self.default_mode]["collection"]:
            raise ValueError(
                f"Коллекция {collection_name!r} не соответствует "
                f"режиму {self.default_mode!r}"
            )

        # Инициализация векторных хранилищ
        print("Инициализация векторных хранилищ...")

        self.vector_store = VectorStore(collection_name=collection_name)

        self.vector_stores = {
            self.default_mode: self.vector_store,
        }

        for mode, settings in self.knowledge_bases.items():
            if mode == self.default_mode:
                continue

            self.vector_stores[mode] = VectorStore(
                collection_name=settings["collection"]
            )

        for mode, settings in self.knowledge_bases.items():
            kb_dir = settings["data_dir"]
            print(f"Синхронизация KB {mode}: {kb_dir}")
            updated = self.vector_stores[mode].sync_data_directory(kb_dir)
            print(f"KB {mode}: обновлено документов — {updated}")

        self.object_registry = None
        for mode, settings in self.knowledge_bases.items():
            validate_catalog(settings)
            if settings.get("catalog"):
                if self.object_registry is None:
                    self.object_registry = ObjectRegistry(DB_DIR / "objects" / "registry.db")
                self.object_registry.sync(mode, settings["data_dir"], settings["catalog"])

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path or config.CACHE_DB_PATH)
        
        print("RAG Pipeline инициализирован (API mode)")
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Создание промпта для LLM с контекстом.
        
        Args:
            query: вопрос пользователя
            context_docs: релевантные документы из векторного хранилища
            
        Returns:
            сформированный промпт
        """
        # Формирование контекста из документов
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Документ {i} ({doc.get('source') or 'unknown'}):\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Создание промпта
        prompt = f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на языке вопроса пользователя

Ответ:"""
        
        return prompt
    
    def _generate_answer(
        self,
        prompt: str,
        mode: str = config.DEFAULT_MODE,
    ) -> str:
        """
        Генерация ответа через OpenAI API.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """

        settings = self.knowledge_bases[mode]
        system_prompt = settings.get("system_prompt", config.SYSTEM_PROMPT)

        response = self.openai_client.chat.completions.create(
            model=settings.get("model", self.model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.get("temperature", config.TEMPERATURE),
            max_tokens=settings.get("max_tokens", config.MAX_TOKENS)
        )
        
        usage = getattr(response, "usage", None)
        _generation_usage.set({"input_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None)})
        return response.choices[0].message.content.strip()
    
    def query(
        self,
        user_query: str,
        use_cache: bool = True,
        mode: str | None = None,
        sources: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Основной метод для обработки запроса пользователя через API.
        
        Поток:
        1. Проверка кеша
        2. Если в кеше нет - поиск в векторном хранилище
        3. Формирование промпта с контекстом
        4. Генерация ответа через LLM API
        5. Сохранение в кеш
        
        Args:
            user_query: запрос пользователя
            use_cache: использовать ли кеш
            
        Returns:
            словарь с ответом и метаданными
        """
        print(f"\n{'='*60}")
        print("Обработка запроса")
        print(f"{'='*60}")
        
        # Определяем режим и соответствующую базу знаний
        mode = mode or self.default_mode

        if mode not in self.knowledge_bases:
            raise ValueError(f"Неизвестный режим работы: {mode}")

        vector_store = self.vector_stores[mode]

        settings = self.knowledge_bases[mode]
        if sources is None and settings.get("catalog"):
            strategy, args = route(user_query, settings["catalog"])
            if strategy != "semantic":
                return execute(self.object_registry, mode, user_query, settings["catalog"], strategy, args)
        model = settings.get("model", self.model)
        use_cache = sources is None and use_cache and settings.get("cache_enabled", True)
        signature = json.dumps(["f2-v1", model, settings.get("system_prompt", config.SYSTEM_PROMPT), settings.get("top_k", config.TOP_K), settings.get("temperature", config.TEMPERATURE), settings.get("max_tokens", config.MAX_TOKENS)], ensure_ascii=False)
        kb_version = vector_store.get_index_version() + ":" + hashlib.sha256(signature.encode()).hexdigest()
        started = time.perf_counter()
        embedding_tokens.set(None)
        _generation_usage.set(None)

        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(
                user_query,
                mode=mode,
                kb_version=kb_version,
            )
            
            if cached_result:
                print("[+] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at"),
                    "model": model,
                    "retrieved_sources": self._sources(cached_result.get("context") or []),
                    "usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
                }
            else:
                print("[-] Ответ не найден в кеше")
        
        # Шаг 2: Поиск релевантных документов
        print("[*] Поиск релевантных документов через API...")
        context_docs = vector_store.search(user_query, top_k=settings.get("top_k", config.TOP_K), **({"sources": sources} if sources is not None else {}))
        if sources is not None and not context_docs:
            return {"query": user_query, "answer": "В указанных источниках информация не найдена.", "from_cache": False, "model": "local-no-context", "retrieved_sources": [], "strategy": "targeted", "usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": embedding_tokens.get() or 0}}
        print(f"[+] Найдено {len(context_docs)} релевантных документов")
        
        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)
        
        # Шаг 4: Генерация ответа через API
        print(f"[*] Генерация ответа через OpenAI API ({model})...")
        answer = self._generate_answer(prompt, mode=mode)
        print("[+] Ответ получен от API")
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = context_docs
            self.cache.set(
                user_query,
                answer,
                context_for_cache,
                mode=mode,
                kb_version=kb_version,
            )
            print("[+] Сохранено в кеш")
        
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": model,
            "mode": "API",
            "strategy": "targeted" if sources is not None else "semantic",
            "retrieved_sources": self._sources(context_docs),
            "usage": {**(_generation_usage.get() or {}), "embedding_tokens": embedding_tokens.get()},
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    @staticmethod
    def _sources(docs):
        return [{"source": doc["source"], "distance": doc.get("distance")} for doc in docs if isinstance(doc, dict) and doc.get("source")]

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики системы.
        
        Returns:
            словарь со статистикой
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API"
        }


if __name__ == "__main__":
    # Тестирование RAG pipeline в API режиме
    import sys
    
    try:
        pipeline = RAGPipeline()
        
        # Тестовые запросы
        test_queries = [
            "Что такое машинное обучение?",
            "Что такое RAG?",
            "Как работают трансформеры?"
        ]
        
        for query in test_queries:
            result = pipeline.query(query)
            print(f"\n{'='*60}")
            print(f"Вопрос: {result['query']}")
            print(f"Из кеша: {result['from_cache']}")
            print(f"Ответ: {result['answer']}")
            print(f"{'='*60}\n")
        
        # Повторный запрос (должен быть из кеша)
        print("\n--- Повторный запрос ---")
        result = pipeline.query(test_queries[0])
        print(f"Из кеша: {result['from_cache']}")
        
        # Статистика
        stats = pipeline.get_stats()
        print(f"\nСтатистика системы:")
        print(f"Векторное хранилище: {stats['vector_store']}")
        print(f"Кеш: {stats['cache']}")
        print(f"Режим: {stats['mode']}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

