"""
Основной RAG pipeline для API режима.
Управляет потоком: запрос -> кеш -> vector search -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List
import os
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache
import config


class RAGPipeline:
    """Основной pipeline для RAG системы в API режиме."""
    
    def __init__(self, 
                 collection_name: str = config.COLLECTION_NAME,
                 cache_db_path: str | None = None,
                 data_file: str | None = None,
                 model: str = config.MODEL):
        """
        Инициализация RAG pipeline.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша
            data_file: путь к файлу с документами
            model: модель OpenAI для генерации ответов
        """
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        data_file = str(data_file or config.DATA_FILE)
        
        # Инициализация компонентов
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(collection_name=collection_name)
        
        # Загрузка документов, если коллекция пустая
        if self.vector_store.collection.count() == 0:
            print(f"Загрузка документов из {data_file}...")
            self.vector_store.load_documents(data_file)
        
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
            context_parts.append(f"Документ {i}:\n{doc['text']}\n")
        
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
- Отвечай на русском языке

Ответ:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Генерация ответа через OpenAI API.
        
        Args:
            prompt: промпт для модели
            
        Returns:
            сгенерированный ответ
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
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
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")
        
        # Шаг 1: Проверка кеша
        if use_cache:
            print("[*] Проверка кеша...")
            cached_result = self.cache.get(user_query)
            
            if cached_result:
                print("[+] Ответ найден в кеше")
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context"),
                    "cached_at": cached_result.get("created_at")
                }
            else:
                print("[-] Ответ не найден в кеше")
        
        # Шаг 2: Поиск релевантных документов
        print("[*] Поиск релевантных документов через API...")
        context_docs = self.vector_store.search(user_query, top_k=config.TOP_K)
        print(f"[+] Найдено {len(context_docs)} релевантных документов")
        
        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта...")
        prompt = self._create_prompt(user_query, context_docs)
        
        # Шаг 4: Генерация ответа через API
        print(f"[*] Генерация ответа через OpenAI API ({self.model})...")
        answer = self._generate_answer(prompt)
        print("[+] Ответ получен от API")
        
        # Шаг 5: Сохранение в кеш
        if use_cache:
            print("[*] Сохранение в кеш...")
            context_for_cache = [doc['text'] for doc in context_docs]
            self.cache.set(user_query, answer, context_for_cache)
            print("[+] Сохранено в кеш")
        
        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API"
        }
    
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

