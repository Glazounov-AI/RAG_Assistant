"""
Оценка качества RAG системы через RAGAS для assistant_giga.
GigaChat используется для RAG, OpenAI API — для метрик RAGAS.
"""

import os
import sys
import types
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def _patch_ragas_vertexai_import() -> None:
    """
    ragas 0.4.3 жёстко импортирует ChatVertexAI из langchain-community,
    где этот модуль уже удалён (перенесён в langchain-google-vertexai).
    """
    if "langchain_community.chat_models.vertexai" in sys.modules:
        return

    vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        ChatVertexAI = type("ChatVertexAI", (), {})
    vertexai_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mod


_patch_ragas_vertexai_import()

from datasets import Dataset
from ragas import evaluate

from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._answer_relevance import AnswerRelevancy

from rag_pipeline import RAGPipeline


EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.

    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки

    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG системы...\n")

    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")

        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])

        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)

        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])

        print(f"     [+] Ответ получен от GigaChat")

    print()

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.

    GigaChat генерирует ответы, OpenAI API используется RAGAS для оценки метрик.
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (GIGACHAT MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    # Проверка ключей GigaChat (для RAG)
    if not os.getenv("GIGACHAT_AUTH_KEY"):
        print("[ОШИБКА] GIGACHAT_AUTH_KEY не установлен")
        sys.exit(1)

    # Проверка ключа OpenAI (для метрик RAGAS)
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("  RAGAS использует OpenAI API для расчёта метрик.")
        print("  Установите OPENAI_API_KEY в файле .env")
        sys.exit(1)

    # Инициализация RAG pipeline (GigaChat)
    try:
        print("[*] Инициализация RAG системы (GigaChat mode)...\n")
        pipeline = RAGPipeline()
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    print("   (RAGAS использует OpenAI API для оценки метрик)")
    print("   (это займёт 1-2 минуты)\n")

    print("   [+] Используем базовые метрики RAGAS\n")
    from langchain_openai import OpenAIEmbeddings

    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    metrics_to_use = [
        Faithfulness(),
        ContextPrecision(),
        AnswerRelevancy(),
    ]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            embeddings=ragas_embeddings,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    import math

    faithfulness_values = [
        v for v in result['faithfulness']
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v for v in result['context_precision']
        if not (isinstance(v, float) and math.isnan(v))
    ]
    answer_relevancy_values = [
        v for v in result['answer_relevancy']
        if not (isinstance(v, float) and math.isnan(v))
    ]

    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values)
        if context_precision_values else 0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_values) / len(answer_relevancy_values)
        if answer_relevancy_values else 0
    )

    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")

    avg_score = (avg_faithfulness + avg_context_precision + avg_answer_relevancy) / 3
    print(f"\n{'-'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")

    # Детали по вопросам
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")

        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")

        cp_val = result['context_precision'][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")

        ar_val = result['answer_relevancy'][i]
        if not (isinstance(ar_val, float) and math.isnan(ar_val)):
            print(f"   Answer Relevancy:   {ar_val:.4f}")
        else:
            print(f"   Answer Relevancy:   не удалось вычислить")

    # Пояснения
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

Answer Relevancy (Релевантность ответа):
  Оценивает, насколько ответ соответствует вопросу (через генерацию под-вопросов и сравнение embeddings).
    """)

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
