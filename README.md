# RAG Assistant

Два варианта RAG-ассистента с общей архитектурой: поиск по документам через ChromaDB, кэширование ответов и генерация на основе найденного контекста.

| Проект | LLM | Эмбеддинги | Назначение |
|---|---|---|---|
| `assistant_api` | OpenAI (`gpt-4o-mini`) | OpenAI (`text-embedding-3-small`) | RAG через OpenAI API + оценка качества RAGAS |
| `assistant_giga` | GigaChat | GigaChat Embeddings | RAG через API Сбера |

## Требования

- Python 3.14+
- Общее виртуальное окружение и общий файл зависимостей `requirements.txt`
- Общий файл `.env` в корне проекта (секреты и ключи API)

## Установка

```powershell
# Установка Python (если ещё не установлен)
winget install Python.Python.3.14

# Создание виртуального окружения
py -3.14 -m venv venv_py314

# Активация
.\venv_py314\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## Настройка

### Переменные окружения (`.env`)

Скопируйте шаблон и заполните ключи:

```powershell
copy .env_example .env
```

- `OPENAI_API_KEY` — нужен для `assistant_api` и `evaluate_ragas.py`
- `GIGACHAT_AUTH_KEY`, `GIGACHAT_RQUID` — нужны для `assistant_giga`

### Параметры RAG (`config.py`)

В каждом проекте свой файл настроек:

- `assistant_api/config.py`
- `assistant_giga/config.py`

Основные параметры:

| Параметр | Описание |
|---|---|
| `MODEL` | модель для генерации ответов |
| `EMBEDDING_MODEL` | модель для эмбеддингов |
| `TOP_K` | число документов при поиске |
| `CHUNK_SIZE` | размер чанка (символы) |
| `CHUNK_OVERLAP` | перекрытие между чанками |
| `TEMPERATURE`, `MAX_TOKENS` | параметры генерации |

### Документы для индексации

Положите текстовые файлы (`.txt`, `.md`, `.markdown`) в папку `data/` внутри каждого проекта. При запуске сканируется папка и в ChromaDB добавляются только **новые** файлы.

## Запуск

Из корня проекта, с активированным виртуальным окружением:

```powershell
# OpenAI-версия
cd assistant_api
python app.py

# GigaChat-версия
cd assistant_giga
python app.py

# Оценка качества RAG (только assistant_api)
cd assistant_api
python evaluate_ragas.py
```

### Команды в консольном приложении

- `stats` — статистика (ChromaDB, кэш, модель)
- `clear` — очистка кэша
- `exit` / `quit` — выход

## Структура проекта

```
RAG_Assistant/
├── .env                  # секреты (не коммитить)
├── .env_example          # шаблон переменных окружения
├── requirements.txt      # общие зависимости
├── assistant_api/
│   ├── app.py            # консольное приложение
│   ├── rag_pipeline.py   # RAG-пайплайн
│   ├── vector_store.py   # ChromaDB + эмбеддинги
│   ├── cache.py          # SQLite-кэш ответов
│   ├── config.py         # настройки
│   ├── paths.py          # пути к db/cache и db/chromadb
│   ├── evaluate_ragas.py # оценка через RAGAS
│   ├── data/docs.txt     # исходные документы
│   └── db/
│       ├── cache/        # файлы кэша (создаётся автоматически)
│       └── chromadb/     # данные ChromaDB
└── assistant_giga/
    ├── app.py
    ├── rag_pipeline.py
    ├── vector_store.py
    ├── gigachat_client.py
    ├── cache.py
    ├── config.py
    ├── paths.py
    ├── data/docs.txt
    └── db/
        ├── cache/
        └── chromadb/
```

## Как работает RAG

1. Запрос пользователя проверяется в кэше
2. Если ответа нет — создаётся эмбеддинг запроса и выполняется поиск в ChromaDB (`TOP_K` документов)
3. Найденный контекст передаётся в LLM
4. Ответ сохраняется в кэш

Кэш и ChromaDB хранятся отдельно для каждого проекта в папке `db/`.
