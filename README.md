# RAG Assistant

Два самостоятельных варианта RAG-ассистента для разных технологических сред: `assistant_api` (OpenAI) и `assistant_giga` (GigaChat). У редакций общая схема работы — поиск по документам Заказчика через ChromaDB, кэширование ответов и генерация на основе найденного контекста, — но у каждой свои config, pipeline, кэш, векторное хранилище и SQLite-журнал. Редакции не являются переключением provider внутри одного приложения; клиент может использовать любую из них независимо.

|     Проект       | LLM            | Эмбеддинги                        | Назначение             |
|------------------|----------------|-----------------------------------|------------------------|
| `assistant_api`  | OpenAI         | OpenAI (`text-embedding-3-small`) | RAG через OpenAI +     |
|                  | (`gpt-4o-mini`)|                                   |  оценка качества RAGAS |
|                  |                |                                   |                        |
| `assistant_giga` | GigaChat       | GigaChat Embeddings               | RAG через API Сбера    |


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
- `RAG_DB_DIR` — необязательная, только для `assistant_api`: базовый каталог хранения данных `db`. Если `RAG_DB_DIR` не задана, локально используется стандартный каталог `assistant_api/db`. При развёртывании переменная позволяет перенести базовый каталог хранения `db` на persistent storage; под этим каталогом находятся `cache`, `logs` и `chromadb`


### Параметры RAG (`config.py`)

В каждом проекте свой файл настроек:

- `assistant_api/config.py`
- `assistant_giga/config.py`

Основные параметры:

|         Параметр            |          Описание            |
|-----------------------------|------------------------------|
| `MODEL`                     | модель для генерации ответов |
| `EMBEDDING_MODEL`           | модель для эмбеддингов       |
| `TOP_K`                     | число документов при поиске  |
| `CHUNK_SIZE`                | размер чанка (символы)       |
| `CHUNK_OVERLAP`             | перекрытие между чанками     |
| `TEMPERATURE`, `MAX_TOKENS` | параметры генерации          |
| `LOGS_DB_PATH`              | путь к SQLite-базе логов (в обеих редакциях) |


### Документы для индексации

Положите текстовые файлы (`.txt`, `.md`, `.markdown`) в папку `data/` внутри каждого проекта. При запуске сканируется папка и в ChromaDB добавляются только **новые** файлы.

## Запуск

Из корня проекта, с активированным виртуальным окружением. Каждый запуск выполняется независимо, из корня репозитория:

```powershell
# Консольное приложение на OpenAI (assistant_api)
cd assistant_api
python app.py
```

```powershell
# Консольное приложение на GigaChat (assistant_giga)
cd assistant_giga
python app.py
```

```powershell
# Оценка качества RAG (только assistant_api)
cd assistant_api
python evaluate_ragas.py
```

В редакции `assistant_api` доступны две параллельные точки входа в одну и ту же редакцию: консольное приложение (`python app.py`) и HTTP API (`python -m uvicorn api:app ...`). Обе используют один и тот же `RAGPipeline`, кэш и журнал запросов.

```powershell
# HTTP API на FastAPI (только assistant_api)
cd assistant_api
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Команды в консольном приложении

- `stats` — статистика (ChromaDB, кэш, модель)
- `clear` — очистка кэша
- `exit` / `quit` — выход


## HTTP API

Редакция `assistant_api` дополнительно предоставляет HTTP-интерфейс на FastAPI. Это параллельная точка входа рядом с консольным приложением: обе работают с одним `RAGPipeline`, кэшем и журналом запросов. У редакции `assistant_giga` HTTP-интерфейса нет.

### `GET /health`

Назначение:

- проверка доступности сервиса;
- не вызывает `RAGPipeline` для обработки пользовательского вопроса;
- не записывается в журнал пользовательских запросов.

Ответ:

```json
{
  "status": "ok",
  "service": "rag-assistant-api"
}
```

### `POST /query`

Принимает JSON вида:

```json
{
  "query": "Текст вопроса"
}
```

Успешный ответ содержит поля:

- `query`
- `answer`
- `from_cache`
- `model`

Запросы через `POST /query` записываются в SQLite-журнал с `source = "api"`. Для текущего HTTP-интерфейса `user_id = NULL`, `username = NULL`. Ошибка записи в журнал не прерывает обработку HTTP-запроса.


## Структура проекта

```
RAG_Assistant/
├── .env                  # секреты (не коммитить)
├── .env_example          # шаблон переменных окружения
├── requirements.txt      # общие зависимости
├── assistant_api/
│   ├── app.py            # консольное приложение
│   ├── api.py            # FastAPI HTTP-интерфейс
│   ├── rag_pipeline.py   # RAG-пайплайн
│   ├── vector_store.py   # ChromaDB + эмбеддинги
│   ├── cache.py          # SQLite-кэш ответов
│   ├── db_logger.py      # SQLite-лог пользовательских запросов
│   ├── config.py         # настройки
│   ├── paths.py          # пути к db/cache, db/logs и db/chromadb
│   ├── evaluate_ragas.py # оценка через RAGAS
│   ├── data/docs.txt     # исходные документы
│   └── db/
│       ├── cache/        # файлы кэша (создаётся автоматически)
│       ├── logs/         # журнал запросов logs.db (создаётся автоматически)
│       └── chromadb/     # данные ChromaDB
└── assistant_giga/
    ├── app.py
    ├── rag_pipeline.py
    ├── vector_store.py
    ├── gigachat_client.py
    ├── cache.py
    ├── db_logger.py      # SQLite-лог пользовательских запросов
    ├── config.py
    ├── paths.py
    ├── data/docs.txt
    └── db/
        ├── cache/
        ├── logs/         # журнал запросов logs.db (создаётся автоматически)
        └── chromadb/
```


## Первый запуск RAG

При первом запуске приложение создает базу данных и индексирует документы из папки /data.
При каждом следующем запуске приложение автоматически проверяет папку /data на предмет наличия новых текстовых документов и добавляет их в существующую базу.
Каждая редакция при запуске консольного приложения также создаёт собственный журнал запросов `db/logs/logs.db`. 


## Как работает RAG

1. Запрос пользователя проверяется в кэше.
2. Если ответа нет — создаётся эмбеддинг запроса и выполняется поиск в ChromaDB (`TOP_K` документов).
3. Найденный контекст передаётся в LLM.
4. Ответ сохраняется в кэш и отправляется пользователю.

Кэш и ChromaDB хранятся отдельно для каждого проекта в папке `db/`.
После ответа пользователю запрос записывается в журнал этой редакции `db/logs/logs.db` (и обычный RAG, и ответ из кэша).

## Логирование запросов

Каждая редакция ведёт свой независимый SQLite-журнал. Каталог `db/logs` создаётся автоматически и исключён из Git (`.gitignore`).

- `assistant_api/db/logs/logs.db` — журнал API-редакции (OpenAI);
- `assistant_giga/db/logs/logs.db` — журнал GigaChat-редакции.

В редакции `assistant_api` пишутся две разновидности записей: по одному на каждый реальный пользовательский вопрос из консольного приложения и из HTTP `POST /query`.

В записи сохраняются: `timestamp`, `user_id`, `username`, `source`, `query`, `response`, `from_cache`, `response_time_ms`, `created_at`.

В `assistant_api` используются два типа источника запросов: `source` = `console` для запросов через консольное приложение и `source` = `api` для запросов через HTTP `POST /query`. Для текущих console и HTTP-интерфейсов `user_id` и `username` пустые (`NULL`). Служебные команды консоли (`stats`, `clear`, `exit` / `quit`) и пустой ввод в журнал не попадают. Скрипт `evaluate_ragas.py` пользовательские запросы в `logs.db` не пишет. У `assistant_giga` HTTP-интерфейса нет — там запросы пишутся только через консольное приложение.

Ошибка записи лога не останавливает работу ассистента: это действует и для консольного приложения, и для HTTP API — ошибка логирования не прерывает обработку HTTP-запроса. В обеих редакциях механизм проверен: обычный RAG-запрос (`from_cache = 0`) и повторный идентичный запрос из кэша (`from_cache = 1`) создают отдельные записи.


## Развёртывание на Railway

В настоящее время на Railway развёрнута только редакция `assistant_api`. Редакция `assistant_giga` остаётся самостоятельной редакцией проекта и в это развёртывание не входит.

Команда запуска на Railway:

```powershell
cd assistant_api && python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

Для работы `assistant_api` на Railway используется переменная окружения `OPENAI_API_KEY` (значение ключа не публикуется).

Для постоянного хранения данных подключён Railway Volume с mount path `/data`. Для приложения установлена переменная:

```powershell
RAG_DB_DIR=/data/db
```

Поэтому данные размещаются в структуре:

```
/data/db/cache
/data/db/logs
/data/db/chromadb
```

Это позволяет сохранять SQLite-кэш, SQLite-журнал запросов и ChromaDB между redeploy контейнера.

### Проверка Railway deployment

Выполненные проверки:

- `GET /health` вернул:

```json
{
  "status": "ok",
  "service": "rag-assistant-api"
}
```

- `POST /query` успешно обработал реальный RAG-запрос;
- при повторном идентичном запросе вернулся `from_cache = true`;
- после выполнения Railway Redeploy тот же запрос снова вернул `from_cache = true`.

Это подтверждает, что кэш сохранился после redeploy и persistent Railway Volume работает с `RAG_DB_DIR=/data/db`.
