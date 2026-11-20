# Трёхслойная память для ai-chat-go (дизайн-документ, v2)

Решение для задачи: разделить информацию на **краткосрочную**, **рабочую** и
**долговременную** память; хранить слои отдельно; явно маршрутизировать запись;
определить правило попадания каждого слоя в промпт.

База: Task11 (актуальный код). Go 1.22.

> **Изменения v1 → v2:**
> 1. **Факты сжатия — основа долговременной памяти**: sticky-facts, извлекаемые
>    при компрессии, являются главным каналом наполнения долговременного хранилища
>    (с дедупликацией и подтверждением), а не просто кэшем ветки.
> 2. **Персистентность: SQLite вместо JSON-файлов.** Три слоя — три группы таблиц
>    в одной БД; добавляется зависимость `modernc.org/sqlite` (pure Go, без cgo).

---

## 1. Анализ известных агентов

| Агент | Краткосрочная | Рабочая | Долговременная | Ключевой приём |
|---|---|---|---|---|
| **Claude Code** | свежее контекстное окно сессии | компакция (сводка) при заполнении окна | CLAUDE.md + авто-память `MEMORY.md` (индекс ≤200 строк всегда) + topic-файлы по требованию | индекс + файлы: в промпт идёт только «горячий» минимум |
| **LangGraph** | checkpointer потока (state+messages) | trim_messages / суммаризация | Store с namespace, поиск по эмбеддингам, топ-K в system prompt | ретрив по релевантности, не «всё подряд» |
| **Letta (MemGPT)** | recall memory (полная история) | core memory — редактируемые блоки в контексте | archival memory (векторная БД) | агент сам редактирует core-блоки инструментами |
| **ChatGPT** | текущий чат | — | «Saved memories» + фоновые выводы из истории | явный + фоновый каналы записи |
| **Mem0** | сообщения хода | — | pipeline: извлечение → сравнение (ADD/UPDATE/DELETE/NOOP) → БД | дедупликация при записи |
| **Cline** | текущий чат | activeContext.md (перезаписывается) | projectbrief / progress / decisionLog | рабочий слой = состояние задачи |
| **OpenCode** | диалог сессии | compress: диапазон → блок-сводка | AGENTS.md | явная операция сжатия |

Выводы: (1) деление «окно / сводка / постоянное хранилище» универсально;
(2) в промпт никогда не попадает всё — либо маленький всегда-блок, либо ретрив по
релевантности; (3) запись в долговременную память — явная и с дедупликацией.

## 2. Что уже есть в Task11

| Слой | Что есть | Оценка |
|---|---|---|
| Краткосрочная | дерево `Server.hist`, активная ветка, окно `CoveredUpTo` | ✅ есть |
| Рабочая | `branchState.Summary` + `branchState.Facts` | ⚠️ прототип |
| Долговременная | — | ❌ отсутствует |

Точки интеграции: `buildModelMessagesLocked()` (server.go:2104) — сборка промпта;
`branchState` (history.go:45) — состояние ветки; `autoCompress` (server.go:2185) —
хук продвижения между слоями; `tools.Definitions/Execute` — инструменты памяти;
`modelHistoryPath` — заменяется на путь к БД. Токенизатора нет → бюджеты считаем
оценкой `len(text)/4` (stdlib).

---

## 3. Хранилище: SQLite (v2)

### 3.1. Драйвер и файл

- **Драйвер:** `modernc.org/sqlite` — чистый Go, без cgo (кросс-компиляция и
  сборка как раньше; альтернатива `mattn/go-sqlite3` требует cgo). Подключается
  через `database/sql`.
- **Файл БД:** один на все модели — `history/chat.db`
  (конфиг `AI_DB_FILE`, по умолчанию `history/chat.db`; пустое значение —
  персистентность выключена, как раньше с `AI_HISTORY_FILE`).
- Прагмы при открытии: `journal_mode=WAL`, `busy_timeout=5000`, `foreign_keys=ON`.
- Разделение слоёв обеспечивается **на уровне схемы**: у каждого слоя свои таблицы;
  строки помечены `model` (пер-модельные контексты, как раньше пер-модельные файлы).
- Живое представление остаётся в памяти (`Server.hist`, `branchState`), БД —
  долговечный слой: `persist()` пишет транзакционно, `restore()` читает при старте.

### 3.2. Схема

```sql
-- ── Слой 1: краткосрочная (текущий диалог) ─────────────────────
CREATE TABLE IF NOT EXISTS messages (
    model        TEXT    NOT NULL,
    id           TEXT    NOT NULL,           -- "m1","m2",...
    parent       TEXT,                        -- "" у корня
    role         TEXT    NOT NULL,           -- user|assistant|tool|system|summary|marker
    content      TEXT,
    tool_calls   TEXT,                        -- JSON []api.ToolCall
    tool_call_id TEXT,
    at           TIMESTAMP NOT NULL,
    PRIMARY KEY (model, id)
);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(model, parent);

-- ── Слой 2: рабочая (состояние текущей задачи, per branch) ─────
CREATE TABLE IF NOT EXISTS branches (
    model          TEXT NOT NULL,
    leaf           TEXT NOT NULL,            -- узел-лист = id ветки
    title          TEXT,
    fork_up_to     TEXT,
    frozen         TEXT,                     -- JSON []foldStage
    covered_up_to  TEXT,
    summary        TEXT,                     -- SUMMARY блока сжатия
    task           TEXT,                     -- TASK блока (цель задачи)
    facts          TEXT,                     -- FACTS блока («ключ: значение», горячий кэш)
    facts_up_to    TEXT,
    notes          TEXT,                     -- заметки/артефакты (task_note)
    plaque         TEXT,
    history_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model, leaf)
);

-- Служебные указатели сессии
CREATE TABLE IF NOT EXISTS session_state (
    model               TEXT PRIMARY KEY,
    active_leaf         TEXT,
    switchable          TEXT,                -- JSON []string
    next_id             INTEGER NOT NULL,
    total_prompt_tokens INTEGER NOT NULL DEFAULT 0
);

-- ── Слой 3: долговременная (основа — подтверждённые факты) ─────
CREATE TABLE IF NOT EXISTS memories (
    model         TEXT    NOT NULL,
    id            TEXT    NOT NULL,          -- "k1","k2",...
    kind          TEXT    NOT NULL CHECK (kind IN ('profile','decision','knowledge')),
    key           TEXT    NOT NULL,          -- нормализованный ключ («имя», «стек»…)
    text          TEXT    NOT NULL,          -- «ключ: значение»
    source        TEXT    NOT NULL,          -- compress | user | tool
    created       TIMESTAMP NOT NULL,
    updated       TIMESTAMP,
    confirmations INTEGER NOT NULL DEFAULT 1, -- сколько раз факт подтверждён сжатием
    hits          INTEGER NOT NULL DEFAULT 0, -- сколько раз попадал в промпт
    last_hit      TIMESTAMP,
    PRIMARY KEY (model, id),
    UNIQUE (model, kind, key)                -- дедупликация по ключу
);
```

### 3.3. Пакет `storage`

Новый пакет `storage/` (рядом с `server/`, тот же модуль):

```go
func Open(path string) (*DB, error)            // прагмы + миграции схемы
func (d *DB) Close() error

// Слой 1
func (d *DB) LoadSession(model string) (*Snapshot, error) // messages+branches+state
func (d *DB) AppendMessage(model string, m MessageRow) error
func (d *DB) SaveBranch(model string, b BranchRow) error  // upsert ветки целиком
func (d *DB) SaveState(model string, st StateRow) error

// Слой 3
func (d *DB) UpsertMemory(model string, it MemoryItem) error
//   INSERT ... ON CONFLICT(model,kind,key) DO UPDATE
//   SET text=excluded.text, updated=now, confirmations=confirmations+1
func (d *DB) ListMemories(model string) ([]MemoryItem, error)
func (d *DB) DeleteMemory(model, id string) error
func (d *DB) TouchMemories(model string, ids []string) error // hits+1, last_hit=now

// Миграция
func (d *DB) ImportLegacyJSON(model, path string) error // .chat-history-*.json → таблицы
```

`Snapshot` возвращает ровно то, чем сейчас заполняется `Server` в `restore()`
(server.go:488) — серверный код меняется минимально: JSON read/write заменяется
вызовами `storage`.

**Миграция при старте:** если для модели БД пуста, а
`history/.chat-history-<model>.json` существует → `ImportLegacyJSON`, после
успеха файл переименовывается в `*.json.migrated`.

---

## 4. Архитектура слоёв и правила

```
┌────────────────────────────────────────────────────────────────┐
│ В промпт (buildModelMessagesLocked):                           │
│ 1. system prompt (cfg.System)                                  │
│ 2. ▓ ДОЛГОВРЕМЕННАЯ (таблица memories)                         │
│      profile — всегда; подтверждённые факты/решения/знания —   │
│      по совпадению ключей с последним сообщением пользователя  │
│ 3. ▓ РАБОЧАЯ (таблица branches)                                │
│      Task + Summary + Facts + Notes — одной system-строкой     │
│ 4. frozen stages (унаследованные ступени сжатия ветки)         │
│ 5. ▓ КРАТКОСРОЧНАЯ (таблица messages): окно активной ветки     │
└────────────────────────────────────────────────────────────────┘
```

### 4.1. Краткосрочная (текущий диалог) — таблица `messages`

- **Запись:** всё по умолчанию (user / assistant / tool-обмен) — единственный
  слой с автозаписью всего; append-only.
- **Правило промпта:** активный путь корень→`activeLeaf`, окно `CoveredUpTo`
  (последние `compressKeep` содержательных ходов дословно). Старше окна —
  выпадает и подлежит продвижению в слой 2.
- **Вытеснение:** `autoCompress` при `ходов > compressThreshold` (не удаляет
  строки — сдвигает указатель окна; полная история остаётся в таблице).

### 4.2. Рабочая (данные текущей задачи) — таблица `branches`

- **Запись (явная маршрутизация):** только через сжатие. Промпт `compressPrompt`
  возвращает три блока: `SUMMARY:` (перезапись `summary`), `TASK:` (перезапись
  `task`, как activeContext у Cline), `FACTS:` (строки «ключ: значение»).
  `notes` — через инструмент `task_note` или вручную из UI.
  Никогда: дословные сообщения, результаты инструментов, код.
- **Правило промпта:** всегда одна system-строка: `[Задача]…[Сводка]…[Факты]…[Заметки]…`,
  бюджет ~800 токенов (оценка символов/4).
- **Жизненный цикл:** живёт с веткой; наследование через существующие
  `inheritBranchStateLocked`/`Frozen`; `/api/reset` чистит `messages`, `branches`,
  `session_state` и **не трогает** `memories`.

### 4.3. Долговременная (профиль, решения, знания) — таблица `memories`

**Основа слоя — факты сжатия.** Pipeline наполнения:

1. `autoCompress` получает от модели блок `FACTS:` со строками вида
   `[профиль] имя: Аня`, `[решение] стек: Go+SQLite`, `[знание] курс: начат`.
2. Каждая строка парсится → `storage.UpsertMemory` (kind из тега, ключ — часть до
   «:», нормализованный: lowercase, без пунктуации). `ON CONFLICT` по
   `(model, kind, key)`: текст обновляется, `confirmations++`.
3. **Стабилизация:** факт с `confirmations ≥ 2` считается подтверждённым —
   «выживший» через минимум одно повторное сжатие. Только подтверждённые факты
   участвуют в авто-инъекции в промпт; кандидаты с `confirmations = 1` доступны
   только через инструмент `recall`. Это защита от шума: случайная remарка не
   станет «долговременной», устойчиво повторяющийся факт — станет.
4. Два остальных канала (явная просьба пользователя, инструмент `remember`)
   записываются с `confirmations = 2` — явное доверие не требует подтверждения.
5. **Дедупликация:** UNIQUE(model, kind, key) + Mem0-семантика: тот же ключ →
   UPDATE значения; модель-оценщик при спорных случаях решает DELETE/NOOP
   (промпт канала LONGTERM сохранён из v1 для запросов «забудь X»).
6. **Запрет на запись:** приветствия, код, результаты инструментов, служебные
   команды. Лимит ≤200 записей на модель; вытеснение по (hits, last_hit).

**Правило промпта:**
- `kind = profile` — **всегда**, каждый запрос, ≤10 строк (как MEMORY.md);
- `decision`/`knowledge` с `confirmations ≥ 2` — по совпадению нормализованных
  ключей/слов с последним сообщением пользователя (stdlib: lowercase,
  стоп-слова, пересечение множеств), топ-N по (hits, updated);
- попадание в промпт → `TouchMemories` (hits+1, last_hit);
- бюджет слоя ~600 токенов; переполнение — режем knowledge раньше профиля;
- формат: `[Память о пользователе]…[Решения]…[Знания]…`.

### 4.4. Сводная таблица

| Слой | Таблица(ы) | Запись | В промпт |
|---|---|---|---|
| Краткосрочная | `messages` | всё, автоматически | окно последних ходов дословно |
| Рабочая | `branches` (+`session_state`) | только сжатие и task_note | одна system-строка, всегда, ≤800 ток. |
| Долговременная | `memories` | факты сжатия (confirmations≥2) + user + tool, дедуп по ключу | profile всегда; остальное по совпадению, ≤600 ток. |

## 5. Пайплайн одного хода

```
пользователь → AppendMessage(user)                [слой 1]
  → buildModelMessagesLocked:
      system + memories(profile + match) + branches(Task/Summary/Facts/Notes)
      + frozen + окно messages
  → tools.RunLoop (remember/recall/forget/task_note пишут в слой 3 / 2 сразу)
  → AppendMessage(tool-обмен, assistant) → persist() (транзакция)
  → autoCompress при превышении порога:
      SUMMARY/TASK/FACTS → SaveBranch (перезапись слоя 2)
      FACTS-строки → UpsertMemory (дедуп, confirmations++) [слой 3]
  → persist()
```

## 6. Инструменты (tools.go)

```go
remember(text, kind "profile"|"decision"|"knowledge") // upsert, confirmations=2
recall(query, kind)          // поиск по memories (включая confirmations=1)
forget(id)                   // DELETE
task_note(text)              // append в branches.notes
```

## 7. API и UI

- `GET /api/memory` — записи слоя 3 (kind, key, text, confirmations, hits, даты).
- `POST /api/memory` — добавить вручную `{text, kind}` (канал user).
- `DELETE /api/memory?id=…`.
- `GET /api/debug` — показать фактически подставленные блоки слоёв 2 и 3.
- UI: панель «Память»: долговременные записи + рабочая карточка ветки
  (Task/Summary/Facts/Notes), добавить/удалить.

## 8. Тесты

1. Профиль попадает в каждый запрос; знание без совпадения ключей — нет,
   с совпадением — да; неподтверждённый факт (confirmations=1) в промпт не идёт.
2. Сжатие: FACTS-строки апсертятся; повторное сжатие с тем же ключом не создаёт
   дубликат, увеличивает confirmations.
3. `remember` через tool-раунд создаёт запись в таблице `memories`.
4. `/api/reset` чистит слой 1–2, слой 3 сохраняется.
5. SQLite: temp-файл БД, перезапуск сервера восстанавливает диалог и ветки.
6. Миграция: legacy JSON импортируется, файл переименован.
7. Бюджеты: knowledge обрезается раньше профиля.

## 9. Порядок внедрения

1. Зависимость `modernc.org/sqlite`, пакет `storage`: схема, `Open`, CRUD,
   `ImportLegacyJSON`; замена `saveState/loadState` на `storage` (слой 1+2).
2. `branchState`+Task/Notes, расширенный промпт сжатия (SUMMARY/TASK/FACTS).
3. Таблица `memories` + инъекция профиля в `buildModelMessagesLocked`.
4. Парсинг FACTS → UpsertMemory (confirmations, стабилизация).
5. Keyword-ретрив decision/knowledge + TouchMemories.
6. Инструменты remember/recall/forget/task_note.
7. API + UI + debug.
8. Тесты, обновление `.env.example` (AI_DB_FILE) и README.

---

## 10. Реализация (выполнено в Task11)

Отклонения от проектных решений выше — уточнения в ходе внедрения:

- **`storage`-пакет** (`storage/storage.go`): схема как в §3.2 (timestamp'ы —
  INTEGER unix-nano, чтобы `MAX(at)` сортировал корректно). Вместо
  построчных `AppendMessage/SaveBranch` — транзакционный
  `SaveSession(HistoryState)`: сервер держит дерево в памяти и пишет
  сессию целиком (тот же объём данных, что писался JSON'ом), а долговременная
  память пишется построчно (`UpsertMemory/ListMemories/DeleteMemory/TouchMemories`).
- **Типы** `chatRecord/branchState/foldStage/historyState` перенесены в
  `storage` (ChatRecord/BranchState/FoldStage/HistoryState + поля Task/Notes);
  в `server/history.go` оставлены type-алиасы, поэтому дерево/ветки/тесты не
  переписывались. `loadState/saveState` остались как legacy-JSON путь
  (фикстуры + миграция).
- **Рабочий блок в промпте** = `Task + Notes` (system-строка «Рабочая память
  текущей задачи»); сводка по-прежнему идёт в пути как role=summary, факты —
  своим блоком «Блок важных фактов» (сохранена обратная совместимость тестов).
- **Промпт Sticky Facts** возвращает `ЗАДАЧА:` + tagged-факты; парсеры
  `splitTaskFacts/parseFactLine`; отсутствие секций сохраняет прежние значения.
- **Инструменты**: `RunLoop(..., exec ...ExecFunc)` — вариадический
  исполнитель (сигнатура обратно совместима, CLI не менялся); сервер передаёт
  `memoryToolExec` (remember/recall/forget/task_note + фолбэк в
  `tools.Execute`); определения — `tools.MemoryDefinitions()`.
- **Конфиг**: `AI_DB_FILE` (по умолчанию `history/chat.db`), пустое значение
  отключает персистентность; `AI_HISTORY_FILE` — только источник миграции
  (файлы переименовываются в `*.migrated`).
- **UI**: блок «Память» в панели «Детали» — список записей с счётчиком
  подтверждений, удаление, добавление; обновляется при открытии панели и
  после каждого обмена.
- **Файлы**: `storage/storage.go`, `storage/storage_test.go`,
  `server/memory.go` (слой 3 + инъекция + инструменты + API),
  `server/memory_test.go`; правки в `server/server.go`, `server/history.go`,
  `tools/tools.go`, `config/config.go`, `static/index.html`,
  `.env.example`, README.
