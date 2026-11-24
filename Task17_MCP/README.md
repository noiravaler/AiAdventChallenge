# Task17_MCP — Orbit MCP: сервер натальных карт и гороскопов

Собственный **MCP-сервер астрологии**: не безликие гороскопы «по знакам», а
реальная астрономия (Swiss Ephemeris) поверх **натальной карты пользователя**,
дополненная живым контекстом (погода, фаза Луны). Сервер агрегирует несколько
REST-API в 9 аккуратных инструментов и отдает агенту факты — трактовки делает
модель.

## Архитектура

```
Агент (Task17 / Claude Desktop / любой MCP-клиент)
   │  MCP: stdio ИЛИ Streamable HTTP (/mcp)
   ▼
orbit-mcp (Go 1.26, SDK modelcontextprotocol/go-sdk)
   ├─ Astrologer-API (self-host, движок Kerykeion/Swiss Ephemeris)
   │     натальные карты, транзиты, синастрия, соляры, фаза Луны, SVG
   ├─ Open-Meteo Geocoding — «Казань» → координаты + IANA-таймзона (без ключа)
   └─ Open-Meteo Weather   — текущая погода и прогноз дня (без ключа)
```

Ключевой прием экономии контекста: у Astrologer-API есть **context-эндпоинты**,
возвращающие компактный XML с фактами (позиции, дома, аспекты), спроектированный
для промптов. Сервер отдает его агенту вместе с урезанным JSON (топ-аспекты по
орбису, распределение стихий/качеств) вместо сотен килобайт сырых эфемерид.

Натальные карты — **состояние сервера**: строятся один раз, сохраняются в
`orbit-charts.json` и переиспользуются всеми инструментами по имени.

## Инструменты (9)

| Tool | Что делает |
| ---- | ---------- |
| `natal_build_chart` | дата+время+город → геокодинг → таймзона → натальная карта (позиции, аспекты, стихии) + XML-контекст; сохраняется по имени |
| `transit_today` | транзиты «сейчас» к сохранённой карте (по умолчанию — последней) |
| `transit_forecast` | календарь 1–31 день: аспекты с орбисом ≤ 3° по каждому дню (self-host → цикл запросов бесплатен) |
| `synastry_compare` | совместимость двух карт: аспекты, дома партнёров, скор Discepolo с расшифровкой |
| `solar_return` | соляр на год — «прогноз от дня рождения» |
| `moon_phase` | фаза Луны, освещённость, восход/заход, ближайшие затмения |
| `vibe_today` | брифинг дня: погода + Луна + тесные транзиты + практические подсказки |
| `chart_render` | SVG-колесо карты (natal/transit/synastry, темы, язык `RU`) |
| `orbit_status` | диагностика: провайдеры + список сохранённых карт |

## Запуск астро-движка (один раз)

Движок — open-source [Astrologer-API](https://github.com/g-battaglia/Astrologer-API)
(AGPLv3). Нужен Python 3.12+ и [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/g-battaglia/Astrologer-API   # ветка master = v6
cd Astrologer-API
uv run libephemeris download leb2-extended   # запас эфемерид, несколько ГБ
uv run poe start                             # поднимет http://127.0.0.1:8000
```

Проверка: `curl http://127.0.0.1:8000/health` → `{"status":"OK"}`.
Hosted-вариант через RapidAPI тоже поддерживается: задайте
`ORBIT_ASTRO_URL=https://astrologer.p.rapidapi.com` и `ORBIT_ASTRO_KEY`.

## Запуск сервера

```sh
cd Task17_MCP
go run ./app                        # MCP over stdio
go run ./app -http 127.0.0.1:5199   # MCP over Streamable HTTP + статус-страница на /
go run ./app -list                  # список инструментов
go run ./app -probe vibe_today -args '{"city":"Казань"}'   # вызов без MCP-клиента
go test ./...                       # офлайн-тесты (фейковые серверы)
```

### Подключение к чату Task17

Чат Task17 подключает Orbit **по умолчанию** (в дефолтном `AI_MCP_URLS`) и сам
поднимает его при старте через `mcpboot`, если сервер ещё не запущен — вся
цепочка стартует одной командой:

```sh
cd ../Task17
go run ./app        # чат → mcpboot поднимет orbit → orbit поднимет движок (если ORBIT_ASTRO_DIR задан)
```

Подключить Orbit вручную (например, в другом каталоге чата) можно так:

```sh
AI_MCP_URLS="orbit=http://127.0.0.1:5199/mcp" go run ./app
```

### Подключение Claude Desktop (stdio)

```json
{
  "mcpServers": {
    "orbit": { "command": "go", "args": ["run", "./app"], "cwd": "/path/to/Task17_MCP" }
  }
}
```

## Конфигурация (`.env` / окружение)

| Переменная | По умолчанию | Смысл |
| ---------- | ------------ | ----- |
| `ORBIT_ASTRO_URL` | `http://127.0.0.1:8000` | астро-движок |
| `ORBIT_ASTRO_VERSION` | `v6` | версия API движка |
| `ORBIT_ASTRO_KEY` | — | `X-RapidAPI-Key` (только для hosted-варианта) |
| `ORBIT_ASTRO_TIMEOUT` | `30s` | лимит запроса к движку |
| `ORBIT_GEO_URL` | `https://geocoding-api.open-meteo.com` | геокодинг |
| `ORBIT_OPENMETEO_URL` | `https://api.open-meteo.com` | погода |
| `ORBIT_DEFAULT_CITY` | `Москва` | город, когда не задан ни city, ни карта |
| `ORBIT_LANG` / `ORBIT_THEME` / `ORBIT_HOUSES` | `RU` / `classic` / `P` | рендер SVG и система домов |
| `ORBIT_STORE_FILE` | `orbit-charts.json` | файл натальных карт (в `.gitignore`) |
| `ORBIT_CHARTS_DIR` | `<каталог store>/charts` | каталог отрисованных SVG-колёс (в `.gitignore`) |
| `ORBIT_ASTRO_DIR` | — | каталог клона Astrologer-API; задаётся → Orbit сам поднимает движок |
| `ORBIT_ENGINE_AUTOSTART` | `auto` | `auto` (каталог + localhost) / `on` / `off` |
| `ORBIT_ENGINE_START_TIMEOUT` | `2m` | сколько ждать `/health` после старта движка |
| `ORBIT_ENGINE_LOG` | `bin/astrologer.log` | stdout/stderr движка (в `.gitignore`) |

## Автозапуск движка

Если задан `ORBIT_ASTRO_DIR` (и `ORBIT_ASTRO_URL` — localhost), Orbit при старте
проверяет `/health` движка и, если тот не отвечает, сам запускает
`uv run poe start` в указанном каталоге (uv ищется в PATH и в `~/.local/bin`).
Логи движка — в `ORBIT_ENGINE_LOG`; поднявшись в `-probe`-режиме, движок
остаётся работать в фоне. **Остановка:** при завершении Orbit (SIGTERM/SIGINT
или штатный shutdown HTTP) движок получает SIGTERM по всей группе процессов
(uv → poe → uvicorn), при зависании — SIGKILL; «уже работавший» движок не
трогается. Для hosted-варианта (RapidAPI) автозапуск не нужен:

```sh
ORBIT_ASTRO_DIR=~/Astrologer-API make run-http   # движок + MCP одной командой
```

## Примеры диалогов

- «Построй мою карту: Анна, 21.07.1995, 14:30, Казань» → `natal_build_chart`
- «Что день грядущий мне готовит?» → `vibe_today` (погода + Луна + транзиты)
- «Когда лучше запустить проект в ближайшие две недели?» → `transit_forecast`
- «Мы с Бобом совместимы?» → `natal_build_chart` (Боб) → `synastry_compare`
- «Нарисуй нашу синастрию» → `chart_render` (type=synastry)

## Отрисовка карт (chart_render)

`chart_render` возвращает результат из двух частей:

- **Модель видит только компактную сводку** (`type`, `chart`, `second`, `bytes`,
  `note`) — ни SVG, ни путей, ни ссылок, чтобы они не попадали в сообщения чата;
- **SVG едет отдельным image-контентом MCP** (`image/svg+xml`): клиент чата
  сохраняет картинку в кэш и показывает её **прямо в сообщении** (чата Task17
  раздаёт каталог картинок по `GET /files/<имя>` и дописывает ссылку в конец
  ответа — фронтенд рендерит колесо инлайновой картинкой).

Дополнительно (для просмотра вручную):

- файл-галерея в `ORBIT_CHARTS_DIR` (`anna-natal.svg`, `anna-synastry-bob.svg` —
  имя детерминировано, повторный рендер перезаписывает файл);
- в HTTP-режиме — `http://127.0.0.1:5199/charts/anna-natal.svg` (только `.svg`,
  прочие файлы наружу не отдаются);
- галерея всех рендеров — на статус-странице `http://127.0.0.1:5199/`.

## Ограничения

- Первый запрос к движку после старта может считаться медленно (инициализация эфемерид) — таймаут `30s` рассчитан с запасом.
- `ORBIT_ASTRO_KEY` не задан → запросы без авторизации, что верно только для self-host.
- Данные рождения остаются на вашей машине — это осознанное преимущество self-host.
