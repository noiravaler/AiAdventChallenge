package orbit

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolDef — описание инструмента для -list и status-страницы.
type ToolDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type invoker func(ctx context.Context, raw json.RawMessage) (any, error)

// Toolkit связывает сервис с MCP: описание инструментов, распаковка
// аргументов, вызов логики и формирование результата.
type Toolkit struct {
	Svc      *Service
	invokers map[string]invoker
	defs     []ToolDef
}

// NewToolkit регистрирует все инструменты Orbit.
func NewToolkit(svc *Service) *Toolkit {
	t := &Toolkit{Svc: svc, invokers: map[string]invoker{}}

	t.add("natal_build_chart",
		"Строит натальную карту: дата+время+город рождения → координаты и таймзона (Open-Meteo) → расчёт у Astrologer-API (Swiss Ephemeris). Возвращает позиции, аспекты, распределение стихий и XML-контекст для трактовки. Карта сохраняется по имени и используется другими инструментами.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in NatalArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.BuildNatal(ctx, in)
		})

	t.add("transit_today",
		"Персональные транзиты на сейчас: текущие положения планет в аспектах к натальной карте (по имени или последней построенной). Верните пользователю трактовку 2-3 самых тесных аспектов.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ChartArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.TransitToday(ctx, in)
		})

	t.add("transit_forecast",
		"Календарь транзитов на окно 1–31 день: по каждому дню — аспекты транзитов к наталу с орбисом до 3°. Полезно для «когда лучше запускать/решать».",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ForecastArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.TransitForecast(ctx, in)
		})

	t.add("synastry_compare",
		"Совместимость двух сохранённых натальных карт: межкартовые аспекты, попадания планет в дома партнёра и скор совместимости (метод Ciro Discepolo) с расшифровкой.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in SynastryArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.Synastry(ctx, in)
		})

	t.add("solar_return",
		"Солярная карта на год: положения планет в момент солнечного возвращения — основа годового прогноза от дня рождения.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in SolarArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.SolarReturn(ctx, in)
		})

	t.add("moon_phase",
		"Фаза Луны на дату и место: название фазы, освещённость, лунный знак, восход/заход и ближайшие лунные/солнечные затмения.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in MoonArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.MoonPhase(ctx, in)
		})

	t.add("vibe_today",
		"Брифинг дня: погода (Open-Meteo) + фаза Луны + тесные транзиты к последней карте + практические подсказки. Соберите из этого персональный утренний брифинг.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in VibeArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.VibeToday(ctx, in)
		})

	t.add("chart_render",
		"SVG-колесо карты (natal, transit или synastry) с темой/языком/заголовком. Изображение доставляется клиенту автоматически отдельным image-контентом и показывается пользователю; также доступно в галерее статус-страницы сервера. В ответе просто подтвердите отрисовку — не вставляйте SVG, пути к файлам и ссылки.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in RenderArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.RenderChart(ctx, in)
		})

	t.add("orbit_status",
		"Диагностика сервера: доступность астро-движка, геокодинга и погоды, список сохранённых натальных карт.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			return t.Svc.Status(ctx)
		})

	t.add("daily_affirmation",
		"Аффирмация дня с affirmations.dev на русском языке со случайным волшебным смайликом. Используйте как доброе напутствие в конце ответа или дополните брифинг vibe_today.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			return t.Svc.Affirmation(ctx)
		})

	t.add("journal_capture",
		"Снимок дня для журнала наблюдений: собирает погоду, фазу Луны, тесные транзиты к последней карте и аффирмацию дня, сохраняет запись за сегодня (повторный вызов перезаписывает её). Основной инструмент для планировщика — один вызов в день.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in JournalArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.JournalCapture(ctx, in)
		})

	t.add("journal_read",
		"Читает накопленные записи дневника наблюдений за период (по умолчанию 7 дней) — сырые снимки: погода, Луна, транзиты, аффирмации.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in JournalArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.JournalRead(ctx, in)
		})

	t.add("journal_summary",
		"Агрегированный отчёт по дневнику за период: средняя/мин/макс температура, дни с осадками, траектория фаз Луны, самые частые транзиты, аффирмации и напоминания. Вызывайте по расписанию для регулярного саммари.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in JournalArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.JournalSummary(ctx, in)
		})

	t.add("pulse_capture",
		"Минутный снимок неба для пульса: фаза Луны и её знак, самый тесный транзит к последней карте, температура. Лёгкий, идемпотентен в пределах минуты — предназначен для частых вызовов планировщиком (хоть каждую минуту), хранится в SQLite.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in PulseArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.PulseCapture(ctx, in)
		})

	t.add("pulse_read",
		"Читает сырые снимки пульса за последние N минут (по умолчанию 60, максимум 1440) — поминутная хронология неба.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in PulseArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.PulseRead(ctx, in)
		})

	t.add("pulse_summary",
		"Агрегированный отчёт по пульсу за окно в часах: моменты смены лунного знака, таймлайн транзитов (когда вошёл/ушёл/точнее всего), температурная кривая.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in PulseArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.PulseSummary(ctx, in)
		})

	// Пайплайн «Подбор благоприятной даты»: шаги идут строго по цепочке,
	// между шагами передаётся только election_id (данные живут на сервере).
	t.add("elect_start",
		"Шаг 1/4 пайплайна подбора благоприятной даты: фиксирует цель, место и натальную карту, создаёт сессию и возвращает election_id — передавайте его во все следующие шаги. Дальше вызовите elect_scan.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ElectStartArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.ElectStart(ctx, in)
		})

	t.add("elect_scan",
		"Шаг 2/4: по каждому дню окна собирает сырые факты — тесные транзиты к наталу (орбис до 3°) и Луну (знак, фаза, стадия). Дальше вызовите elect_score.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ElectStepArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.ElectScan(ctx, in)
		})

	t.add("elect_score",
		"Шаг 3/4: оценивает просканированные дни (гармоничные/напряжённые аспекты, знак и фаза Луны против цели) и выбирает топ-3. Дальше вызовите elect_report.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ElectStepArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.ElectScore(ctx, in)
		})

	t.add("elect_report",
		"Шаг 4/4 (финальный): собирает markdown-отчёт подбора — цель, место, топ-даты с обоснованием, аффирмация. Цепочка завершена (next пустой).",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in ElectStepArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.ElectReport(ctx, in)
		})

	// Пайплайн «Утренний брифинг»: start → collect → compose.
	t.add("brief_start",
		"Шаг 1/3 пайплайна утреннего брифинга: фиксирует город и создаёт сессию, возвращает brief_id — передавайте его в brief_collect и brief_compose. Дальше вызовите brief_collect.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in BriefStartArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.BriefStart(ctx, in)
		})

	t.add("brief_collect",
		"Шаг 2/3: наполняет сессию секциями — погода, Луна, тесные транзиты к последней карте, подсказки дня. Дальше вызовите brief_compose.",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in BriefStepArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.BriefCollect(ctx, in)
		})

	t.add("brief_compose",
		"Шаг 3/3 (финальный): собирает markdown-брифинг из собранных секций. Цепочка завершена (next пустой).",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in BriefStepArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.Svc.BriefCompose(ctx, in)
		})

	return t
}

// add регистрирует один инструмент.
func (t *Toolkit) add(name, description string, fn invoker) {
	t.defs = append(t.defs, ToolDef{Name: name, Description: description})
	t.invokers[name] = fn
}

// Defs — список описаний (для -list и status-страницы).
func (t *Toolkit) Defs() []ToolDef { return t.defs }

// Invoke вызывает инструмент по имени (используется режимом -probe).
func (t *Toolkit) Invoke(ctx context.Context, name string, raw json.RawMessage) (any, error) {
	fn, ok := t.invokers[name]
	if !ok {
		return nil, fmt.Errorf("неизвестный инструмент %q; доступно: %v", name, names(t.defs))
	}
	return fn(ctx, raw)
}

// Attach регистрирует инструменты на MCP-сервере. Аргументы типизированы:
// SDK сам строит input-схему из json/jsonschema-тегов и валидирует вход.
func (t *Toolkit) Attach(server *mcp.Server) {
	mcp.AddTool(server, t.tool("natal_build_chart"), func(ctx context.Context, _ *mcp.CallToolRequest, in NatalArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "natal_build_chart", func() (any, error) { return t.Svc.BuildNatal(ctx, in) })
	})
	mcp.AddTool(server, t.tool("transit_today"), func(ctx context.Context, _ *mcp.CallToolRequest, in ChartArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "transit_today", func() (any, error) { return t.Svc.TransitToday(ctx, in) })
	})
	mcp.AddTool(server, t.tool("transit_forecast"), func(ctx context.Context, _ *mcp.CallToolRequest, in ForecastArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "transit_forecast", func() (any, error) { return t.Svc.TransitForecast(ctx, in) })
	})
	mcp.AddTool(server, t.tool("synastry_compare"), func(ctx context.Context, _ *mcp.CallToolRequest, in SynastryArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "synastry_compare", func() (any, error) { return t.Svc.Synastry(ctx, in) })
	})
	mcp.AddTool(server, t.tool("solar_return"), func(ctx context.Context, _ *mcp.CallToolRequest, in SolarArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "solar_return", func() (any, error) { return t.Svc.SolarReturn(ctx, in) })
	})
	mcp.AddTool(server, t.tool("moon_phase"), func(ctx context.Context, _ *mcp.CallToolRequest, in MoonArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "moon_phase", func() (any, error) { return t.Svc.MoonPhase(ctx, in) })
	})
	mcp.AddTool(server, t.tool("vibe_today"), func(ctx context.Context, _ *mcp.CallToolRequest, in VibeArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "vibe_today", func() (any, error) { return t.Svc.VibeToday(ctx, in) })
	})
	mcp.AddTool(server, t.tool("chart_render"), func(ctx context.Context, _ *mcp.CallToolRequest, in RenderArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "chart_render", func() (any, error) { return t.Svc.RenderChart(ctx, in) })
	})
	mcp.AddTool(server, t.tool("orbit_status"), func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "orbit_status", func() (any, error) { return t.Svc.Status(ctx) })
	})
	mcp.AddTool(server, t.tool("daily_affirmation"), func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "daily_affirmation", func() (any, error) { return t.Svc.Affirmation(ctx) })
	})
	mcp.AddTool(server, t.tool("journal_capture"), func(ctx context.Context, _ *mcp.CallToolRequest, in JournalArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "journal_capture", func() (any, error) { return t.Svc.JournalCapture(ctx, in) })
	})
	mcp.AddTool(server, t.tool("journal_read"), func(ctx context.Context, _ *mcp.CallToolRequest, in JournalArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "journal_read", func() (any, error) { return t.Svc.JournalRead(ctx, in) })
	})
	mcp.AddTool(server, t.tool("journal_summary"), func(ctx context.Context, _ *mcp.CallToolRequest, in JournalArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "journal_summary", func() (any, error) { return t.Svc.JournalSummary(ctx, in) })
	})
	mcp.AddTool(server, t.tool("pulse_capture"), func(ctx context.Context, _ *mcp.CallToolRequest, in PulseArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "pulse_capture", func() (any, error) { return t.Svc.PulseCapture(ctx, in) })
	})
	mcp.AddTool(server, t.tool("pulse_read"), func(ctx context.Context, _ *mcp.CallToolRequest, in PulseArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "pulse_read", func() (any, error) { return t.Svc.PulseRead(ctx, in) })
	})
	mcp.AddTool(server, t.tool("pulse_summary"), func(ctx context.Context, _ *mcp.CallToolRequest, in PulseArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "pulse_summary", func() (any, error) { return t.Svc.PulseSummary(ctx, in) })
	})
	mcp.AddTool(server, t.tool("elect_start"), func(ctx context.Context, _ *mcp.CallToolRequest, in ElectStartArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "elect_start", func() (any, error) { return t.Svc.ElectStart(ctx, in) })
	})
	mcp.AddTool(server, t.tool("elect_scan"), func(ctx context.Context, _ *mcp.CallToolRequest, in ElectStepArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "elect_scan", func() (any, error) { return t.Svc.ElectScan(ctx, in) })
	})
	mcp.AddTool(server, t.tool("elect_score"), func(ctx context.Context, _ *mcp.CallToolRequest, in ElectStepArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "elect_score", func() (any, error) { return t.Svc.ElectScore(ctx, in) })
	})
	mcp.AddTool(server, t.tool("elect_report"), func(ctx context.Context, _ *mcp.CallToolRequest, in ElectStepArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "elect_report", func() (any, error) { return t.Svc.ElectReport(ctx, in) })
	})
	mcp.AddTool(server, t.tool("brief_start"), func(ctx context.Context, _ *mcp.CallToolRequest, in BriefStartArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "brief_start", func() (any, error) { return t.Svc.BriefStart(ctx, in) })
	})
	mcp.AddTool(server, t.tool("brief_collect"), func(ctx context.Context, _ *mcp.CallToolRequest, in BriefStepArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "brief_collect", func() (any, error) { return t.Svc.BriefCollect(ctx, in) })
	})
	mcp.AddTool(server, t.tool("brief_compose"), func(ctx context.Context, _ *mcp.CallToolRequest, in BriefStepArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "brief_compose", func() (any, error) { return t.Svc.BriefCompose(ctx, in) })
	})
}

// tool — описание инструмента по имени.
func (t *Toolkit) tool(name string) *mcp.Tool {
	for _, d := range t.defs {
		if d.Name == name {
			return &mcp.Tool{Name: d.Name, Description: d.Description}
		}
	}
	return &mcp.Tool{Name: name}
}

// call — общий каркас: логика → результат MCP, ошибка → IsError-результат.
func (t *Toolkit) call(ctx context.Context, name string, fn func() (any, error)) (*mcp.CallToolResult, any, error) {
	payload, err := fn()
	if err != nil {
		log.Printf("tool %s: %v", name, err)
		return errorResult(err), nil, nil
	}
	if rr, ok := payload.(*RenderResult); ok {
		return imageResult(rr), payload, nil
	}
	return okResult(payload), payload, nil
}

// decode разбирает JSON-аргументы инструмента.
func decode(raw json.RawMessage, into any) error {
	if len(raw) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("неверные аргументы: %w", err)
	}
	return nil
}

// okResult — текстовый результат с компактным JSON-payload.
func okResult(payload any) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: marshal(payload)}},
	}
}

// imageResult — текстовая сводка для модели плюс изображение для клиента:
// клиент показывает картинку пользователю, а модель видит только текст.
func imageResult(rr *RenderResult) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: marshal(rr.Model)},
			&mcp.ImageContent{Data: rr.SVG, MIMEType: "image/svg+xml"},
		},
	}
}

// errorResult — ошибка инструмента (IsError=true): модель увидит текст.
func errorResult(err error) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: "⚠️ " + err.Error()}},
	}
}

// marshal — JSON без лишних экранирований, читаемый в логах.
func marshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"error":"payload: %v"}`, err)
	}
	return string(b)
}

func names(defs []ToolDef) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	return out
}
