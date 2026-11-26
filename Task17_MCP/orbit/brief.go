package orbit

// Утренний брифинг — пайплайн из трёх инструментов:
//
//	brief_start → brief_collect → brief_compose
//
// brief_start только фиксирует место и создаёт сессию, brief_collect
// наполняет её секциями (погода, Луна, транзиты — переиспользует vibe_today),
// brief_compose склеивает финальный markdown-брифинг. Между шагами модель
// передаёт только brief_id. Обрывы цепочки: simulate ("error"/"empty"),
// сбой провайдера, истёкший TTL, невалидный порядок шагов.

import (
	"context"
	"fmt"
	"strings"
	"time"

	"ai-chat-go/openmeteo"
)

// briefTTL — время жизни сессии брифинга.
const briefTTL = time.Hour

// BriefSession — состояние пайплайна брифинга.
type BriefSession struct {
	ID        string
	City      string // исходная строка пользователя (для resolveLocation)
	Place     openmeteo.Location
	Status    string // started → collected → done
	Data      map[string]any // секции брифинга: weather, moon, transits, hints
	CreatedAt time.Time
}

// briefLookup находит живую сессию брифинга (TTL-checked).
func (s *Service) briefLookup(id string) (*BriefSession, error) {
	id = strings.TrimSpace(id)
	s.briefMu.Lock()
	defer s.briefMu.Unlock()
	sess, ok := s.briefSessions[id]
	if !ok {
		return nil, fmt.Errorf("брифинг %q не найден (истёк или ID неверен) — начните заново с brief_start", id)
	}
	if s.now().Sub(sess.CreatedAt) > briefTTL {
		delete(s.briefSessions, id)
		return nil, fmt.Errorf("брифинг %q истёк (TTL %s) — начните заново с brief_start", id, briefTTL)
	}
	return sess, nil
}

// ---------- Шаг 1: brief_start ----------

// BriefStart создаёт сессию брифинга: геокодирует место и запоминает его.
func (s *Service) BriefStart(ctx context.Context, in BriefStartArgs) (map[string]any, error) {
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге brief_start (запрошен аргументом simulate)")
	}
	loc, err := s.resolveLocation(ctx, in.City)
	if err != nil {
		return nil, fmt.Errorf("геокодинг: %w", err)
	}
	_, wantEmpty := simulateOf(in.Simulate)

	sess := &BriefSession{
		ID:        newSessionID("br_"),
		City:      strings.TrimSpace(in.City),
		Place:     loc,
		Status:    "started",
		Data:      map[string]any{},
		CreatedAt: s.now(),
	}
	s.briefMu.Lock()
	s.briefSessions[sess.ID] = sess
	s.briefMu.Unlock()

	details := map[string]any{"place": loc.Label()}
	if wantEmpty {
		details = map[string]any{}
	}
	return map[string]any{
		"brief_id": sess.ID,
		"status":   sess.Status,
		"next":     "brief_collect",
		"details":  details,
	}, nil
}

// ---------- Шаг 2: brief_collect ----------

// BriefCollect наполняет сессию секциями брифинга — переиспользует
// vibe_today целиком (погода с кэшем, Луна, тесные транзиты к последней
// карте, детерминированные подсказки). simulate="empty" собирает сессию,
// из которой секции выпали, — цепочка сломается позже, на brief_compose.
func (s *Service) BriefCollect(ctx context.Context, in BriefStepArgs) (map[string]any, error) {
	sess, err := s.briefLookup(in.BriefID)
	if err != nil {
		return nil, err
	}
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге brief_collect — брифинг %s остался в статусе %s", sess.ID, sess.Status)
	}
	if sess.Status != "started" {
		return nil, fmt.Errorf("шаг brief_collect уже выполнен (статус %s) — вызовите brief_compose", sess.Status)
	}

	_, wantEmpty := simulateOf(in.Simulate)
	data := map[string]any{"place": sess.Place.Label()}
	if !wantEmpty {
		vibe, verr := s.VibeToday(ctx, VibeArgs{City: sess.City})
		if verr != nil {
			return nil, fmt.Errorf("сбор данных брифинга: %w", verr)
		}
		for _, key := range []string{"date", "city", "weather", "moon", "next_lunar_eclipse", "transits", "hints"} {
			if v, ok := vibe[key]; ok {
				data[key] = v
			}
		}
	}

	s.briefMu.Lock()
	sess.Data = data
	sess.Status = "collected"
	s.briefMu.Unlock()

	sections := 0
	for _, key := range []string{"weather", "moon", "transits", "hints"} {
		if _, ok := data[key]; ok {
			sections++
		}
	}
	return map[string]any{
		"brief_id": sess.ID,
		"status":   "collected",
		"next":     "brief_compose",
		"details":  map[string]any{"sections": sections, "date": data["date"]},
	}, nil
}

// ---------- Шаг 3: brief_compose ----------

// BriefCompose собирает финальный markdown-брифинг из секций сессии.
// Естественный обрыв: collect вернул пустоту (simulate="empty" или всё
// упало молча) — compose не из чего собирать.
func (s *Service) BriefCompose(ctx context.Context, in BriefStepArgs) (map[string]any, error) {
	sess, err := s.briefLookup(in.BriefID)
	if err != nil {
		return nil, err
	}
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге brief_compose — брифинг %s остался в статусе %s", sess.ID, sess.Status)
	}
	if sess.Status != "collected" {
		return nil, fmt.Errorf("данные не собраны (статус %s) — сначала brief_collect", sess.Status)
	}
	sections := 0
	for _, key := range []string{"weather", "moon", "transits", "hints"} {
		if _, ok := sess.Data[key]; ok {
			sections++
		}
	}
	if sections == 0 {
		return nil, fmt.Errorf("секции брифинга не собраны — вызовите brief_collect (или начните заново с brief_start)")
	}

	_, wantEmpty := simulateOf(in.Simulate)
	var b strings.Builder
	fmt.Fprintf(&b, "# Утренний брифинг — %s\n\n", sess.Place.Label())
	if date := str(sess.Data["date"]); date != "" {
		fmt.Fprintf(&b, "_%s_\n\n", date)
	}
	if w, ok := sess.Data["weather"].(openmeteo.Weather); ok {
		fmt.Fprintf(&b, "**Погода:** %s, %.1f °C (ощущается %.1f), за день %.1f…%.1f °C, осадки %d%%, ветер %.0f км/ч.\n\n",
			w.Description, w.Temperature, w.Apparent, w.TempMin, w.TempMax, w.PrecipProb, w.WindSpeed)
	}
	if m, ok := sess.Data["moon"].(map[string]any); ok {
		fmt.Fprintf(&b, "**Луна:** %s %s, знак %s, освещённость %v.\n\n",
			str(m["emoji"]), str(m["name"]), str(m["moon_sign"]), m["illumination"])
	}
	if e, ok := sess.Data["next_lunar_eclipse"].(map[string]any); ok && e != nil && e["date"] != nil {
		fmt.Fprintf(&b, "Ближайшее лунное затмение: %v (%v).\n\n", e["date"], e["type"])
	}
	if tr, ok := sess.Data["transits"].(map[string]any); ok {
		if note := str(tr["note"]); note != "" {
			fmt.Fprintf(&b, "**Транзиты:** %s\n\n", note)
		} else if chart := str(tr["chart"]); chart != "" {
			fmt.Fprintf(&b, "**Транзиты к карте %s:**\n", chart)
			if aspects, ok := tr["aspects"].([]Aspect); ok {
				for i, a := range aspects {
					if i == 5 {
						break
					}
					fmt.Fprintf(&b, "- %s (орбис %.1f°)\n", a.String(), a.Orbit)
				}
			}
			b.WriteString("\n")
		}
	}
	if hints, ok := sess.Data["hints"].([]string); ok && len(hints) > 0 {
		b.WriteString("**Подсказки дня:**\n")
		for _, h := range hints {
			b.WriteString("- " + h + "\n")
		}
		b.WriteString("\n")
	}
	if res, aerr := s.Affirm.Russian(ctx); aerr == nil {
		fmt.Fprintf(&b, "_Аффирмация дня: %s %s_\n", res.Affirmation, res.Emoji)
	}

	brief := b.String()
	if wantEmpty {
		brief = ""
	}
	s.briefMu.Lock()
	sess.Status = "done"
	s.briefMu.Unlock()
	return map[string]any{
		"brief_id": sess.ID,
		"status":   "done",
		"next":     "", // цепочка завершена
		"details": map[string]any{
			"brief": brief,
		},
	}, nil
}
