package orbit

// Подбор благоприятной даты — пайплайн из четырёх инструментов:
//
//	elect_start → elect_scan → elect_score → elect_report
//
// Промежуточные данные (аспекты по дням, Луна, скоры) живут на сервере в
// сессии; модель между шагами передаёт только election_id, а поле next в
// каждом ответе подсказывает, какой инструмент звать дальше. Цепочку можно
// сломать в середине: аргумент simulate ("error"/"empty") даёт детермини-
// рованный сбой, естественные обрывы — отсутствие карты, сбой движка,
// истёкший TTL сессии или невалидный порядок шагов.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"time"

	"ai-chat-go/openmeteo"
)

// electTTL — время жизни сессии подбора: после него любой шаг требует
// начать заново (естественный обрыв цепочки по времени).
const electTTL = time.Hour

// ElectSession — состояние пайплайна подбора даты.
type ElectSession struct {
	ID        string
	Goal      string
	City      string // исходная строка пользователя (для логов и повторного геокодинга)
	Place     openmeteo.Location
	Chart     string // имя натальной карты
	Window    int    // окно поиска в днях
	Status    string // started → scanned → scored → done
	Days      []ElectDay
	Top       []ElectDayScore
	CreatedAt time.Time
}

// ElectDay — сырые факты одного дня (результат elect_scan).
type ElectDay struct {
	Date      string   `json:"date"`
	MoonSign  string   `json:"moon_sign,omitempty"`  // знак Луны: Psc, Gem…
	MoonPhase string   `json:"moon_phase,omitempty"` // название фазы
	MoonStage string   `json:"moon_stage,omitempty"` // waxing / waning
	Aspects   []Aspect `json:"aspects,omitempty"`    // транзиты к наталу, орбис ≤3°
}

// ElectDayScore — оценка одного дня (результат elect_score).
type ElectDayScore struct {
	Date    string   `json:"date"`
	Score   int      `json:"score"`
	Reasons []string `json:"reasons,omitempty"`
}

// newSessionID — короткий ID сессии пайплайна с префиксом.
func newSessionID(prefix string) string {
	b := make([]byte, 4)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%s%x", prefix, time.Now().UnixNano())
	}
	return prefix + hex.EncodeToString(b)
}

// simulateOf разбирает аргумент simulate: ("error", "empty").
func simulateOf(v string) (wantErr, wantEmpty bool) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "error":
		return true, false
	case "empty":
		return false, true
	}
	return false, false
}

// electLookup находит живую сессию подбора (TTL-checked).
func (s *Service) electLookup(id string) (*ElectSession, error) {
	id = strings.TrimSpace(id)
	s.electMu.Lock()
	defer s.electMu.Unlock()
	sess, ok := s.electSessions[id]
	if !ok {
		return nil, fmt.Errorf("сессия подбора %q не найдена (истекла или ID неверен) — начните заново с elect_start", id)
	}
	if s.now().Sub(sess.CreatedAt) > electTTL {
		delete(s.electSessions, id)
		return nil, fmt.Errorf("сессия подбора %q истекла (TTL %s) — начните заново с elect_start", id, electTTL)
	}
	return sess, nil
}

// placeTZ — таймзона места сессии (UTC при сбое).
func placeTZ(loc openmeteo.Location) *time.Location {
	tz, err := time.LoadLocation(loc.Timezone)
	if err != nil {
		return time.UTC
	}
	return tz
}

// ---------- Шаг 1: elect_start ----------

// ElectStart создаёт сессию подбора: цель, место (геокодинг), натальная
// карта и окно поиска. simulate="empty" вернёт успешный пустой результат
// (окно 0) — пустота прокатится дальше и сломает цепочку на elect_score.
func (s *Service) ElectStart(ctx context.Context, in ElectStartArgs) (map[string]any, error) {
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге elect_start (запрошен аргументом simulate)")
	}
	goal := strings.TrimSpace(in.Goal)
	if goal == "" {
		return nil, fmt.Errorf("задайте цель подбора (goal), например «запуск блога»")
	}
	loc, err := s.resolveLocation(ctx, in.City)
	if err != nil {
		return nil, fmt.Errorf("геокодинг: %w", err)
	}
	ch, err := s.Store.Get(in.Chart)
	if err != nil {
		return nil, fmt.Errorf("натальная карта: %w", err)
	}
	window := 14
	if in.WindowDays != nil {
		window = *in.WindowDays
	}
	_, wantEmpty := simulateOf(in.Simulate)
	if wantEmpty {
		window = 0
	}
	if window < 0 {
		window = 0
	}
	if window > 31 {
		window = 31 // лимит движка транзитов (как в transit_forecast)
	}

	sess := &ElectSession{
		ID:        newSessionID("el_"),
		Goal:      goal,
		City:      strings.TrimSpace(in.City),
		Place:     loc,
		Chart:     ch.Name,
		Window:    window,
		Status:    "started",
		CreatedAt: s.now(),
	}
	s.electMu.Lock()
	s.electSessions[sess.ID] = sess
	s.electMu.Unlock()

	details := map[string]any{
		"goal":        goal,
		"place":       loc.Label(),
		"chart":       ch.Name,
		"window_days": window,
	}
	if wantEmpty {
		details = map[string]any{}
	}
	return map[string]any{
		"election_id": sess.ID,
		"status":      sess.Status,
		"next":        "elect_scan",
		"details":     details,
	}, nil
}

// ---------- Шаг 2: elect_scan ----------

// ElectScan собирает по каждому дню окна сырые факты: тесные транзиты к
// наталу (орбис ≤3°, как в transit_forecast) и Луну (знак, фаза, стадия).
// Естественный обрыв: сбой движка на любом дне.
func (s *Service) ElectScan(ctx context.Context, in ElectStepArgs) (map[string]any, error) {
	sess, err := s.electLookup(in.ElectionID)
	if err != nil {
		return nil, err
	}
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге elect_scan — сессия %s осталась в статусе %s", sess.ID, sess.Status)
	}
	if sess.Status != "started" {
		return nil, fmt.Errorf("шаг elect_scan уже выполнен (статус %s) — вызовите elect_score", sess.Status)
	}
	ch, err := s.Store.Get(sess.Chart)
	if err != nil {
		return nil, err
	}
	tz := placeTZ(sess.Place)
	start := s.now().In(tz)

	_, wantEmpty := simulateOf(in.Simulate)
	days := make([]ElectDay, 0, sess.Window)
	for i := 0; i < sess.Window; i++ {
		day := start.AddDate(0, 0, i)
		noon := time.Date(day.Year(), day.Month(), day.Day(), 12, 0, 0, 0, tz)
		label := day.Format("2006-01-02")

		resp, terr := s.Astro.TransitData(ctx, &ch.Subject, subjectPtr(transitSubject(ch, noon)))
		if terr != nil {
			return nil, fmt.Errorf("скан, день %s: транзиты: %w", label, terr)
		}
		mresp, merr := s.Astro.MoonPhaseContext(ctx, noon, sess.Place.Latitude, sess.Place.Longitude, sess.Place.Timezone)
		if merr != nil {
			return nil, fmt.Errorf("скан, день %s: Луна: %w", label, merr)
		}
		moon := mapAny(mapAny(mresp["moon_phase_overview"])["moon"])
		days = append(days, ElectDay{
			Date:      label,
			MoonSign:  str(mapAny(moon["zodiac"])["moon_sign"]),
			MoonPhase: str(moon["phase_name"]),
			MoonStage: str(moon["stage"]),
			Aspects:   Aspects(ChartData(resp), 3.0, 4),
		})
	}
	if wantEmpty {
		days = nil
	}

	s.electMu.Lock()
	sess.Days = days
	sess.Status = "scanned"
	s.electMu.Unlock()
	return map[string]any{
		"election_id": sess.ID,
		"status":      "scanned",
		"next":        "elect_score",
		"details":     map[string]any{"days": sess.Window, "candidates": len(days)},
	}, nil
}

// ---------- Шаг 3: elect_score ----------

// goalSigns — знаки Луны (аббревиатуры движка), резонирующие с целью.
func goalSigns(goal string) []string {
	g := strings.ToLower(goal)
	switch {
	case containsAny(g, "блог", "текст", "пись", "публикац", "стать", "переговор", "учеб", "учёб", "экзамен", "интерв", "созвон", "питч", "pitch", "blog", "interview"):
		return []string{"gem", "lib", "aqu"}
	case containsAny(g, "старт", "запуск", "спорт", "соревнов", "конкурс", "трениров", "начать", "открыт"):
		return []string{"ari", "leo", "sgr"}
	case containsAny(g, "финанс", "деньг", "бюджет", "сделк", "покупк", "продаж", "инвест", "долг"):
		return []string{"tau", "vir", "cap"}
	case containsAny(g, "свадьб", "роман", "знаком", "отношен", "творч", "вечерин", "праздн"):
		return []string{"lib", "tau", "leo", "psc"}
	default:
		return nil
	}
}

// containsAny — есть ли в строке хотя бы одна из подстрок.
func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// sign3 — знак к аббревиатуре из трёх букв (движок может отдать и полное имя).
func sign3(sign string) string {
	sign = strings.ToLower(strings.TrimSpace(sign))
	if len(sign) > 3 {
		sign = sign[:3]
	}
	return sign
}

// scoreDay — эвристический скор дня: гармоничные аспекты и растущая Луна
// плюсуют, напряжённые минусуют, знак Луны сверяется с целью.
func scoreDay(d ElectDay, goal string) (int, []string) {
	score := 0
	reasons := []string{}
	for _, a := range d.Aspects {
		weight := 2 + int(math.Round(3-a.Orbit)) // чем теснее, тем сильнее
		switch a.Aspect {
		case "trine", "sextile":
			score += weight
			reasons = append(reasons, fmt.Sprintf("%s (орбис %.1f°) — гармоничный аспект", a.String(), a.Orbit))
		case "square", "opposition":
			score -= weight
			reasons = append(reasons, fmt.Sprintf("%s (орбис %.1f°) — напряжённый аспект", a.String(), a.Orbit))
		case "conjunction":
			score++
			reasons = append(reasons, fmt.Sprintf("%s — соединение, усиление (нейтрально)", a.String()))
		}
	}
	if signs := goalSigns(goal); len(signs) > 0 && slices.Contains(signs, sign3(d.MoonSign)) {
		score += 3
		reasons = append(reasons, fmt.Sprintf("Луна в %s — резонанс с целью «%s»", d.MoonSign, goal))
	}
	if strings.EqualFold(d.MoonStage, "waxing") {
		score++
		reasons = append(reasons, "Луна растёт — благоприятно для роста и начинаний")
	}
	return score, reasons
}

// ElectScore ранжирует просканированные дни и сохраняет топ-3 в сессию.
// Естественный обрыв: скан вернул 0 дней (пустота с elect_start пустым
// окном доезжает сюда и рвёт цепочку).
func (s *Service) ElectScore(ctx context.Context, in ElectStepArgs) (map[string]any, error) {
	sess, err := s.electLookup(in.ElectionID)
	if err != nil {
		return nil, err
	}
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге elect_score — сессия %s осталась в статусе %s", sess.ID, sess.Status)
	}
	if sess.Status != "scanned" {
		return nil, fmt.Errorf("данные не просканированы (статус %s) — сначала elect_scan", sess.Status)
	}
	if len(sess.Days) == 0 {
		return nil, fmt.Errorf("кандидатов нет — сканирование вернуло 0 дней; проверьте окно и начните заново с elect_start")
	}

	_, wantEmpty := simulateOf(in.Simulate)
	scores := make([]ElectDayScore, 0, len(sess.Days))
	for _, d := range sess.Days {
		score, reasons := scoreDay(d, sess.Goal)
		scores = append(scores, ElectDayScore{Date: d.Date, Score: score, Reasons: reasons})
	}
	sort.SliceStable(scores, func(i, j int) bool { return scores[i].Score > scores[j].Score })
	top := scores
	if len(top) > 3 {
		top = top[:3]
	}
	if wantEmpty {
		top = []ElectDayScore{}
	}

	s.electMu.Lock()
	sess.Top = top
	sess.Status = "scored"
	s.electMu.Unlock()
	return map[string]any{
		"election_id": sess.ID,
		"status":      "scored",
		"next":        "elect_report",
		"details":     map[string]any{"evaluated": len(scores), "top": top},
	}, nil
}

// ---------- Шаг 4: elect_report ----------

// ElectReport собирает финальный markdown-отчёт: цель, место, топ-даты с
// обоснованием и аффирмацией-напутствием. Это последний шаг цепочки
// (next пустой) — после него session остаётся в статусе done для повторного
// чтения, пока не истечёт TTL.
func (s *Service) ElectReport(ctx context.Context, in ElectStepArgs) (map[string]any, error) {
	sess, err := s.electLookup(in.ElectionID)
	if err != nil {
		return nil, err
	}
	if wantErr, _ := simulateOf(in.Simulate); wantErr {
		return nil, fmt.Errorf("simulate: сбой на шаге elect_report — сессия %s осталась в статусе %s", sess.ID, sess.Status)
	}
	if sess.Status != "scored" {
		return nil, fmt.Errorf("дни не оценены (статус %s) — сначала elect_score", sess.Status)
	}
	if len(sess.Top) == 0 {
		return nil, fmt.Errorf("нет оценённых дней — отчёт строить не из чего; начните заново с elect_start")
	}

	_, wantEmpty := simulateOf(in.Simulate)
	var b strings.Builder
	fmt.Fprintf(&b, "## Подбор даты: %s\n\n", sess.Goal)
	fmt.Fprintf(&b, "Карта: %s · Место: %s · Окно: %d дн.\n\n", sess.Chart, sess.Place.Label(), sess.Window)
	medals := []string{"🥇", "🥈", "🥉"}
	for i, t := range sess.Top {
		medal := fmt.Sprintf("%d.", i+1)
		if i < len(medals) {
			medal = medals[i]
		}
		fmt.Fprintf(&b, "### %s %s — %d очк.\n", medal, t.Date, t.Score)
		for _, r := range t.Reasons {
			b.WriteString("- " + r + "\n")
		}
		b.WriteString("\n")
	}
	if res, aerr := s.Affirm.Russian(ctx); aerr == nil {
		fmt.Fprintf(&b, "_Аффирмация: %s %s_\n", res.Affirmation, res.Emoji)
	}
	b.WriteString("\n_Оценки эвристические: аспекты, знак и фаза Луны. Финальную трактовку формулирует модель._\n")

	report := b.String()
	if wantEmpty {
		report = ""
	}
	s.electMu.Lock()
	sess.Status = "done"
	s.electMu.Unlock()
	return map[string]any{
		"election_id": sess.ID,
		"status":      "done",
		"next":        "", // цепочка завершена
		"details": map[string]any{
			"top":    sess.Top,
			"report": report,
		},
	}, nil
}
