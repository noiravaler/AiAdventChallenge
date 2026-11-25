package orbit

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"
	"time"
)

// JournalEntry — суточный снимок наблюдений: погода, Луна, тесные транзиты
// и аффирмация дня. Числовые поля — указатели: «не снимали» ≠ «ноль».
type JournalEntry struct {
	Date        string    `json:"date"` // YYYY-MM-DD (локальная дата места)
	City        string    `json:"city,omitempty"`
	CapturedAt  time.Time `json:"captured_at"`
	Temp        *float64  `json:"temp,omitempty"`     // °C на момент съёма
	TempMax     *float64  `json:"temp_max,omitempty"` // прогноз на день
	TempMin     *float64  `json:"temp_min,omitempty"`
	PrecipProb  *int      `json:"precip_probability,omitempty"` // % вероятность осадков
	Weather     string    `json:"weather,omitempty"`            // «ясно», «дождь»…
	MoonPhase   string    `json:"moon_phase,omitempty"`         // «полнолуние»…
	MoonSign    string    `json:"moon_sign,omitempty"`
	Affirmation string    `json:"affirmation,omitempty"`
	Emoji       string    `json:"emoji,omitempty"`
	Chart       string    `json:"chart,omitempty"`    // имя карты транзитов
	Transits    []Aspect  `json:"transits,omitempty"` // топ-3 тесных аспекта
}

// Journal — дневник наблюдений: JSON-файл, не более одной записи на дату
// (повторный снимок за тот же день перезаписывает её). Все методы безопасны
// для конкурентного использования.
type Journal struct {
	path string

	mu      sync.Mutex
	entries map[string]*JournalEntry // ключ — дата YYYY-MM-DD
}

type journalFile struct {
	Entries []*JournalEntry `json:"entries"`
}

// NewJournal загружает дневник с диска (отсутствующий файл — не ошибка).
func NewJournal(path string) (*Journal, error) {
	j := &Journal{path: path, entries: map[string]*JournalEntry{}}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return j, nil
		}
		return nil, err
	}
	var f journalFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("битый файл дневника %s: %w", path, err)
	}
	for _, e := range f.Entries {
		if e == nil || !journalDateRe.MatchString(e.Date) {
			continue
		}
		j.entries[e.Date] = e
	}
	return j, nil
}

// Upsert добавляет запись или перезаписывает запись за ту же дату.
func (j *Journal) Upsert(e *JournalEntry) error {
	if e == nil || !journalDateRe.MatchString(e.Date) {
		return fmt.Errorf("запись без корректной даты YYYY-MM-DD не может быть сохранена")
	}
	if e.CapturedAt.IsZero() {
		e.CapturedAt = time.Now()
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	j.entries[e.Date] = e
	return j.persistLocked()
}

// Since возвращает записи за последние days дней (включая сегодня),
// отсортированные по дате. days <= 0 — все записи. today — YYYY-MM-DD
// для тестов; пусто — текущая дата.
func (j *Journal) Since(days int, today string) []*JournalEntry {
	j.mu.Lock()
	defer j.mu.Unlock()

	dates := make([]string, 0, len(j.entries))
	for d := range j.entries {
		dates = append(dates, d)
	}
	sort.Strings(dates)

	cutoff := ""
	if days > 0 {
		if today == "" {
			today = time.Now().Format("2006-01-02")
		}
		if base, err := time.Parse("2006-01-02", today); err == nil {
			cutoff = base.AddDate(0, 0, -(days - 1)).Format("2006-01-02")
		}
	}

	out := make([]*JournalEntry, 0, len(dates))
	for _, d := range dates {
		if cutoff != "" && d < cutoff {
			continue
		}
		out = append(out, j.entries[d])
	}
	return out
}

// Count — число записей в дневнике.
func (j *Journal) Count() int {
	j.mu.Lock()
	defer j.mu.Unlock()
	return len(j.entries)
}

func (j *Journal) persistLocked() error {
	f := journalFile{}
	for _, d := range sortedKeys(j.entries) {
		f.Entries = append(f.Entries, j.entries[d])
	}
	raw, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(j.path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := j.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, j.path)
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// transitStat — агрегат по одному транзиту за период.
type transitStat struct {
	Transit  string  `json:"transit"`
	Days     int     `json:"days"`
	MinOrbit float64 `json:"min_orbit"`
}

// Summarize — агрегированный отчёт по записям дневника за период:
// температура, осадки, фазы Луны, частые транзиты, аффирмации, напоминания.
// today — YYYY-MM-DD для тестов; пусто — текущая дата.
func Summarize(entries []*JournalEntry, days int, today string) map[string]any {
	if today == "" {
		today = time.Now().Format("2006-01-02")
	}
	out := map[string]any{
		"period_days":    days,
		"days_with_data": len(entries),
	}
	if len(entries) == 0 {
		out["note"] = "журнал пуст — снимки делает journal_capture (по расписанию или вручную)"
		return out
	}

	out["first_date"] = entries[0].Date
	out["last_date"] = entries[len(entries)-1].Date

	// Температура по «текущей» на момент съёма.
	temps := []float64{}
	for _, e := range entries {
		if e.Temp != nil {
			temps = append(temps, *e.Temp)
		}
	}
	if len(temps) > 0 {
		sum, min, max := 0.0, temps[0], temps[0]
		for _, t := range temps {
			sum += t
			min = math.Min(min, t)
			max = math.Max(max, t)
		}
		out["temperature"] = map[string]any{
			"avg": round1(sum / float64(len(temps))),
			"min": round1(min),
			"max": round1(max),
		}
	}

	// Осадки.
	var precipDates []string
	maxProb := 0
	for _, e := range entries {
		if e.PrecipProb != nil {
			maxProb = max(maxProb, *e.PrecipProb)
			if *e.PrecipProb >= 50 {
				precipDates = append(precipDates, e.Date)
			}
		}
	}
	if maxProb > 0 {
		daysPrecip := len(precipDates)
		if daysPrecip > 10 {
			precipDates = append(precipDates[:10], "…")
		}
		out["precipitation"] = map[string]any{
			"days_with_precip": daysPrecip,
			"max_probability":  maxProb,
			"dates":            precipDates,
		}
	}

	// Самые частые описания погоды.
	weatherCount := map[string]int{}
	for _, e := range entries {
		if e.Weather != "" {
			weatherCount[e.Weather]++
		}
	}
	if len(weatherCount) > 0 {
		type wc struct {
			Weather string `json:"weather"`
			Days    int    `json:"days"`
		}
		var list []wc
		for w, n := range weatherCount {
			list = append(list, wc{w, n})
		}
		sort.Slice(list, func(i, k int) bool {
			if list[i].Days != list[k].Days {
				return list[i].Days > list[k].Days
			}
			return list[i].Weather < list[k].Weather
		})
		if len(list) > 3 {
			list = list[:3]
		}
		out["weather_top"] = list
	}

	// Траектория фаз Луны — уникальные фазы по порядку дат.
	moonSeen := map[string]bool{}
	var moonPath []string
	for _, e := range entries {
		if e.MoonPhase != "" && !moonSeen[e.MoonPhase] {
			moonSeen[e.MoonPhase] = true
			moonPath = append(moonPath, e.MoonPhase)
		}
	}
	if len(moonPath) > 0 {
		out["moon_phases"] = moonPath
	}

	// Частые транзиты: «Sun trine Moon» → сколько дней, минимальный орбис.
	transits := map[string]*transitStat{}
	hasChart := false
	for _, e := range entries {
		if e.Chart != "" {
			hasChart = true
		}
		seenDay := map[string]bool{}
		for _, a := range e.Transits {
			k := fmt.Sprintf("%s %s %s", a.P1, a.Aspect, a.P2)
			if seenDay[k] {
				continue
			}
			seenDay[k] = true
			st, ok := transits[k]
			if !ok {
				st = &transitStat{Transit: k, MinOrbit: a.Orbit}
				transits[k] = st
			}
			st.Days++
			st.MinOrbit = math.Min(st.MinOrbit, a.Orbit)
		}
	}
	if len(transits) > 0 {
		var list []transitStat
		for _, st := range transits {
			list = append(list, *st)
		}
		sort.Slice(list, func(i, k int) bool {
			if list[i].Days != list[k].Days {
				return list[i].Days > list[k].Days
			}
			return list[i].MinOrbit < list[k].MinOrbit
		})
		if len(list) > 5 {
			list = list[:5]
		}
		out["top_transits"] = list
	}

	// Аффирмации.
	var lastAff, lastEmoji string
	for _, e := range entries {
		if e.Affirmation != "" {
			lastAff, lastEmoji = e.Affirmation, e.Emoji
		}
	}
	if lastAff != "" {
		count := 0
		for _, e := range entries {
			if e.Affirmation != "" {
				count++
			}
		}
		out["affirmations"] = map[string]any{
			"count": count,
			"last":  lastAff,
			"emoji": lastEmoji,
		}
	}

	out["reminders"] = summarizeReminders(entries, days, today, hasChart)
	return out
}

// summarizeReminders — детерминированные напоминания из накопленных данных
// (трактовки делает модель поверх фактов).
func summarizeReminders(entries []*JournalEntry, days int, today string, hasChart bool) []string {
	var out []string

	precip := 0
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, e := range entries {
		if e.PrecipProb != nil && *e.PrecipProb >= 50 {
			precip++
		}
		if e.Temp != nil {
			lo = math.Min(lo, *e.Temp)
			hi = math.Max(hi, *e.Temp)
		}
	}
	if precip >= 3 {
		out = append(out, fmt.Sprintf("осадки вероятны в %d из %d дней — держите зонт под рукой", precip, len(entries)))
	}
	if !math.IsInf(lo, 0) && !math.IsInf(hi, 0) && hi-lo >= 10 {
		out = append(out, fmt.Sprintf("перепад температур за период %.0f °C — одевайтесь слоями", hi-lo))
	}
	if len(entries) < days && len(entries) > 0 {
		out = append(out, fmt.Sprintf("журнал неполный: снимков %d из %d дней — проверьте, что планировщик вызывает journal_capture ежедневно", len(entries), days))
	}
	if !hasChart {
		out = append(out, "персональных транзитов нет — постройте натальную карту инструментом natal_build_chart")
	}
	if len(out) == 0 {
		out = append(out, "без ярких событий за период — день под контролем")
	}
	return out
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }

// journalDateRe — дата YYYY-MM-DD.
var journalDateRe = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)

// daysArg — период из аргументов: 1–365, по умолчанию 7.
func daysArg(days *int) int {
	d := 7
	if days != nil {
		d = *days
	}
	if d < 1 {
		d = 1
	}
	if d > 365 {
		d = 365
	}
	return d
}
