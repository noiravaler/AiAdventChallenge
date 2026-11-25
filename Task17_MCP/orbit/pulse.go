package orbit

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	_ "modernc.org/sqlite" // pure-Go драйвер SQLite (без cgo)
)

// PulseRow — минутный снимок неба: что происходит «прямо сейчас».
// Все изменчивые поля — строками из движка/погоды, числа — указатели.
type PulseRow struct {
	Ts        string   `json:"ts"`   // YYYY-MM-DDTHH:MM (локальная минута, первичный ключ)
	Date      string   `json:"date"` // YYYY-MM-DD
	Time      string   `json:"time"` // HH:MM
	City      string   `json:"city,omitempty"`
	Temp      *float64 `json:"temp,omitempty"` // °C (кэш погоды ~30 мин)
	MoonPhase string   `json:"moon_phase,omitempty"`
	MoonSign  string   `json:"moon_sign,omitempty"`
	Chart     string   `json:"chart,omitempty"`   // карта, к которой считался транзит
	Transit   string   `json:"transit,omitempty"` // самый тесный: "Sun trine Moon"
	Orbit     *float64 `json:"orbit,omitempty"`   // его орбис
}

// PulseStore — накопитель пульса в SQLite: одна строка на минуту,
// повторный снимок за ту же минуту перезаписывает её (INSERT OR REPLACE).
type PulseStore struct {
	db *sql.DB
}

// pulseSchema — таблица + индекс по дате (для ретенции и выборок за день).
const pulseSchema = `
CREATE TABLE IF NOT EXISTS pulse (
	ts         TEXT PRIMARY KEY,
	date       TEXT NOT NULL,
	time       TEXT NOT NULL,
	city       TEXT,
	temp       REAL,
	moon_phase TEXT,
	moon_sign  TEXT,
	chart      TEXT,
	transit    TEXT,
	orbit      REAL
);
CREATE INDEX IF NOT EXISTS idx_pulse_date ON pulse(date);
`

// NewPulseStore открывает/создаёт SQLite-базу пульса.
func NewPulseStore(path string) (*PulseStore, error) {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// Один коннект: единственный писатель, нет SQLITE_BUSY; для минутного
	// темпа (запись < 1 мс) этого достаточно с большим запасом.
	db.SetMaxOpenConns(1)
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL",
		"PRAGMA busy_timeout=3000",
		"PRAGMA synchronous=NORMAL",
	} {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			return nil, fmt.Errorf("pragma %q: %w", pragma, err)
		}
	}
	if _, err := db.Exec(pulseSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("схема пульса: %w", err)
	}
	return &PulseStore{db: db}, nil
}

// Close закрывает базу.
func (p *PulseStore) Close() error { return p.db.Close() }

// Upsert сохраняет снимок минуты (перезапись при повторе).
func (p *PulseStore) Upsert(r *PulseRow) error {
	if r == nil || r.Ts == "" {
		return fmt.Errorf("снимок пульса без ts не может быть сохранён")
	}
	_, err := p.db.Exec(
		`INSERT INTO pulse (ts, date, time, city, temp, moon_phase, moon_sign, chart, transit, orbit)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(ts) DO UPDATE SET
		   date=excluded.date, time=excluded.time, city=excluded.city, temp=excluded.temp,
		   moon_phase=excluded.moon_phase, moon_sign=excluded.moon_sign, chart=excluded.chart,
		   transit=excluded.transit, orbit=excluded.orbit`,
		r.Ts, r.Date, r.Time, r.City, r.Temp, r.MoonPhase, r.MoonSign, r.Chart, r.Transit, r.Orbit)
	return err
}

// Since возвращает снимки за последние minutes минут (включая текущую),
// отсортированные по времени. minutes <= 0 — последние 60.
func (p *PulseStore) Since(minutes int, now time.Time) ([]*PulseRow, error) {
	if minutes <= 0 {
		minutes = 60
	}
	cutoff := now.Add(-time.Duration(minutes) * time.Minute).Format("2006-01-02T15:04")
	rows, err := p.db.Query(
		`SELECT ts, date, time, city, temp, moon_phase, moon_sign, chart, transit, orbit
		 FROM pulse WHERE ts >= ? ORDER BY ts`, cutoff)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanPulse(rows)
}

// Count — всего снимков в базе.
func (p *PulseStore) Count() (int, error) {
	var n int
	err := p.db.QueryRow(`SELECT COUNT(*) FROM pulse`).Scan(&n)
	return n, err
}

// Cleanup удаляет снимки старше days дней, возвращает число удалённых.
func (p *PulseStore) Cleanup(days int, now time.Time) (int64, error) {
	if days <= 0 {
		days = 30
	}
	cutoff := now.AddDate(0, 0, -days).Format("2006-01-02")
	res, err := p.db.Exec(`DELETE FROM pulse WHERE date < ?`, cutoff)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return n, nil
}

func scanPulse(rows *sql.Rows) ([]*PulseRow, error) {
	var out []*PulseRow
	for rows.Next() {
		r := &PulseRow{}
		var temp, orbit sql.NullFloat64
		var city, phase, sign, chart, transit sql.NullString
		if err := rows.Scan(&r.Ts, &r.Date, &r.Time, &city, &temp, &phase, &sign, &chart, &transit, &orbit); err != nil {
			return nil, err
		}
		r.City, r.MoonPhase, r.MoonSign, r.Chart, r.Transit =
			city.String, phase.String, sign.String, chart.String, transit.String
		if temp.Valid {
			v := temp.Float64
			r.Temp = &v
		}
		if orbit.Valid {
			v := orbit.Float64
			r.Orbit = &v
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// minutesArg — окно чтения пульса: 1–1440 минут, по умолчанию 60.
func minutesArg(m *int) int {
	v := 60
	if m != nil {
		v = *m
	}
	if v < 1 {
		v = 1
	}
	if v > 1440 {
		v = 1440
	}
	return v
}

// hoursArg — окно отчёта пульса: 1–720 часов, по умолчанию 24.
func hoursArg(h *int) int {
	v := 24
	if h != nil {
		v = *h
	}
	if v < 1 {
		v = 1
	}
	if v > 720 {
		v = 720
	}
	return v
}

// transitSpan — агрегат по одному транзиту в окне пульса.
type transitSpan struct {
	Transit   string  `json:"transit"`
	Minutes   int     `json:"minutes"`
	FirstSeen string  `json:"first_seen"`
	LastSeen  string  `json:"last_seen"`
	MinOrbit  float64 `json:"min_orbit"`
}

// SummarizePulse — агрегированный отчёт по снимкам пульса: события смены
// лунного знака, таймлайн транзитов, температурная кривая.
func SummarizePulse(rows []*PulseRow, minutes int) map[string]any {
	out := map[string]any{
		"minutes": minutes,
		"pulses":  len(rows),
	}
	if len(rows) == 0 {
		out["note"] = "снимков нет — пульс собирает pulse_capture (по расписанию, например каждую минуту)"
		return out
	}
	out["from"], out["to"] = rows[0].Ts, rows[len(rows)-1].Ts

	// Смена лунного знака — ingress: знак X вступил в силу в HH:MM.
	var ingresses []map[string]any
	current := ""
	for _, r := range rows {
		if r.MoonSign == "" || r.MoonSign == current {
			continue
		}
		if current != "" {
			ingresses = append(ingresses, map[string]any{"sign": r.MoonSign, "at": r.Time})
		}
		current = r.MoonSign
	}
	if current != "" {
		out["moon_sign_now"] = current
	}
	if len(ingresses) > 0 {
		out["moon_ingresses"] = ingresses
	}

	// Фазы Луны по порядку наблюдения.
	phaseSeen := map[string]bool{}
	var phases []string
	for _, r := range rows {
		if r.MoonPhase != "" && !phaseSeen[r.MoonPhase] {
			phaseSeen[r.MoonPhase] = true
			phases = append(phases, r.MoonPhase)
		}
	}
	if len(phases) > 0 {
		out["moon_phases"] = phases
	}

	// Таймлайн транзитов: сколько минут виден, когда появился/ушёл.
	spans := map[string]*transitSpan{}
	for _, r := range rows {
		if r.Transit == "" {
			continue
		}
		sp, ok := spans[r.Transit]
		if !ok {
			sp = &transitSpan{Transit: r.Transit, FirstSeen: r.Time, LastSeen: r.Time, MinOrbit: 1e9}
			if r.Orbit != nil {
				sp.MinOrbit = *r.Orbit
			}
			spans[r.Transit] = sp
		}
		sp.Minutes++
		sp.LastSeen = r.Time
		if r.Orbit != nil && *r.Orbit < sp.MinOrbit {
			sp.MinOrbit = *r.Orbit
		}
	}
	if len(spans) > 0 {
		list := make([]transitSpan, 0, len(spans))
		for _, sp := range spans {
			list = append(list, *sp)
		}
		sort.Slice(list, func(i, j int) bool {
			if list[i].Minutes != list[j].Minutes {
				return list[i].Minutes > list[j].Minutes
			}
			return list[i].MinOrbit < list[j].MinOrbit
		})
		if len(list) > 8 {
			list = list[:8]
		}
		out["transit_timeline"] = list
	}

	// Температура.
	sum, n := 0.0, 0
	lo, hi := 0.0, 0.0
	for _, r := range rows {
		if r.Temp == nil {
			continue
		}
		if n == 0 {
			lo, hi = *r.Temp, *r.Temp
		}
		sum += *r.Temp
		n++
		lo = min(lo, *r.Temp)
		hi = max(hi, *r.Temp)
	}
	if n > 0 {
		out["temperature"] = map[string]any{
			"avg": round1(sum / float64(n)),
			"min": round1(lo),
			"max": round1(hi),
		}
	}
	return out
}
