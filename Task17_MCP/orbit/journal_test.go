package orbit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"ai-chat-go/affirmations"
)

// testEntry — снимок дня с указателями на значения.
func testEntry(date string, temp float64, precip int) *JournalEntry {
	t := temp
	tmax, tmin := temp+2, temp-3
	p := precip
	return &JournalEntry{
		Date:       date,
		City:       "Казань",
		CapturedAt: time.Now(),
		Temp:       &t, TempMax: &tmax, TempMin: &tmin,
		PrecipProb: &p,
		Weather:    "ясно",
		MoonPhase:  "полнолуние",
		MoonSign:   "Psc",
	}
}

// TestJournalUpsertPersist — одна запись на дату, персистентность на диск.
func TestJournalUpsertPersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.json")
	j, err := NewJournal(path)
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	if err := j.Upsert(testEntry("2026-11-20", 5, 10)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	// Перезапись той же даты.
	e2 := testEntry("2026-11-20", 7, 60)
	if err := j.Upsert(e2); err != nil {
		t.Fatalf("Upsert повторный: %v", err)
	}
	if err := j.Upsert(testEntry("2026-11-21", 3, 80)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if got := j.Count(); got != 2 {
		t.Fatalf("ожидалось 2 записи (одна на дату), получено %d", got)
	}

	// Перезагрузка с диска.
	j2, err := NewJournal(path)
	if err != nil {
		t.Fatalf("перезагрузка: %v", err)
	}
	entries := j2.Since(0, "")
	if len(entries) != 2 || entries[0].Date != "2026-11-20" || entries[1].Date != "2026-11-21" {
		t.Fatalf("неожиданные записи после перезагрузки: %+v", entries)
	}
	if *entries[0].Temp != 7 {
		t.Fatalf("перезапись за дату не применилась: temp=%v", *entries[0].Temp)
	}

	// Некорректная дата отвергается.
	if err := j2.Upsert(&JournalEntry{Date: "20.11.2026"}); err == nil {
		t.Fatal("запись с некорректной датой должна давать ошибку")
	}
}

// TestJournalSince — фильтр по окну дней.
func TestJournalSince(t *testing.T) {
	j, err := NewJournal(filepath.Join(t.TempDir(), "journal.json"))
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	for _, d := range []string{"2026-11-01", "2026-11-19", "2026-11-24", "2026-11-25"} {
		if err := j.Upsert(testEntry(d, 0, 0)); err != nil {
			t.Fatalf("Upsert: %v", err)
		}
	}
	got := j.Since(3, "2026-11-25")
	if len(got) != 2 || got[0].Date != "2026-11-24" || got[1].Date != "2026-11-25" {
		t.Fatalf("окно 3 дня: ожидались 24–25, получено %+v", dates(got))
	}
	got = j.Since(0, "2026-11-25")
	if len(got) != 4 {
		t.Fatalf("days=0 должен отдавать все записи, получено %d", len(got))
	}
}

func dates(entries []*JournalEntry) []string {
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.Date)
	}
	return out
}

// TestSummarize — агрегаты: температура, осадки, Луна, транзиты, напоминания.
func TestSummarize(t *testing.T) {
	e1 := testEntry("2026-11-24", 5, 70) // осадки
	e1.MoonPhase = "полнолуние"
	e1.Chart = "Anna"
	e1.Transits = []Aspect{
		{P1: "Sun", P2: "Moon", Aspect: "trine", Orbit: 1.0},
		{P1: "Venus", P2: "Mars", Aspect: "square", Orbit: 3.0},
	}
	e2 := testEntry("2026-11-25", 15, 10)
	e2.Weather = "дождь"
	e2.MoonPhase = "убывающая Луна"
	e2.Chart = "Anna"
	e2.Transits = []Aspect{
		{P1: "Sun", P2: "Moon", Aspect: "trine", Orbit: 2.0},
		{P1: "Mercury", P2: "Jupiter", Aspect: "sextile", Orbit: 0.5},
	}
	e2.Affirmation, e2.Emoji = "Ты справишься", "✨"

	out := Summarize([]*JournalEntry{e1, e2}, 7, "2026-11-25")

	if out["days_with_data"].(int) != 2 {
		t.Fatalf("days_with_data: %v", out["days_with_data"])
	}
	temp := out["temperature"].(map[string]any)
	if temp["min"].(float64) != 5 || temp["max"].(float64) != 15 || temp["avg"].(float64) != 10 {
		t.Fatalf("температура: %+v", temp)
	}
	pr := out["precipitation"].(map[string]any)
	if pr["days_with_precip"].(int) != 1 || pr["max_probability"].(int) != 70 {
		t.Fatalf("осадки: %+v", pr)
	}
	if len(out["moon_phases"].([]string)) != 2 {
		t.Fatalf("фазы Луны: %v", out["moon_phases"])
	}
	tt := out["top_transits"].([]transitStat)
	if len(tt) != 3 || tt[0].Transit != "Sun trine Moon" || tt[0].Days != 2 {
		t.Fatalf("топ-транзиты: %+v", tt)
	}
	aff := out["affirmations"].(map[string]any)
	if aff["count"].(int) != 1 || aff["last"].(string) != "Ты справишься" {
		t.Fatalf("аффирмации: %+v", aff)
	}
	rems := out["reminders"].([]string)
	if len(rems) == 0 {
		t.Fatal("напоминания пустые")
	}
}

// TestSummarizeEmpty — пустой журнал.
func TestSummarizeEmpty(t *testing.T) {
	out := Summarize(nil, 7, "2026-11-25")
	if out["days_with_data"].(int) != 0 || out["note"] == nil {
		t.Fatalf("пустой журнал: %+v", out)
	}
}

// TestJournalCaptureAndSummary — полный путь по фейкам: снимок → отчёт.
func TestJournalCaptureAndSummary(t *testing.T) {
	f := newFakes(t)

	// Фейковые аффирмации + перевод, чтобы тест не ходил в интернет.
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"affirmation":"You are capable of amazing things"}`))
	}))
	defer src.Close()
	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[[["Ты способен на удивительные вещи","You are capable of amazing things",null,null,10]],null,"en"]`))
	}))
	defer tr.Close()
	f.svc.Affirm = affirmations.New(src.URL, tr.URL+"/translate_a/single", time.Second)

	j, err := NewJournal(filepath.Join(t.TempDir(), "journal.json"))
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	f.svc.WithJournal(j)

	cap, err := f.svc.JournalCapture(context.Background(), JournalArgs{City: "Казань"})
	if err != nil {
		t.Fatalf("JournalCapture: %v", err)
	}
	if cap["saved"] != true || cap["affirmation"] != "Ты способен на удивительные вещи" {
		t.Fatalf("снимок: %+v", cap)
	}
	if j.Count() != 1 {
		t.Fatalf("в журнале %d записей, ожидалась 1", j.Count())
	}

	// Файл на диске существует и читается.
	raw, err := os.ReadFile(j.path)
	if err != nil {
		t.Fatalf("файл дневника не записан: %v", err)
	}
	var stored journalFile
	if err := json.Unmarshal(raw, &stored); err != nil || len(stored.Entries) != 1 {
		t.Fatalf("битый файл дневника: %v", err)
	}

	sum, err := f.svc.JournalSummary(context.Background(), JournalArgs{})
	if err != nil {
		t.Fatalf("JournalSummary: %v", err)
	}
	if sum["days_with_data"].(int) != 1 {
		t.Fatalf("summary: %+v", sum)
	}
	if _, ok := sum["moon_phases"]; !ok {
		t.Fatalf("в summary нет фаз Луны: %+v", sum)
	}

	read, err := f.svc.JournalRead(context.Background(), JournalArgs{})
	if err != nil {
		t.Fatalf("JournalRead: %v", err)
	}
	if read["count"].(int) != 1 {
		t.Fatalf("read: %+v", read)
	}
}

// TestJournalNotConfigured — журнал не настроен → понятная ошибка.
func TestJournalNotConfigured(t *testing.T) {
	f := newFakes(t)
	if _, err := f.svc.JournalCapture(context.Background(), JournalArgs{City: "Казань"}); err == nil {
		t.Fatal("ожидалась ошибка «журнал не настроен»")
	}
}
