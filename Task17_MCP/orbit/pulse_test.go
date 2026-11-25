package orbit

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

// pulseRow — снимок с фиксированным временем.
func pulseRow(ts string, temp float64, sign, transit string, orbit float64) *PulseRow {
	tp, o := temp, orbit
	return &PulseRow{
		Ts: ts, Date: ts[:10], Time: ts[11:16],
		Temp: &tp, MoonSign: sign, Transit: transit, Orbit: &o,
		MoonPhase: "Waxing Gibbous", City: "Казань",
	}
}

// TestPulseUpsertReplace — одна строка на минуту: повтор перезаписывает.
func TestPulseUpsertReplace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pulse.db")
	p, err := NewPulseStore(path)
	if err != nil {
		t.Fatalf("NewPulseStore: %v", err)
	}
	defer p.Close()

	if err := p.Upsert(pulseRow("2026-11-25T10:00", 5, "Psc", "Sun trine Moon", 2.0)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := p.Upsert(pulseRow("2026-11-25T10:00", 6, "Psc", "Sun trine Moon", 1.0)); err != nil {
		t.Fatalf("Upsert повторный: %v", err)
	}
	if n, _ := p.Count(); n != 1 {
		t.Fatalf("ожидалась 1 строка на минуту, получено %d", n)
	}
	rows, err := p.Since(0, time.Date(2026, 11, 25, 10, 1, 0, 0, time.UTC))
	if err != nil || len(rows) != 1 {
		t.Fatalf("Since: %v rows=%d", err, len(rows))
	}
	if *rows[0].Temp != 6 || *rows[0].Orbit != 1.0 {
		t.Fatalf("перезапись не применилась: %+v", rows[0])
	}
}

// TestPulseSinceWindow — окно чтения по минутам.
func TestPulseSinceWindow(t *testing.T) {
	p, err := NewPulseStore(filepath.Join(t.TempDir(), "pulse.db"))
	if err != nil {
		t.Fatalf("NewPulseStore: %v", err)
	}
	defer p.Close()

	for _, ts := range []string{"2026-11-25T09:57", "2026-11-25T09:59", "2026-11-25T10:00", "2026-11-25T10:01"} {
		if err := p.Upsert(pulseRow(ts, 0, "Psc", "", 0)); err != nil {
			t.Fatalf("Upsert: %v", err)
		}
	}
	now := time.Date(2026, 11, 25, 10, 1, 0, 0, time.UTC)
	rows, err := p.Since(3, now) // 09:59–10:01
	if err != nil {
		t.Fatalf("Since: %v", err)
	}
	if len(rows) != 3 || rows[0].Ts != "2026-11-25T09:59" {
		t.Fatalf("окно 3 мин: %+v", rows)
	}
}

// TestSummarizePulse — ингрессы Луны, таймлайн транзитов, температура.
func TestSummarizePulse(t *testing.T) {
	rows := []*PulseRow{
		pulseRow("2026-11-25T10:00", 5, "Psc", "Sun trine Moon", 2.0),
		pulseRow("2026-11-25T10:01", 6, "Psc", "Sun trine Moon", 1.5),
		pulseRow("2026-11-25T10:02", 7, "Ari", "Sun trine Moon", 1.0), // ingress в 10:02
		pulseRow("2026-11-25T10:03", 8, "Ari", "Venus square Mars", 0.8),
	}
	out := SummarizePulse(rows, 240)

	if out["pulses"].(int) != 4 || out["from"].(string) != "2026-11-25T10:00" {
		t.Fatalf("шапка: %+v", out)
	}
	if out["moon_sign_now"].(string) != "Ari" {
		t.Fatalf("текущий знак: %v", out["moon_sign_now"])
	}
	ing := out["moon_ingresses"].([]map[string]any)
	if len(ing) != 1 || ing[0]["sign"] != "Ari" || ing[0]["at"] != "10:02" {
		t.Fatalf("ингрессы: %+v", ing)
	}
	tl := out["transit_timeline"].([]transitSpan)
	if len(tl) != 2 || tl[0].Transit != "Sun trine Moon" || tl[0].Minutes != 3 || tl[0].MinOrbit != 1.0 {
		t.Fatalf("таймлайн: %+v", tl)
	}
	temp := out["temperature"].(map[string]any)
	if temp["min"].(float64) != 5 || temp["max"].(float64) != 8 || temp["avg"].(float64) != 6.5 {
		t.Fatalf("температура: %+v", temp)
	}
}

// TestSummarizePulseEmpty — пустое окно.
func TestSummarizePulseEmpty(t *testing.T) {
	out := SummarizePulse(nil, 60)
	if out["pulses"].(int) != 0 || out["note"] == nil {
		t.Fatalf("пустой пульс: %+v", out)
	}
}

// TestPulseCleanup — ретенция: старые даты удаляются.
func TestPulseCleanup(t *testing.T) {
	p, err := NewPulseStore(filepath.Join(t.TempDir(), "pulse.db"))
	if err != nil {
		t.Fatalf("NewPulseStore: %v", err)
	}
	defer p.Close()

	if err := p.Upsert(pulseRow("2026-10-01T10:00", 0, "Psc", "", 0)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if err := p.Upsert(pulseRow("2026-11-25T10:00", 0, "Psc", "", 0)); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	now := time.Date(2026, 11, 25, 10, 1, 0, 0, time.UTC)
	removed, err := p.Cleanup(30, now)
	if err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if removed != 1 {
		t.Fatalf("ожидалось удалить 1 строку, удалено %d", removed)
	}
	if n, _ := p.Count(); n != 1 {
		t.Fatalf("после ретенции осталось %d строк", n)
	}
}

// TestPulseCaptureAndSummary — сервисный путь по фейкам: снимок → отчёт.
func TestPulseCaptureAndSummary(t *testing.T) {
	f := newFakes(t)
	p, err := NewPulseStore(filepath.Join(t.TempDir(), "pulse.db"))
	if err != nil {
		t.Fatalf("NewPulseStore: %v", err)
	}
	defer p.Close()
	f.svc.WithPulse(p)

	cap, err := f.svc.PulseCapture(context.Background(), PulseArgs{City: "Казань"})
	if err != nil {
		t.Fatalf("PulseCapture: %v", err)
	}
	if cap["saved"] != true {
		t.Fatalf("снимок: %+v", cap)
	}
	if n, _ := p.Count(); n != 1 {
		t.Fatalf("в базе %d строк, ожидалась 1", n)
	}

	sum, err := f.svc.PulseSummary(context.Background(), PulseArgs{})
	if err != nil {
		t.Fatalf("PulseSummary: %v", err)
	}
	if sum["pulses"].(int) != 1 {
		t.Fatalf("summary: %+v", sum)
	}

	read, err := f.svc.PulseRead(context.Background(), PulseArgs{})
	if err != nil {
		t.Fatalf("PulseRead: %v", err)
	}
	if read["count"].(int) != 1 {
		t.Fatalf("read: %+v", read)
	}
}

// TestPulseNotConfigured — пульс не настроен → понятная ошибка.
func TestPulseNotConfigured(t *testing.T) {
	f := newFakes(t)
	if _, err := f.svc.PulseCapture(context.Background(), PulseArgs{City: "Казань"}); err == nil {
		t.Fatal("ожидалась ошибка «пульс не настроен»")
	}
}
