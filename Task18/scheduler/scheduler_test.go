package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --------------------------------------------------------------- ParseSchedule

func TestParseSchedule(t *testing.T) {
	cases := []struct {
		spec     string
		interval time.Duration
		times    int
		weekdays int
		wantErr  bool
	}{
		{spec: "@every 30m", interval: 30 * time.Minute},
		{spec: "@every 2h", interval: 2 * time.Hour},
		{spec: "@every 1m", interval: time.Minute},
		{spec: "09:00", times: 1},
		{spec: "09:00,18:30", times: 2},
		{spec: "  9:05  ", times: 1},
		{spec: "[пн,ср,пт] 09:00", times: 1, weekdays: 3},
		{spec: "[mon,fri] 07:15,19:45", times: 2, weekdays: 2},
		{spec: "", wantErr: true},
		{spec: "@every 10s", wantErr: true},      // ниже минимума
		{spec: "@every nonsense", wantErr: true}, // не длительность
		{spec: "@every", wantErr: true},
		{spec: "25:00", wantErr: true},
		{spec: "09:60", wantErr: true},
		{spec: "0900", wantErr: true},
		{spec: "[xyz] 09:00", wantErr: true},
		{spec: "[пн 09:00", wantErr: true}, // не закрыта скобка
	}
	for _, c := range cases {
		s, err := ParseSchedule(c.spec)
		if c.wantErr {
			if err == nil {
				t.Errorf("ParseSchedule(%q): ожидалась ошибка, получено %+v", c.spec, s)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseSchedule(%q): неожиданная ошибка: %v", c.spec, err)
			continue
		}
		if s.Interval != c.interval {
			t.Errorf("ParseSchedule(%q): interval = %v, ожидалось %v", c.spec, s.Interval, c.interval)
		}
		if len(s.Times) != c.times {
			t.Errorf("ParseSchedule(%q): times = %d, ожидалось %d", c.spec, len(s.Times), c.times)
		}
		if len(s.Weekdays) != c.weekdays {
			t.Errorf("ParseSchedule(%q): weekdays = %d, ожидалось %d", c.spec, len(s.Weekdays), c.weekdays)
		}
	}
}

func TestScheduleNextInterval(t *testing.T) {
	s, err := ParseSchedule("@every 30m")
	if err != nil {
		t.Fatal(err)
	}
	base := time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)
	next := s.Next(base, time.Local)
	if !next.Equal(base.Add(30 * time.Minute)) {
		t.Fatalf("Next = %v, ожидалось %v", next, base.Add(30*time.Minute))
	}
}

func TestScheduleNextDailyTimes(t *testing.T) {
	s, err := ParseSchedule("09:00,18:30")
	if err != nil {
		t.Fatal(err)
	}
	// 10:00 → следующий момент 18:30 того же дня.
	base := time.Date(2026, 3, 10, 10, 0, 0, 0, time.Local)
	next := s.Next(base, time.Local)
	want := time.Date(2026, 3, 10, 18, 30, 0, 0, time.Local)
	if !next.Equal(want) {
		t.Fatalf("Next = %v, ожидалось %v", next, want)
	}
	// 19:00 → 09:00 следующего дня.
	base = time.Date(2026, 3, 10, 19, 0, 0, 0, time.Local)
	next = s.Next(base, time.Local)
	want = time.Date(2026, 3, 11, 9, 0, 0, 0, time.Local)
	if !next.Equal(want) {
		t.Fatalf("Next = %v, ожидалось %v", next, want)
	}
	// Ровно в момент срабатывания → следующий день (строго после after).
	base = time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)
	next = s.Next(base, time.Local)
	want = time.Date(2026, 3, 10, 18, 30, 0, 0, time.Local)
	if !next.Equal(want) {
		t.Fatalf("Next = %v, ожидалось %v", next, want)
	}
}

func TestScheduleNextWeekdayFilter(t *testing.T) {
	s, err := ParseSchedule("[пн] 09:00") // только понедельник
	if err != nil {
		t.Fatal(err)
	}
	// 10 марта 2026 — вторник; ближайший понедельник — 16 марта.
	base := time.Date(2026, 3, 10, 12, 0, 0, 0, time.Local)
	next := s.Next(base, time.Local)
	want := time.Date(2026, 3, 16, 9, 0, 0, 0, time.Local)
	if !next.Equal(want) {
		t.Fatalf("Next = %v (%s), ожидалось %v", next, next.Weekday(), want)
	}
}

func TestScheduleNextTimezone(t *testing.T) {
	s, err := ParseSchedule("09:00")
	if err != nil {
		t.Fatal(err)
	}
	ny, err := time.LoadLocation("America/New_York")
	if err != nil {
		t.Skip("нет базы таймзон:", err)
	}
	// 20:00 UTC 10 марта = 16:00 в Нью-Йорке → 09:00 11 марта в Нью-Йорке.
	base := time.Date(2026, 3, 10, 20, 0, 0, 0, time.UTC)
	next := s.Next(base, ny)
	want := time.Date(2026, 3, 11, 9, 0, 0, 0, ny)
	if !next.Equal(want) {
		t.Fatalf("Next = %v, ожидалось %v", next, want)
	}
}

// ------------------------------------------------------------------- engine

// fakeClock — управляемые вручную часы для детерминированных тестов.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// waitFor polls cond until it holds or the deadline expires.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("не дождались: %s", what)
}

func testJob(name, schedule string) Job {
	return Job{Name: name, Prompt: "тестовый промпт " + name, Schedule: schedule, Enabled: true}
}

func TestSchedulerFiresDueJob(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)}
	fired := make(chan Job, 4)
	s := New(func(ctx context.Context, job Job) (string, error) {
		fired <- job
		return "ответ " + job.Name, nil
	}, Options{Tick: 5 * time.Millisecond, Now: clock.Now, Logf: func(string, ...any) {}})

	if _, err := s.Add(testJob("тик", "@every 1m")); err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer s.Stop()
	clock.Advance(time.Minute) // NextRun был «старт + 1м»

	var got Job
	select {
	case got = <-fired:
	case <-time.After(3 * time.Second):
		t.Fatal("задача не сработала")
	}
	if got.Name != "тик" {
		t.Fatalf("имя задачи: %q", got.Name)
	}

	jobs := s.List()
	if len(jobs) != 1 {
		t.Fatalf("jobs = %d", len(jobs))
	}
	j := jobs[0]
	if !j.LastOK || j.LastError != "" {
		t.Fatalf("LastOK=%v LastError=%q", j.LastOK, j.LastError)
	}
	if j.Running {
		t.Fatal("задача помечена как выполняющаяся после завершения")
	}
	// NextRun перевзведён от момента завершения: ~старт+1м.
	if !j.NextRun.After(clock.Now()) {
		t.Fatalf("NextRun = %v не в будущем (now=%v)", j.NextRun, clock.Now())
	}
	res := s.Results(j.ID)
	if len(res) != 1 || !res[0].OK || res[0].Answer != "ответ тик" {
		t.Fatalf("результаты: %+v", res)
	}
}

func TestSchedulerSkipsMissedAndRecomputesOnLoad(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)}
	s := New(nil, Options{Now: clock.Now, Logf: func(string, ...any) {}})

	// Ежедневная задача на 09:00: добавляем ровно в 09:00 → NextRun = завтра.
	j, err := s.Add(testJob("утро", "09:00"))
	if err != nil {
		t.Fatal(err)
	}
	want := time.Date(2026, 3, 11, 9, 0, 0, 0, time.Local)
	if !j.NextRun.Equal(want) {
		t.Fatalf("NextRun = %v, ожидалось %v", j.NextRun, want)
	}

	// Перезагрузка «после пропуска» (часы перескочили на 3 дня): запуск не
	// догоняется, NextRun переносится на ближайший будущий момент.
	clock.Advance(72 * time.Hour)
	db := filepath.Join(t.TempDir(), "sched.db")
	s = New(nil, Options{DB: db, Now: clock.Now, Logf: func(string, ...any) {}})
	if _, err := s.Add(Job{ID: j.ID, Name: j.Name, Prompt: j.Prompt, Schedule: j.Schedule, Enabled: true}); err != nil {
		t.Fatal(err)
	}
	s2 := New(nil, Options{DB: db, Now: clock.Now, Logf: func(string, ...any) {}})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	got := s2.List()
	if len(got) != 1 {
		t.Fatalf("после загрузки задач: %d", len(got))
	}
	if !got[0].NextRun.After(clock.Now()) {
		t.Fatalf("NextRun = %v должен быть в будущем (now=%v)", got[0].NextRun, clock.Now())
	}
}

func TestSchedulerNoConcurrentRunsOfSameJob(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)}
	var active, maxActive int64
	release := make(chan struct{})
	s := New(func(ctx context.Context, job Job) (string, error) {
		cur := atomic.AddInt64(&active, 1)
		for {
			old := atomic.LoadInt64(&maxActive)
			if cur <= old || atomic.CompareAndSwapInt64(&maxActive, old, cur) {
				break
			}
		}
		<-release
		atomic.AddInt64(&active, -1)
		return "ok", nil
	}, Options{Tick: 5 * time.Millisecond, Now: clock.Now, Logf: func(string, ...any) {}})

	if _, err := s.Add(testJob("долгая", "@every 1m")); err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer s.Stop()

	// Переводим часы за срок задачи и тикаем несколько раз, пока она ещё
	// выполняется: параллельного запуска быть не должно.
	clock.Advance(2 * time.Minute)
	s.fireDue()
	for i := 0; i < 20; i++ {
		time.Sleep(time.Millisecond)
		s.fireDue()
	}
	close(release)
	waitFor(t, "завершения задачи", func() bool { return atomic.LoadInt64(&active) == 0 })

	if m := atomic.LoadInt64(&maxActive); m != 1 {
		t.Fatalf("максимум параллельных запусков = %d, ожидался 1", m)
	}
	j := s.List()[0]
	if j.Running {
		t.Fatal("Running не сброшен")
	}
	if len(s.Results(j.ID)) != 1 {
		t.Fatalf("результатов: %d, ожидался 1", len(s.Results(j.ID)))
	}
}

func TestSchedulerRunNowDoesNotReschedule(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)}
	fired := make(chan struct{}, 4)
	s := New(func(ctx context.Context, job Job) (string, error) {
		fired <- struct{}{}
		return "ok", nil
	}, Options{Now: clock.Now, Logf: func(string, ...any) {}})

	j, err := s.Add(testJob("ручная", "09:00")) // NextRun = завтра 09:00
	if err != nil {
		t.Fatal(err)
	}
	before := j.NextRun

	if err := s.RunNow(j.ID); err != nil {
		t.Fatal(err)
	}
	select {
	case <-fired:
	case <-time.After(3 * time.Second):
		t.Fatal("ручной запуск не сработал")
	}
	waitFor(t, "завершения ручного запуска", func() bool {
		jobs := s.List()
		return len(jobs) == 1 && !jobs[0].Running && len(s.Results(j.ID)) == 1
	})
	after := s.List()[0]
	if !after.NextRun.Equal(before) {
		t.Fatalf("RunNow сдвинул NextRun: было %v, стало %v", before, after.NextRun)
	}
	if after.LastRun.IsZero() {
		t.Fatal("LastRun не проставлен")
	}

	// Повторный ручной запуск во время выполнения → ошибка.
	if err := s.RunNow(j.ID); err != nil {
		t.Fatalf("RunNow после завершения: %v", err)
	}
	<-fired
	if err := s.RunNow("no-such-id"); err == nil {
		t.Fatal("ожидалась ошибка для неизвестного id")
	}
}

func TestSchedulerToggleDisabledJobNotFired(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)}
	var count int64
	s := New(func(ctx context.Context, job Job) (string, error) {
		atomic.AddInt64(&count, 1)
		return "ok", nil
	}, Options{Tick: 5 * time.Millisecond, Now: clock.Now, Logf: func(string, ...any) {}})

	j, err := s.Add(testJob("выкл", "@every 1m"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Toggle(j.ID, false); err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer s.Stop()
	clock.Advance(2 * time.Hour)
	s.fireDue()
	time.Sleep(20 * time.Millisecond)
	if n := atomic.LoadInt64(&count); n != 0 {
		t.Fatalf("выключенная задача сработала %d раз", n)
	}
	// Включение перевзводит NextRun в будущее.
	after, err := s.Toggle(j.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	if !after.NextRun.After(clock.Now()) {
		t.Fatalf("после включения NextRun = %v не в будущем", after.NextRun)
	}
	clock.Advance(time.Minute)
	s.fireDue()
	waitFor(t, "запуска после включения", func() bool { return atomic.LoadInt64(&count) == 1 })
}

func TestSchedulerJobErrorsRecorded(t *testing.T) {
	s := New(func(ctx context.Context, job Job) (string, error) {
		return "", fmt.Errorf("API недоступен")
	}, Options{Now: func() time.Time { return time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local) }, Logf: func(string, ...any) {}})

	j, err := s.Add(testJob("падает", "@every 1h"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RunNow(j.ID); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "завершения падения", func() bool {
		r := s.Results(j.ID)
		return len(r) == 1 && !r[0].OK
	})
	jobs := s.List()
	if jobs[0].LastOK {
		t.Fatal("LastOK должен быть false")
	}
	if jobs[0].LastError == "" {
		t.Fatal("LastError пуст")
	}
}

func TestSchedulerPersistenceRoundTrip(t *testing.T) {
	db := filepath.Join(t.TempDir(), "sched.db")
	s := New(nil, Options{DB: db, Logf: func(string, ...any) {}})
	j1, err := s.Add(testJob("один", "@every 1h"))
	if err != nil {
		t.Fatal(err)
	}
	j2, err := s.Add(Job{Name: "два", Tool: "orbit__pulse_capture", Schedule: "09:00", Enabled: false})
	if err != nil {
		t.Fatal(err)
	}

	s2 := New(nil, Options{DB: db, Logf: func(string, ...any) {}})
	if err := s2.Load(); err != nil {
		t.Fatal(err)
	}
	jobs := s2.List()
	if len(jobs) != 2 {
		t.Fatalf("задач после загрузки: %d", len(jobs))
	}
	if jobs[0].ID != j1.ID || jobs[0].Name != j1.Name || jobs[0].Schedule != "@every 1h" || !jobs[0].Enabled {
		t.Errorf("задача 1: %+v", jobs[0])
	}
	if jobs[1].ID != j2.ID || jobs[1].Enabled || jobs[1].Tool != "orbit__pulse_capture" {
		t.Errorf("задача 2 (tool-режим должен пережить базу): %+v", jobs[1])
	}
	if jobs[1].NextRun.Compare(time.Time{}) != 0 {
		t.Errorf("у выключенной задачи NextRun должен быть нулевым: %v", jobs[1].NextRun)
	}
	if jobs[0].NextRun.IsZero() {
		t.Errorf("у включённой задачи NextRun должен быть проставлен")
	}

	// Журнал результатов тоже живёт в базе: запуск (runner nil → ошибка)
	// переживает переоткрытие планировщика.
	if err := s2.RunNow(j1.ID); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "результат в базе", func() bool { return len(s2.Results(j1.ID)) > 0 })
	s2r := New(nil, Options{DB: db, Logf: func(string, ...any) {}})
	if err := s2r.Load(); err != nil {
		t.Fatal(err)
	}
	ring := s2r.Results(j1.ID)
	if len(ring) != 1 || ring[0].OK || ring[0].Error == "" {
		t.Fatalf("журнал после перезагрузки: %+v", ring)
	}

	// Remove сохраняется.
	if err := s2r.Remove(j1.ID); err != nil {
		t.Fatal(err)
	}
	s3 := New(nil, Options{DB: db, Logf: func(string, ...any) {}})
	if err := s3.Load(); err != nil {
		t.Fatal(err)
	}
	if got := s3.List(); len(got) != 1 || got[0].ID != j2.ID {
		t.Fatalf("после удаления: %+v", got)
	}
	if got := s3.Results(j1.ID); len(got) != 0 {
		t.Fatalf("журнал удалённой задачи должен быть вычищен: %+v", got)
	}
}

func TestSchedulerValidation(t *testing.T) {
	s := New(nil, Options{})
	if _, err := s.Add(Job{Name: "", Prompt: "p", Schedule: "@every 1h"}); err == nil {
		t.Error("ожидалась ошибка: пустое имя")
	}
	if _, err := s.Add(Job{Name: "n", Prompt: "", Schedule: "@every 1h"}); err == nil {
		t.Error("ожидалась ошибка: нет ни промпта, ни инструмента")
	}
	// Tool-режим: промпт не нужен — задача зовёт MCP-инструмент напрямую.
	if _, err := s.Add(Job{Name: "n", Tool: "orbit__pulse_capture", Schedule: "@every 1h"}); err != nil {
		t.Errorf("tool-режим без промпта должен проходить валидацию: %v", err)
	}
	if _, err := s.Add(Job{Name: "n", Prompt: "p", Schedule: "мусор"}); err == nil {
		t.Error("ожидалась ошибка: расписание")
	}
	if _, err := s.Add(Job{Name: "n", Prompt: "p", Schedule: "@every 1h", TZ: "Марс/Олимп"}); err == nil {
		t.Error("ожидалась ошибка: таймзона")
	}
	if err := s.Remove("нет"); err == nil {
		t.Error("ожидалась ошибка: неизвестный id")
	}
}

func TestSchedulerUpdateKeepsResults(t *testing.T) {
	s := New(func(ctx context.Context, job Job) (string, error) { return "ок", nil },
		Options{Logf: func(string, ...any) {}})
	j, err := s.Add(testJob("задача", "@every 1h"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RunNow(j.ID); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "первого запуска", func() bool { return len(s.Results(j.ID)) == 1 })

	updated, err := s.Update(Job{ID: j.ID, Name: "задача v2", Prompt: "новый", Schedule: "@every 2h", Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	if updated.Schedule != "@every 2h" || updated.Name != "задача v2" {
		t.Fatalf("обновление не применилось: %+v", updated)
	}
	if len(s.Results(j.ID)) != 1 {
		t.Fatal("результаты потеряны при обновлении")
	}
}

func TestSchedulerResultRingLimit(t *testing.T) {
	var n int64
	s := New(func(ctx context.Context, job Job) (string, error) {
		return fmt.Sprintf("ответ %d", atomic.AddInt64(&n, 1)), nil
	}, Options{MaxResults: 3, Logf: func(string, ...any) {}})
	j, err := s.Add(testJob("кольцо", "@every 1h"))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if err := s.RunNow(j.ID); err != nil {
			t.Fatal(err)
		}
		waitFor(t, fmt.Sprintf("запуска %d", i+1), func() bool { return int(atomic.LoadInt64(&n)) == i+1 })
	}
	// Кольцо держит последние три записи, по порядку.
	waitFor(t, "обрезки кольца", func() bool { return len(s.Results(j.ID)) == 3 })
	res := s.Results(j.ID)
	if res[0].Answer != "ответ 3" || res[2].Answer != "ответ 5" {
		t.Fatalf("кольцо: %+v", res)
	}
}

func TestSchedulerAnswerTruncated(t *testing.T) {
	s := New(func(ctx context.Context, job Job) (string, error) {
		return "абвгдежзик", nil
	}, Options{MaxAnswer: 5, Logf: func(string, ...any) {}})
	j, err := s.Add(testJob("обрезка", "@every 1h"))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RunNow(j.ID); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "результата", func() bool { return len(s.Results(j.ID)) == 1 })
	res := s.Results(j.ID)[0]
	if res.Answer != "абвгд…" {
		t.Fatalf("answer = %q, ожидался %q", res.Answer, "абвгд…")
	}
}

func TestJobJSONRoundTrip(t *testing.T) {
	now := time.Date(2026, 3, 10, 9, 0, 0, 0, time.Local)
	j := Job{ID: "x", Name: "имя", Prompt: "промпт", Tool: "orbit__pulse_capture", Schedule: "@every 1h", Enabled: true, NextRun: now}
	b, err := json.Marshal(j)
	if err != nil {
		t.Fatal(err)
	}
	var back Job
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back != j {
		t.Fatalf("round trip: %+v != %+v", back, j)
	}
}
