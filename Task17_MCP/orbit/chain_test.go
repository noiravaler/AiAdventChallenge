package orbit

// Тесты цепочек (пайплайнов) elect_* и brief_*: happy path, инъекция сбоев
// через simulate, естественные обрывы (пустой результат первого шага,
// неверный порядок шагов, TTL сессии).

import (
	"context"
	"strings"
	"testing"
	"time"
)

// buildTestChart строит натальную карту через фейковый движок — она нужна
// elect_start'у (натал, к которому считаются транзиты).
func buildTestChart(t *testing.T, f *fakes) {
	t.Helper()
	if _, err := f.svc.BuildNatal(context.Background(), NatalArgs{
		Name: "Anna", BirthDate: "1995-07-21", BirthTime: "14:30", City: "Kazan",
	}); err != nil {
		t.Fatalf("BuildNatal: %v", err)
	}
}

// electRun проводит цепочку целиком и возвращает payload каждого шага.
func electRun(t *testing.T, f *fakes, startArgs ElectStartArgs) []map[string]any {
	t.Helper()
	ctx := context.Background()

	r1, err := f.svc.ElectStart(ctx, startArgs)
	if err != nil {
		t.Fatalf("elect_start: %v", err)
	}
	if r1["next"] != "elect_scan" {
		t.Fatalf("elect_start: next = %v, ожидался elect_scan", r1["next"])
	}
	id := r1["election_id"].(string)

	r2, err := f.svc.ElectScan(ctx, ElectStepArgs{ElectionID: id})
	if err != nil {
		t.Fatalf("elect_scan: %v", err)
	}
	if r2["next"] != "elect_score" {
		t.Fatalf("elect_scan: next = %v, ожидался elect_score", r2["next"])
	}

	r3, err := f.svc.ElectScore(ctx, ElectStepArgs{ElectionID: id})
	if err != nil {
		t.Fatalf("elect_score: %v", err)
	}
	if r3["next"] != "elect_report" {
		t.Fatalf("elect_score: next = %v, ожидался elect_report", r3["next"])
	}

	r4, err := f.svc.ElectReport(ctx, ElectStepArgs{ElectionID: id})
	if err != nil {
		t.Fatalf("elect_report: %v", err)
	}
	if r4["next"] != "" {
		t.Fatalf("elect_report: next = %v, ожидалась пустая строка (конец цепочки)", r4["next"])
	}
	return []map[string]any{r1, r2, r3, r4}
}

// TestElectChainHappyPath — полный цикл подбора даты: статусы, next,
// обращение к движку по каждому дню, отчёт с обоснованием.
func TestElectChainHappyPath(t *testing.T) {
	f := newFakes(t)
	buildTestChart(t, f)

	window := 3
	steps := electRun(t, f, ElectStartArgs{Goal: "запуск блога", City: "Kazan", WindowDays: &window})

	// Скан сходил в движок за транзитами и Луной по каждому дню.
	if got := f.hits("/api/v6/chart-data/transit"); got != window {
		t.Fatalf("запросов транзитов = %d, ожидалось %d", got, window)
	}
	if got := f.hits("/api/v6/moon-phase/context"); got != window {
		t.Fatalf("запросов Луны = %d, ожидалось %d", got, window)
	}

	// Скор: у фейка каждый день — трин с орбисом 1.4° (2+round(1.6)=4)
	// плюс растущая Луна (+1); знак Psc не резонирует с целью «блог».
	details := steps[2]["details"].(map[string]any)
	top := details["top"].([]ElectDayScore)
	if len(top) == 0 || top[0].Score != 5 {
		t.Fatalf("неожиданный скор: %+v", top)
	}
	if len(top[0].Reasons) == 0 {
		t.Fatalf("у топ-дня нет обоснований: %+v", top[0])
	}

	// Отчёт: markdown с целью, датой и обоснованием.
	report := steps[3]["details"].(map[string]any)["report"].(string)
	for _, want := range []string{"запуск блога", "Kazan", top[0].Date, "гармоничный аспект"} {
		if !strings.Contains(report, want) {
			t.Fatalf("в отчёте нет %q:\n%s", want, report)
		}
	}
}

// TestElectScanSimulateError — второй инструмент возвращает ошибку по
// запросу: цепочка обрывается на elect_scan, сессия остаётся в started.
func TestElectScanSimulateError(t *testing.T) {
	f := newFakes(t)
	buildTestChart(t, f)
	ctx := context.Background()

	r1, err := f.svc.ElectStart(ctx, ElectStartArgs{Goal: "запуск блога", City: "Kazan"})
	if err != nil {
		t.Fatalf("elect_start: %v", err)
	}
	id := r1["election_id"].(string)

	if _, err := f.svc.ElectScan(ctx, ElectStepArgs{ElectionID: id, Simulate: "error"}); err == nil {
		t.Fatal("ожидалась ошибка elect_scan (simulate=error)")
	}
	// Сессия жива, но шаг не выполнен: повторный скан без simulate проходит.
	if _, err := f.svc.ElectScan(ctx, ElectStepArgs{ElectionID: id}); err != nil {
		t.Fatalf("повторный elect_scan после simulate: %v", err)
	}
}

// TestElectStartSimulateEmpty — первый инструмент вернул пустой результат:
// цепочка НЕ рвётся сразу — пустота прокатывается через elect_scan и
// обрывает её elect_score (нечего оценивать).
func TestElectStartSimulateEmpty(t *testing.T) {
	f := newFakes(t)
	buildTestChart(t, f)
	ctx := context.Background()

	r1, err := f.svc.ElectStart(ctx, ElectStartArgs{Goal: "запуск блога", City: "Kazan", Simulate: "empty"})
	if err != nil {
		t.Fatalf("elect_start (simulate=empty): %v — ожидался успешный пустой результат", err)
	}
	if details := r1["details"].(map[string]any); len(details) != 0 {
		t.Fatalf("ожидался пустой details, получено %v", details)
	}
	id := r1["election_id"].(string)

	r2, err := f.svc.ElectScan(ctx, ElectStepArgs{ElectionID: id})
	if err != nil {
		t.Fatalf("elect_scan: %v", err)
	}
	if got := r2["details"].(map[string]any)["candidates"]; got != 0 {
		t.Fatalf("кандидатов = %v, ожидалось 0", got)
	}
	if _, err := f.svc.ElectScore(ctx, ElectStepArgs{ElectionID: id}); err == nil {
		t.Fatal("ожидалась ошибка elect_score на пустых кандидатах (обрыв цепочки на шаге 3)")
	}
}

// TestElectOutOfOrder — шаги нельзя вызывать вразнобой: score без scan.
func TestElectOutOfOrder(t *testing.T) {
	f := newFakes(t)
	buildTestChart(t, f)
	ctx := context.Background()

	r1, err := f.svc.ElectStart(ctx, ElectStartArgs{Goal: "запуск блога", City: "Kazan"})
	if err != nil {
		t.Fatalf("elect_start: %v", err)
	}
	if _, err := f.svc.ElectScore(ctx, ElectStepArgs{ElectionID: r1["election_id"].(string)}); err == nil {
		t.Fatal("ожидалась ошибка: elect_score до elect_scan")
	}
}

// TestElectTTL — сессия истекает: шаг после TTL требует начать заново.
func TestElectTTL(t *testing.T) {
	f := newFakes(t)
	buildTestChart(t, f)
	ctx := context.Background()

	r1, err := f.svc.ElectStart(ctx, ElectStartArgs{Goal: "запуск блога", City: "Kazan"})
	if err != nil {
		t.Fatalf("elect_start: %v", err)
	}
	f.svc.Now = func() time.Time { return time.Date(2026, 2, 10, 11, 0, 0, 0, time.UTC) } // +2 часа
	if _, err := f.svc.ElectScan(ctx, ElectStepArgs{ElectionID: r1["election_id"].(string)}); err == nil || !strings.Contains(err.Error(), "истекла") {
		t.Fatalf("ожидалась ошибка истёкшей сессии, получено: %v", err)
	}
}

// TestBriefChain — полный цикл брифинга: collect наполняет секции,
// compose склеивает markdown со всеми секциями.
func TestBriefChain(t *testing.T) {
	f := newFakes(t)
	ctx := context.Background()

	r1, err := f.svc.BriefStart(ctx, BriefStartArgs{City: "Kazan"})
	if err != nil {
		t.Fatalf("brief_start: %v", err)
	}
	id := r1["brief_id"].(string)
	if r1["next"] != "brief_collect" {
		t.Fatalf("brief_start: next = %v", r1["next"])
	}

	r2, err := f.svc.BriefCollect(ctx, BriefStepArgs{BriefID: id})
	if err != nil {
		t.Fatalf("brief_collect: %v", err)
	}
	if got := r2["details"].(map[string]any)["sections"]; got != 4 {
		t.Fatalf("секций собрано = %v, ожидалось 4", got)
	}

	r3, err := f.svc.BriefCompose(ctx, BriefStepArgs{BriefID: id})
	if err != nil {
		t.Fatalf("brief_compose: %v", err)
	}
	if r3["next"] != "" {
		t.Fatalf("brief_compose: next = %v, ожидалась пустая строка", r3["next"])
	}
	brief := r3["details"].(map[string]any)["brief"].(string)
	for _, want := range []string{"Утренний брифинг", "Погода", "Луна", "Транзиты", "Подсказки"} {
		if !strings.Contains(brief, want) {
			t.Fatalf("в брифинге нет %q:\n%s", want, brief)
		}
	}
}

// TestBriefBrokenChain — оба сценария обрыва брифинга: ошибка на collect
// (fail-fast) и пустой collect (обрыв позже, на compose).
func TestBriefBrokenChain(t *testing.T) {
	f := newFakes(t)
	ctx := context.Background()

	// Ошибка на шаге 2: брифинг остаётся в started, collect можно повторить.
	r1, err := f.svc.BriefStart(ctx, BriefStartArgs{City: "Kazan"})
	if err != nil {
		t.Fatalf("brief_start: %v", err)
	}
	id := r1["brief_id"].(string)
	if _, err := f.svc.BriefCollect(ctx, BriefStepArgs{BriefID: id, Simulate: "error"}); err == nil {
		t.Fatal("ожидалась ошибка brief_collect (simulate=error)")
	}
	if _, err := f.svc.BriefCollect(ctx, BriefStepArgs{BriefID: id}); err != nil {
		t.Fatalf("повторный brief_collect после simulate: %v", err)
	}

	// Пустой шаг 2: collect успешен, но секций нет — compose обрывается.
	r1b, err := f.svc.BriefStart(ctx, BriefStartArgs{City: "Kazan"})
	if err != nil {
		t.Fatalf("brief_start: %v", err)
	}
	idb := r1b["brief_id"].(string)
	if _, err := f.svc.BriefCollect(ctx, BriefStepArgs{BriefID: idb, Simulate: "empty"}); err != nil {
		t.Fatalf("brief_collect (simulate=empty): %v", err)
	}
	if _, err := f.svc.BriefCompose(ctx, BriefStepArgs{BriefID: idb}); err == nil {
		t.Fatal("ожидалась ошибка brief_compose: секции не собраны")
	}
}
