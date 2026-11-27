package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-chat-go/scheduler"
)

// newTestScheduler installs a scheduler with an instant fake runner and
// restores the default on cleanup.
func newTestScheduler(t *testing.T) *scheduler.Scheduler {
	t.Helper()
	s := scheduler.New(func(ctx context.Context, job scheduler.Job) (string, error) {
		return "ответ " + job.Name, nil
	}, scheduler.Options{Logf: func(string, ...any) {}})
	scheduler.Install(s)
	t.Cleanup(func() { scheduler.Install(nil) })
	return s
}

// TestSchedulerEndpoints — CRUD и результаты через HTTP API планировщика.
func TestSchedulerEndpoints(t *testing.T) {
	newTestScheduler(t)
	ts := newTestServer(t, "http://127.0.0.1:1/none")

	// Пусто на старте.
	jobs := getJobs(t, ts)
	if len(jobs) != 0 {
		t.Fatalf("на старте задач: %d", len(jobs))
	}

	// Добавление.
	resp, err := http.Post(ts.URL+"/api/scheduler/add", "application/json",
		strings.NewReader(`{"name":"брифинг","prompt":"собери брифинг","schedule":"@every 1h","enabled":true}`))
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("add: %d %s", resp.StatusCode, body)
	}
	jobs = decodeJobs(t, body)
	if len(jobs) != 1 || jobs[0].Name != "брифинг" || jobs[0].ID == "" {
		t.Fatalf("после add: %+v", jobs)
	}
	id := jobs[0].ID

	// Некорректное расписание → 400.
	resp, _ = http.Post(ts.URL+"/api/scheduler/add", "application/json",
		strings.NewReader(`{"name":"плохая","prompt":"p","schedule":"мусор","enabled":true}`))
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("add с мусорным расписанием: %d", resp.StatusCode)
	}
	readAll(t, resp)

	// Тумблер.
	resp, _ = http.Post(ts.URL+"/api/scheduler/toggle", "application/json",
		strings.NewReader(`{"id":"`+id+`","enabled":false}`))
	body = readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("toggle: %d %s", resp.StatusCode, body)
	}
	if jobs = decodeJobs(t, body); len(jobs) != 1 || jobs[0].Enabled {
		t.Fatalf("после toggle: %+v", jobs)
	}

	// Ручной запуск (выключенной задачи тоже можно).
	resp, _ = http.Post(ts.URL+"/api/scheduler/run", "application/json",
		strings.NewReader(`{"id":"`+id+`"}`))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("run: %d %s", resp.StatusCode, readAll(t, resp))
	}
	readAll(t, resp)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		res := getResults(t, ts, id)
		if len(res) == 1 && res[0].OK && res[0].Answer == "ответ брифинг" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	res := getResults(t, ts, id)
	if len(res) != 1 || !res[0].OK || res[0].Answer != "ответ брифинг" {
		t.Fatalf("результаты: %+v", res)
	}

	// Неизвестный id → пустой список результатов.
	if res := getResults(t, ts, "nope"); len(res) != 0 {
		t.Fatalf("результаты неизвестной задачи: %+v", res)
	}

	// Удаление.
	resp, _ = http.Post(ts.URL+"/api/scheduler/remove", "application/json",
		strings.NewReader(`{"id":"`+id+`"}`))
	body = readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("remove: %d %s", resp.StatusCode, body)
	}
	if jobs = decodeJobs(t, body); len(jobs) != 0 {
		t.Fatalf("после remove: %+v", jobs)
	}

	// Установка доступна и после Remove: повторный запуск неизвестной задачи → 409/400.
	resp, _ = http.Post(ts.URL+"/api/scheduler/run", "application/json",
		strings.NewReader(`{"id":"`+id+`"}`))
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("run несуществующей: %d", resp.StatusCode)
	}
	readAll(t, resp)
}

// TestSchedulerDisabledEndpoints — без установленного планировщика GET
// возвращает пустой список, POST отвечает 503.
func TestSchedulerDisabledEndpoints(t *testing.T) {
	scheduler.Install(nil)
	ts := newTestServer(t, "http://127.0.0.1:1/none")

	resp, err := http.Get(ts.URL + "/api/scheduler")
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET: %d %s", resp.StatusCode, body)
	}
	if jobs := decodeJobs(t, body); len(jobs) != 0 {
		t.Fatalf("jobs: %+v", jobs)
	}

	resp, _ = http.Post(ts.URL+"/api/scheduler/add", "application/json", strings.NewReader(`{}`))
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("POST без планировщика: %d", resp.StatusCode)
	}
	readAll(t, resp)
}

func getJobs(t *testing.T, ts *httptest.Server) []scheduler.Job {
	t.Helper()
	resp, err := http.Get(ts.URL + "/api/scheduler")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	return decodeJobs(t, readAll(t, resp))
}

func getResults(t *testing.T, ts *httptest.Server, id string) []scheduler.Result {
	t.Helper()
	resp, err := http.Get(ts.URL + "/api/scheduler/results?id=" + id)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Results []scheduler.Result `json:"results"`
	}
	if err := json.Unmarshal(readAll(t, resp), &out); err != nil {
		t.Fatal(err)
	}
	return out.Results
}

func decodeJobs(t *testing.T, body []byte) []scheduler.Job {
	t.Helper()
	var out struct {
		Jobs []scheduler.Job `json:"jobs"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("json: %v (%s)", err, body)
	}
	return out.Jobs
}

func readAll(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 1024)
	for {
		n, err := resp.Body.Read(tmp)
		buf = append(buf, tmp[:n]...)
		if err != nil {
			break
		}
	}
	return buf
}

// TestDebugSchedulerJournal — последние запуски всех задач попадают в
// GET /api/debug (журнал панели «Детали»): свежие сверху, успешные с ответом,
// упавшие с текстом ошибки.
func TestDebugSchedulerJournal(t *testing.T) {
	s := scheduler.New(func(ctx context.Context, job scheduler.Job) (string, error) {
		if job.Name == "ломается" {
			return "", errors.New("mcp недоступен")
		}
		return "ответ " + job.Name, nil
	}, scheduler.Options{Logf: func(string, ...any) {}})
	scheduler.Install(s)
	t.Cleanup(func() { scheduler.Install(nil) })

	ts := newTestServer(t, "http://127.0.0.1:1/none")
	add := func(name string) string {
		resp, err := http.Post(ts.URL+"/api/scheduler/add", "application/json",
			strings.NewReader(`{"name":"`+name+`","prompt":"p","schedule":"@every 1h","enabled":true}`))
		if err != nil {
			t.Fatal(err)
		}
		body := readAll(t, resp)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("add %s: %d %s", name, resp.StatusCode, body)
		}
		jobs := decodeJobs(t, body)
		for _, j := range jobs {
			if j.Name == name {
				return j.ID
			}
		}
		t.Fatalf("add %s: задачи нет в ответе: %+v", name, jobs)
		return ""
	}
	idOK, idBad := add("брифинг"), add("ломается")

	// Ручные запуски обеих задач; выполнение асинхронное — ждём результаты.
	for _, id := range []string{idOK, idBad} {
		if err := s.RunNow(id); err != nil {
			t.Fatalf("RunNow: %v", err)
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if len(s.Results(idOK)) > 0 && len(s.Results(idBad)) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(s.Results(idOK)) == 0 || len(s.Results(idBad)) == 0 {
		t.Fatal("запуски не записали результаты")
	}

	resp, err := http.Get(ts.URL + "/api/debug")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var dbg struct {
		Scheduler []struct {
			Job    string    `json:"job"`
			At     time.Time `json:"at"`
			OK     bool      `json:"ok"`
			DurMS  int64     `json:"dur_ms"`
			Answer string    `json:"answer"`
			Error  string    `json:"error"`
		} `json:"scheduler"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&dbg); err != nil {
		t.Fatal(err)
	}
	if len(dbg.Scheduler) < 2 {
		t.Fatalf("в журнале %d запусков, хотят ≥2: %+v", len(dbg.Scheduler), dbg.Scheduler)
	}
	// Свежие сверху.
	for i := 1; i < len(dbg.Scheduler); i++ {
		if dbg.Scheduler[i].At.After(dbg.Scheduler[i-1].At) {
			t.Fatalf("журнал не отсортирован: %+v", dbg.Scheduler)
		}
	}
	var okRun, badRun bool
	for _, r := range dbg.Scheduler {
		switch {
		case r.Job == "брифинг":
			if !r.OK || r.Answer != "ответ брифинг" || r.Error != "" {
				t.Fatalf("успешный запуск: %+v", r)
			}
			okRun = true
		case r.Job == "ломается":
			if r.OK || !strings.Contains(r.Error, "mcp недоступен") {
				t.Fatalf("упавший запуск: %+v", r)
			}
			badRun = true
		}
	}
	if !okRun || !badRun {
		t.Fatalf("ожидали записи обеих задач: ok=%v bad=%v (%+v)", okRun, badRun, dbg.Scheduler)
	}
}
