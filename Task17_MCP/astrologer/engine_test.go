package astrologer

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// newEngineServer — фейковый движок: n первых запросов к /health дают 503
// (инициализация), дальше 200.
func newEngineServer(t *testing.T, failFirst int64) *httptest.Server {
	t.Helper()
	var n atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if n.Add(1) <= failFirst {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		fmt.Fprint(w, `{"status":"OK"}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestEnsureAlreadyRunning(t *testing.T) {
	srv := newEngineServer(t, 0)
	sup := NewSupervisor(EngineOptions{BaseURL: srv.URL})
	// command не должен вызываться: движок уже работает.
	sup.command = func(string, io.Writer) (*exec.Cmd, error) {
		t.Fatal("spawn не ожидался")
		return nil, errors.New("unexpected spawn")
	}
	st, err := sup.Ensure(context.Background())
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if st != "running" {
		t.Fatalf("статус = %q, хочу running", st)
	}
}

func TestEnsureBootsEngine(t *testing.T) {
	srv := newEngineServer(t, 1) // первый health — 503, дальше OK
	var spawned int
	sup := NewSupervisor(EngineOptions{BaseURL: srv.URL, Dir: t.TempDir(), Every: 5 * time.Millisecond, Timeout: 5 * time.Second})
	sup.command = func(dir string, _ io.Writer) (*exec.Cmd, error) {
		spawned++
		if dir == "" {
			t.Fatal("каталог клона пуст")
		}
		return exec.Command("true"), nil // безвредная «пустая» команда
	}
	st, err := sup.Ensure(context.Background())
	if err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	if st != "started" {
		t.Fatalf("статус = %q, хочу started", st)
	}
	if spawned != 1 {
		t.Fatalf("spawn вызван %d раз, хочу 1", spawned)
	}
}

func TestEnsureNoDir(t *testing.T) {
	sup := NewSupervisor(EngineOptions{BaseURL: "http://127.0.0.1:1"}) // порт заведомо закрыт
	_, err := sup.Ensure(context.Background())
	if err == nil || !strings.Contains(err.Error(), "ORBIT_ASTRO_DIR") {
		t.Fatalf("хочу ошибку про ORBIT_ASTRO_DIR, получили: %v", err)
	}
}

func TestStopWithoutProcess(t *testing.T) {
	sup := NewSupervisor(EngineOptions{BaseURL: "http://127.0.0.1:1"})
	sup.Stop() // не должно паниковать
}

// spawnSleep возвращает «длинный» процесс в собственной группе — как движок.
func spawnSleep(dir string, out io.Writer, sink **exec.Cmd) (*exec.Cmd, error) {
	cmd := exec.Command("sleep", "300")
	cmd.Dir = dir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	*sink = cmd
	return cmd, nil
}

// TestStopBeforeStartCompletes: Stop() успевает отработать до того, как Ensure
// запомнит процесс, — только что запущенный движок должен быть погашен, а не
// оставлен висеть сиротой.
func TestStopBeforeStartCompletes(t *testing.T) {
	entered := make(chan struct{}, 1)
	gate := make(chan struct{})
	var cmd *exec.Cmd
	sup := NewSupervisor(EngineOptions{
		BaseURL: "http://127.0.0.1:1", // health всегда недоступен
		Dir:     t.TempDir(),
		Every:   5 * time.Millisecond,
		Timeout: 30 * time.Second,
	})
	sup.command = func(dir string, out io.Writer) (*exec.Cmd, error) {
		entered <- struct{}{}
		<-gate // держим, пока тест не вызовет Stop()
		return spawnSleep(dir, out, &cmd)
	}

	errCh := make(chan error, 1)
	go func() {
		_, err := sup.Ensure(context.Background())
		errCh <- err
	}()

	<-entered      // command вызван, процесс ещё не запущен
	sup.Stop()     // флаг stopped ставится при cmd == nil
	close(gate)    // выпускаем command → Ensure запустит процесс
	err := <-errCh // Ensure должен сразу погасить процесс и выйти с ошибкой

	if err == nil || !strings.Contains(err.Error(), "остановлен во время запуска") {
		t.Fatalf("хочу ошибку про остановку при запуске, получили: %v", err)
	}
	if cmd == nil || alive(cmd.Process.Pid) {
		t.Fatalf("запущенный процесс должен быть погашен")
	}
}

// TestStopDuringStartupWait: Stop() во время ожидания готовности — Ensure
// выходит сразу с ошибкой, процесс гасится, дедлайн не выжидается.
func TestStopDuringStartupWait(t *testing.T) {
	var cmd *exec.Cmd
	sup := NewSupervisor(EngineOptions{
		BaseURL: "http://127.0.0.1:1", // health всегда недоступен
		Dir:     t.TempDir(),
		Every:   5 * time.Millisecond,
		Timeout: 30 * time.Second,
	})
	sup.command = func(dir string, out io.Writer) (*exec.Cmd, error) {
		return spawnSleep(dir, out, &cmd)
	}

	errCh := make(chan error, 1)
	go func() {
		_, err := sup.Ensure(context.Background())
		errCh <- err
	}()

	time.Sleep(150 * time.Millisecond) // Ensure в цикле опроса
	sup.Stop()

	select {
	case err := <-errCh:
		if err == nil || !strings.Contains(err.Error(), "остановлен во время запуска") {
			t.Fatalf("хочу ошибку про остановку при запуске, получили: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Ensure не завершился после Stop() — дедлайн выжидается")
	}
	if cmd == nil || alive(cmd.Process.Pid) {
		t.Fatalf("запущенный процесс должен быть погашен")
	}
}
