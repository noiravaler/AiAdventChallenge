// Supervisor — автозапуск self-hosted Astrologer-API из Orbit MCP.
//
// Перед стартом сервер проверяет /health движка; если тот не отвечает,
// супервизор поднимает `uv run poe start` в каталоге клона и ждёт готовности.
// Процесс запускается в собственной группе процессов, чтобы при остановке
// Orbit можно было завершить всю цепочку (uv → python/uvicorn) разом.
package astrologer

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// EngineOptions — параметры супервизора движка.
type EngineOptions struct {
	BaseURL string        // базовый URL движка, напр. http://127.0.0.1:8000
	Dir     string        // каталог клона Astrologer-API (где pyproject.toml)
	Timeout time.Duration // сколько ждать готовности после запуска (0 = 2 мин)
	Every   time.Duration // период опроса /health (0 = 2 c; для тестов)
	LogFile string        // файл для stdout/stderr движка (0 = bin/astrologer.log)
}

// Supervisor управляет процессом движка: запуск, ожидание готовности, остановка.
type Supervisor struct {
	opt     EngineOptions
	command func(dir string, out io.Writer) (*exec.Cmd, error) // переопределяется в тестах

	mu      sync.Mutex
	cmd     *exec.Cmd
	logFile *os.File
	stopped bool // Stop() вызван: запуск движка более не нужен
}

// NewSupervisor создаёт супервизор с настройками по умолчанию.
func NewSupervisor(opt EngineOptions) *Supervisor {
	if opt.Timeout <= 0 {
		opt.Timeout = 2 * time.Minute
	}
	return &Supervisor{opt: opt, command: defaultCommand}
}

// Ensure убеждается, что движок отвечает на /health. Если нет — запускает
// `uv run poe start` и ждёт готовности. Возвращает "running", если движок
// уже работал, и "started", если его поднял супервизор.
func (s *Supervisor) Ensure(ctx context.Context) (string, error) {
	if s.healthy() {
		return "running", nil
	}
	if s.opt.Dir == "" {
		return "", fmt.Errorf("ORBIT_ASTRO_DIR не задан: укажите каталог клона Astrologer-API, чтобы Orbit поднимал движок сам")
	}

	logFile, err := openLog(s.opt.LogFile)
	if err != nil {
		return "", fmt.Errorf("лог движка: %w", err)
	}
	cmd, err := s.command(s.opt.Dir, logFile)
	if err != nil {
		logFile.Close()
		return "", err
	}
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return "", fmt.Errorf("запуск движка: %w", err)
	}

	s.mu.Lock()
	if s.stopped {
		// Stop() успел отработать до записи cmd: гасим только что
		// запущенный процесс, иначе он останется висеть без присмотра.
		s.mu.Unlock()
		stopProc(cmd)
		_ = logFile.Close()
		return "", fmt.Errorf("движок остановлен во время запуска")
	}
	s.cmd, s.logFile = cmd, logFile
	s.mu.Unlock()

	every := s.opt.Every
	if every <= 0 {
		every = 2 * time.Second
	}
	deadline := time.Now().Add(s.opt.Timeout)
	for {
		if s.healthy() {
			return "started", nil
		}
		if s.isStopped() {
			// Stop() погасил процесс — выходим сразу, не ждём дедлайна.
			return "", fmt.Errorf("движок остановлен во время запуска")
		}
		if time.Now().After(deadline) {
			return "", fmt.Errorf("движок не ответил на /health за %s; логи: %s", s.opt.Timeout, logPath(s.opt.LogFile))
		}
		select {
		case <-ctx.Done():
			return "", fmt.Errorf("ожидание движка прервано: %w", ctx.Err())
		case <-time.After(every):
		}
	}
}

// Stop останавливает движок, запущенный супервизором (вся группа процессов).
// Уже работавший движок (Ensure → "running") не трогаем: он не наш.
// Вызов безопасен в любой момент, в том числе пока Ensure ещё стартует:
// флаг stopped не даст запустить «осиротевший» движок после Stop.
func (s *Supervisor) Stop() {
	s.mu.Lock()
	s.stopped = true
	cmd, f := s.cmd, s.logFile
	s.cmd, s.logFile = nil, nil
	s.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		if f != nil {
			_ = f.Close()
		}
		return
	}
	stopProc(cmd)
	if f != nil {
		_ = f.Close()
	}
}

func (s *Supervisor) isStopped() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stopped
}

// stopProc гасит процесс и всю его группу: SIGTERM, льготный период,
// затем SIGKILL; дожидается завершения.
func stopProc(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && alive(pid) {
		time.Sleep(200 * time.Millisecond)
	}
	if alive(pid) {
		_ = syscall.Kill(-pid, syscall.SIGKILL)
	}
	_ = cmd.Wait()
}

// healthy проверяет /health движка (503 во время инициализации = не готов).
func (s *Supervisor) healthy() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		strings.TrimRight(s.opt.BaseURL, "/")+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// defaultCommand строит процесс `uv run poe start` в каталоге клона.
func defaultCommand(dir string, out io.Writer) (*exec.Cmd, error) {
	uv, err := findUv()
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(uv, "run", "poe", "start")
	cmd.Dir = dir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // своя группа: остановка разом
	return cmd, nil
}

// findUv ищет uv в PATH и в стандартных местах установки: автозапуск должен
// работать и из неинтерактивных оболочек, где ~/.local/bin не в PATH.
func findUv() (string, error) {
	if p, err := exec.LookPath("uv"); err == nil {
		return p, nil
	}
	home, _ := os.UserHomeDir()
	for _, cand := range []string{
		filepath.Join(home, ".local", "bin", "uv"),
		filepath.Join(home, ".cargo", "bin", "uv"),
		"/opt/homebrew/bin/uv",
		"/usr/local/bin/uv",
	} {
		if st, err := os.Stat(cand); err == nil && !st.IsDir() {
			return cand, nil
		}
	}
	return "", fmt.Errorf("uv не найден ни в PATH, ни в ~/.local/bin: установите (curl -LsSf https://astral.sh/uv/install.sh | sh) и переоткройте терминал")
}

func openLog(path string) (*os.File, error) {
	path = logPath(path)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

func logPath(path string) string {
	if strings.TrimSpace(path) == "" {
		return "bin/astrologer.log"
	}
	return path
}

func alive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}
