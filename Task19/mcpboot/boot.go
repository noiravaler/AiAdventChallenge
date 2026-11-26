// Package mcpboot поднимает локальные MCP-серверы перед подключением чата.
//
// Перед DialAll приложение проверяет доступность локальных endpoint'ов и, если
// сервер ещё не запущен, стартует его из своего каталога. Поддержан slug
// "orbit" (Task17_MCP): он слушает Streamable HTTP и сам поднимает астро-движок
// Astrologer-API, поэтому чату достаточно одного запуска `ai-chat`.
package mcpboot

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"ai-chat-go/config"
)

// Options — параметры автозапуска.
type Options struct {
	Dir     string        // каталог MCP-сервера (напр. ../Task17_MCP); пусто → пропуск
	Bin     string        // готовый бинарник; пусто → go run ./app
	Timeout time.Duration // сколько ждать готовности сервера (0 = 90 c)
	LogFile string        // файл логов сервера (относительные пути — внутри Dir)
}

var (
	command = defaultCommand // подменяется в тестах
	probe   = defaultProbe   // подменяется в тестах
)

const (
	probeTimeout = 3 * time.Second
	probeEvery   = 1 * time.Second
)

// Ensure проверяет локальные MCP-серверы и поднимает те, что не отвечают.
// Возвращает человекочитаемые сообщения для журнала приложения.
func Ensure(ctx context.Context, servers []config.MCPServer, opt Options) []string {
	if opt.Timeout <= 0 {
		opt.Timeout = 90 * time.Second
	}
	var msgs []string
	for _, s := range servers {
		if !strings.EqualFold(s.Slug, "orbit") {
			continue // автозапуск поддержан только для orbit
		}
		if !isLocal(s.URL) {
			msgs = append(msgs, fmt.Sprintf("orbit: URL %s не локальный — автозапуск пропущен", s.URL))
			continue
		}
		msgs = append(msgs, ensureOrbit(ctx, s, opt)...)
	}
	return msgs
}

// ensureOrbit поднимает Orbit MCP, если его /api/status не отвечает.
func ensureOrbit(ctx context.Context, s config.MCPServer, opt Options) []string {
	base := baseURL(s.URL)
	if probe(ctx, base+"/api/status") {
		return []string{fmt.Sprintf("orbit уже запущен (%s)", base)}
	}
	if opt.Dir == "" {
		return []string{"orbit не отвечает, автозапуск пропущен: задайте AI_MCP_DIR (каталог Task17_MCP)"}
	}
	if !isServerDir(opt.Dir) {
		return []string{fmt.Sprintf("orbit: в %s нет go.mod — это не каталог сервера, автозапуск пропущен", opt.Dir)}
	}

	logFile, err := openLog(opt)
	if err != nil {
		return []string{fmt.Sprintf("orbit: лог недоступен: %v", err)}
	}
	cmd, err := command(opt, hostPort(s.URL), logFile)
	if err != nil {
		logFile.Close()
		return []string{fmt.Sprintf("orbit: %v", err)}
	}
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return []string{fmt.Sprintf("orbit: запуск: %v", err)}
	}
	track(cmd, logFile)

	deadline := time.Now().Add(opt.Timeout)
	for {
		if probe(ctx, base+"/api/status") {
			return []string{fmt.Sprintf("orbit поднят (%s), лог: %s", base, logDisplay(opt))}
		}
		if time.Now().After(deadline) {
			return []string{fmt.Sprintf("orbit не ответил на /api/status за %s; лог: %s", opt.Timeout, logDisplay(opt))}
		}
		select {
		case <-ctx.Done():
			return []string{fmt.Sprintf("orbit: ожидание прервано: %v", ctx.Err())}
		case <-time.After(probeEvery):
		}
	}
}

// defaultCommand строит процесс MCP-сервера в режиме Streamable HTTP.
func defaultCommand(opt Options, addr string, out io.Writer) (*exec.Cmd, error) {
	var cmd *exec.Cmd
	if opt.Bin != "" {
		cmd = exec.Command(opt.Bin, "-http", addr)
	} else {
		if _, err := exec.LookPath("go"); err != nil {
			return nil, fmt.Errorf("go не найден в PATH (или соберите бинарник и задайте AI_MCP_BIN)")
		}
		cmd = exec.Command("go", "run", "./app", "-http", addr)
	}
	cmd.Dir = opt.Dir
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // своя группа: остановка разом
	return cmd, nil
}

// defaultProbe считает сервер живым при любом HTTP-ответе: для целей
// автозапуска важно лишь, что порт кто-то слушает.
func defaultProbe(ctx context.Context, raw string) bool {
	pctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(pctx, http.MethodGet, raw, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return true
}

// Реестр поднятых процессов: StopAll завершает их при выходе чата.
var (
	mu      sync.Mutex
	running []*exec.Cmd
	logs    []*os.File
)

func track(cmd *exec.Cmd, f *os.File) {
	mu.Lock()
	running = append(running, cmd)
	logs = append(logs, f)
	mu.Unlock()
}

// StopAll останавливает все MCP-серверы, поднятые через Ensure (SIGTERM всей
// группе процессов, затем SIGKILL по истечении льготного периода).
func StopAll() {
	mu.Lock()
	cmds, files := running, logs
	running, logs = nil, nil
	mu.Unlock()
	for _, cmd := range cmds {
		// 8 c хватает orbit на shutdown HTTP и остановку своего движка
		// (движок гасится параллельно с HTTP, SIGTERM уходит сразу).
		stopOne(cmd, 8*time.Second)
	}
	for _, f := range files {
		_ = f.Close()
	}
}

func stopOne(cmd *exec.Cmd, grace time.Duration) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	done := make(chan struct{})
	go func() { _ = cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(grace):
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		<-done
	}
}

// isServerDir reports whether dir looks like the MCP server home: it exists
// and contains go.mod (spurious or empty directories are rejected).
func isServerDir(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, "go.mod"))
	return err == nil && !st.IsDir()
}

// baseURL убирает суффикс /mcp: http://127.0.0.1:5199/mcp → http://127.0.0.1:5199.
func baseURL(raw string) string {
	u := strings.TrimRight(raw, "/")
	if strings.HasSuffix(strings.ToLower(u), "/mcp") {
		u = u[:len(u)-len("/mcp")]
	}
	return u
}

// hostPort извлекает host:port для флага -http; без порта — дефолт Orbit.
func hostPort(raw string) string {
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Host
	}
	return "127.0.0.1:5199"
}

func isLocal(raw string) bool {
	lower := strings.ToLower(raw)
	for _, marker := range []string{"localhost", "127.0.0.1", "[::1]"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func openLog(opt Options) (*os.File, error) {
	path := logPath(opt)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
}

func logPath(opt Options) string {
	path := opt.LogFile
	if strings.TrimSpace(path) == "" {
		path = "bin/orbit-mcp.log"
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(opt.Dir, path)
	}
	return path
}

func logDisplay(opt Options) string { return logPath(opt) }
