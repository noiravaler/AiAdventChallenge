// Package mcpboot поднимает локальные MCP-серверы перед подключением чата.
//
// Перед DialAll приложение проверяет доступность локальных endpoint'ов и, если
// сервер ещё не запущен, стартует его из своего каталога. Поддержаны slugs
// "orbit" (Task17_MCP, порт 5199) и "affirm" (Task20_MCP, порт 5200): оба
// слушают Streamable HTTP; orbit сам поднимает астро-движок Astrologer-API,
// поэтому чату достаточно одного запуска `ai-chat`.
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

// Options — параметры автозапуска. Поля Dir/Bin/LogFile относятся к orbit;
// affirm имеет собственные AffirmDir/AffirmBin/AffirmLog, чтобы серверы можно
// было поднимать и останавливать независимо друг от друга.
type Options struct {
	Dir     string        // каталог orbit (напр. ../Task17_MCP); пусто → пропуск
	Bin     string        // готовый бинарник; пусто → go run ./app
	Timeout time.Duration // сколько ждать готовности сервера (0 = 90 c)
	LogFile string        // файл логов сервера (относительные пути — внутри Dir)

	AffirmDir  string // каталог affirm (Task20_MCP); пусто → пропуск
	AffirmBin  string // готовый бинарник; пусто → go run ./app
	AffirmLog  string // файл логов affirm (относительные пути — внутри AffirmDir)
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
		if !Bootable(s.Slug) {
			continue // автозапуск поддержан только для локальных серверов
		}
		if !isLocal(s.URL) {
			msgs = append(msgs, fmt.Sprintf("%s: URL %s не локальный — автозапуск пропущен", strings.ToLower(s.Slug), s.URL))
			continue
		}
		msgs = append(msgs, ensureOne(ctx, s, opt)...)
	}
	return msgs
}

// EnsureSlug поднимает один локальный сервер по запросу (кнопка «поднять» в
// UI) — в отличие от Ensure не зависит от автозапуска при старте приложения.
func EnsureSlug(ctx context.Context, s config.MCPServer, opt Options) []string {
	if !Bootable(s.Slug) {
		return []string{fmt.Sprintf("%s: автозапуск не поддержан (только локальные orbit и affirm)", s.Slug)}
	}
	if !isLocal(s.URL) {
		return []string{fmt.Sprintf("%s: URL %s не локальный — автозапуск пропущен", strings.ToLower(s.Slug), s.URL)}
	}
	if opt.Timeout <= 0 {
		opt.Timeout = 90 * time.Second
	}
	return ensureOne(ctx, s, opt)
}

// Bootable сообщает, является ли slug локальным сервером, который чат умеет
// поднимать и останавливать сам.
func Bootable(slug string) bool {
	switch strings.ToLower(strings.TrimSpace(slug)) {
	case "orbit", "affirm":
		return true
	}
	return false
}

// perSlug выделяет параметры конкретного сервера: у affirm свой каталог,
// бинарник и лог, у orbit — базовые поля Options.
func perSlug(slug string, opt Options) Options {
	if slug == "affirm" {
		log := opt.AffirmLog
		if strings.TrimSpace(log) == "" {
			log = "bin/affirm-mcp.log"
		}
		return Options{Dir: opt.AffirmDir, Bin: opt.AffirmBin, Timeout: opt.Timeout, LogFile: log}
	}
	return opt
}

// ensureOne поднимает сервер s, если его /api/status не отвечает.
func ensureOne(ctx context.Context, s config.MCPServer, opt Options) []string {
	slug := strings.ToLower(s.Slug)
	o := perSlug(slug, opt)
	base := baseURL(s.URL)
	status := base + "/api/status"
	if probe(ctx, status) {
		return []string{fmt.Sprintf("%s уже запущен (%s)", slug, base)}
	}
	if IsRunning(slug) {
		return []string{fmt.Sprintf("%s: процесс уже запущен чатом, но %s не отвечает; лог: %s", slug, status, logDisplay(o))}
	}
	if o.Dir == "" {
		env := "AI_MCP_DIR"
		if slug == "affirm" {
			env = "AI_MCP_AFFIRM_DIR"
		}
		return []string{fmt.Sprintf("%s не отвечает, автозапуск пропущен: задайте %s (каталог сервера)", slug, env)}
	}
	if !isServerDir(o.Dir) {
		return []string{fmt.Sprintf("%s: в %s нет go.mod — это не каталог сервера, автозапуск пропущен", slug, o.Dir)}
	}

	logFile, err := openLog(o)
	if err != nil {
		return []string{fmt.Sprintf("%s: лог недоступен: %v", slug, err)}
	}
	cmd, err := command(o, hostPort(s.URL, defaultPort(slug)), logFile)
	if err != nil {
		logFile.Close()
		return []string{fmt.Sprintf("%s: %v", slug, err)}
	}
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return []string{fmt.Sprintf("%s: запуск: %v", slug, err)}
	}
	track(slug, cmd, logFile)

	deadline := time.Now().Add(o.Timeout)
	for {
		if probe(ctx, status) {
			return []string{fmt.Sprintf("%s поднят (%s), лог: %s", slug, base, logDisplay(o))}
		}
		if time.Now().After(deadline) {
			return []string{fmt.Sprintf("%s не ответил на /api/status за %s; лог: %s", slug, o.Timeout, logDisplay(o))}
		}
		select {
		case <-ctx.Done():
			return []string{fmt.Sprintf("%s: ожидание прервано: %v", slug, ctx.Err())}
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

// Реестр поднятых процессов по slug: StopAll завершает их при выходе чата,
// StopSlug — по кнопке «стоп» для конкретного сервера.
var (
	mu      sync.Mutex
	running = map[string]*proc{}
)

type proc struct {
	cmd  *exec.Cmd
	log  *os.File
	done chan struct{} // закрывается при выходе процесса
}

func track(slug string, cmd *exec.Cmd, f *os.File) {
	p := &proc{cmd: cmd, log: f, done: make(chan struct{})}
	mu.Lock()
	running[slug] = p
	mu.Unlock()
	// Единственный Wait на процесс — здесь: завершившийся процесс сам
	// убирается из реестра, чтобы «поднять» мог перезапустить упавший сервер.
	go func() {
		_ = cmd.Wait()
		close(p.done)
		mu.Lock()
		if cur := running[slug]; cur == p {
			delete(running, slug)
		}
		mu.Unlock()
	}()
}

// IsRunning сообщает, поднят ли сервер самим чатом (процесс в реестре).
func IsRunning(slug string) bool {
	mu.Lock()
	defer mu.Unlock()
	_, ok := running[strings.ToLower(strings.TrimSpace(slug))]
	return ok
}

// pop убирает сервер из реестра и возвращает его процесс (или nil).
func pop(slug string) *proc {
	mu.Lock()
	defer mu.Unlock()
	p := running[slug]
	delete(running, slug)
	return p
}

// StopSlug останавливает сервер, поднятый чатом (SIGTERM всей группе, затем
// SIGKILL по истечении льготного периода). Возвращает true, если процесс был.
func StopSlug(slug string) bool {
	p := pop(strings.ToLower(strings.TrimSpace(slug)))
	if p == nil {
		return false
	}
	stopOne(p, 8*time.Second)
	_ = p.log.Close()
	return true
}

// StopAll останавливает все MCP-серверы, поднятые через Ensure/EnsureSlug.
func StopAll() {
	mu.Lock()
	procs := running
	running = map[string]*proc{}
	mu.Unlock()
	for _, p := range procs {
		// 8 c хватает orbit на shutdown HTTP и остановку своего движка
		// (движок гасится параллельно с HTTP, SIGTERM уходит сразу).
		stopOne(p, 8*time.Second)
		_ = p.log.Close()
	}
}

func stopOne(p *proc, grace time.Duration) {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return
	}
	pid := p.cmd.Process.Pid
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	select {
	case <-p.done:
	case <-time.After(grace):
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		<-p.done
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

// defaultPort — порт сервера по умолчанию, когда URL не парсится.
func defaultPort(slug string) string {
	if slug == "affirm" {
		return "127.0.0.1:5200"
	}
	return "127.0.0.1:5199"
}

// hostPort извлекает host:port для флага -http; без порта — дефолт сервера.
func hostPort(raw, fallback string) string {
	if u, err := url.Parse(raw); err == nil && u.Host != "" {
		return u.Host
	}
	return fallback
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
