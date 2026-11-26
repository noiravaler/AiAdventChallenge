package mcpboot

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"ai-chat-go/config"
)

// с настоящими процессами тесты не работают: подменяем command и probe.
func stubCommand(t *testing.T, dir *string, args *[]string) {
	t.Helper()
	old := command
	command = func(opt Options, addr string, _ io.Writer) (*exec.Cmd, error) {
		if dir != nil {
			*dir = opt.Dir
		}
		if args != nil {
			*args = []string{addr}
		}
		return exec.Command("true"), nil
	}
	t.Cleanup(func() { command = old })
}

func stubProbe(t *testing.T, upAfter int64) {
	t.Helper()
	old := probe
	var n atomic.Int64
	probe = func(context.Context, string) bool {
		return n.Add(1) > upAfter
	}
	t.Cleanup(func() { probe = old })
}

func orbitServer(url string) config.MCPServer {
	return config.MCPServer{Slug: "orbit", Name: "Orbit", URL: url}
}

func TestBaseURL(t *testing.T) {
	tests := map[string]string{
		"http://127.0.0.1:5199/mcp":   "http://127.0.0.1:5199",
		"http://127.0.0.1:5199/mcp/":  "http://127.0.0.1:5199",
		"http://localhost:5199/MCP":   "http://localhost:5199",
		"http://127.0.0.1:5199":       "http://127.0.0.1:5199",
		"https://mcp.example.com/mcp": "https://mcp.example.com",
	}
	for raw, want := range tests {
		if got := baseURL(raw); got != want {
			t.Errorf("baseURL(%q) = %q, хочу %q", raw, got, want)
		}
	}
}

func TestEnsureSkipsRemoteAndForeignSlugs(t *testing.T) {
	stubProbe(t, 0)
	command = func(Options, string, io.Writer) (*exec.Cmd, error) {
		t.Fatal("spawn не ожидался")
		return nil, fmt.Errorf("unexpected spawn")
	}
	msgs := Ensure(context.Background(), []config.MCPServer{
		{Slug: "dubai", Name: "Dubai Grocery Prices", URL: "https://prices.brainy.ae/mcp"},
		orbitServer("https://astro.example.com/mcp"), // orbit, но не локальный
	}, Options{Dir: "/tmp"})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "не локальный") {
		t.Fatalf("хочу одно сообщение про не-локальный URL, получили: %v", msgs)
	}
}

func TestEnsureAlreadyUp(t *testing.T) {
	stubProbe(t, 0)
	command = func(Options, string, io.Writer) (*exec.Cmd, error) {
		t.Fatal("spawn не ожидался")
		return nil, fmt.Errorf("unexpected spawn")
	}
	msgs := Ensure(context.Background(), []config.MCPServer{orbitServer("http://127.0.0.1:5199/mcp")}, Options{Dir: "/tmp"})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "уже запущен") {
		t.Fatalf("хочу сообщение о работающем сервере, получили: %v", msgs)
	}
}

func TestEnsureNoDir(t *testing.T) {
	stubProbe(t, 1<<62) // сервер всегда вниз
	msgs := Ensure(context.Background(), []config.MCPServer{orbitServer("http://127.0.0.1:5199/mcp")}, Options{})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "AI_MCP_DIR") {
		t.Fatalf("хочу подсказку про AI_MCP_DIR, получили: %v", msgs)
	}
}

func TestEnsureSpawnsOrbit(t *testing.T) {
	stubProbe(t, 1) // первый probe — сервер вниз, дальше «поднялся»
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fake\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var gotDir string
	var args []string
	stubCommand(t, &gotDir, &args)
	msgs := Ensure(context.Background(),
		[]config.MCPServer{orbitServer("http://127.0.0.1:5199/mcp")},
		Options{Dir: dir, Timeout: 5 * time.Second})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "поднят") {
		t.Fatalf("хочу сообщение о подъёме, получили: %v", msgs)
	}
	if gotDir != dir {
		t.Fatalf("cmd.Dir = %q, хочу %q", gotDir, dir)
	}
	if len(args) != 1 || args[0] != "127.0.0.1:5199" {
		t.Fatalf("адрес для -http = %v, хочу 127.0.0.1:5199", args)
	}
}

func TestEnsureRejectsDirWithoutGoMod(t *testing.T) {
	stubProbe(t, 1<<62) // сервер всегда вниз
	stubCommand(t, nil, nil)
	msgs := Ensure(context.Background(),
		[]config.MCPServer{orbitServer("http://127.0.0.1:5199/mcp")},
		Options{Dir: t.TempDir(), Timeout: time.Second}) // пустой каталог без go.mod
	if len(msgs) != 1 || !strings.Contains(msgs[0], "go.mod") {
		t.Fatalf("хочу отказ из-за отсутствия go.mod, получили: %v", msgs)
	}
}

func TestEnsureSpawnTimeoutMessage(t *testing.T) {
	stubProbe(t, 1<<62) // сервер так и не поднялся
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fake\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	stubCommand(t, nil, nil)
	msgs := Ensure(context.Background(),
		[]config.MCPServer{orbitServer("http://127.0.0.1:5199/mcp")},
		Options{Dir: dir, Timeout: 30 * time.Millisecond})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "не ответил") {
		t.Fatalf("хочу сообщение о таймауте, получили: %v", msgs)
	}
}

func TestDefaultProbeAcceptsAnyHTTPAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound) // даже 404 значит «порт живой»
	}))
	t.Cleanup(srv.Close)
	if !defaultProbe(context.Background(), srv.URL) {
		t.Fatal("probe должен вернуть true на любой HTTP-ответ")
	}
}
