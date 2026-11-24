// Package server — режим Streamable HTTP: MCP-эндпоинт /mcp (для Task16,
// Claude Desktop и т.п.) плюс мини-страница статуса на /.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"log"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"ai-chat-go/orbit"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

//go:embed static/index.html
var indexHTML []byte

// New создаёт HTTP-сервер с MCP-эндпоинтом и страницей статуса.
func New(addr string, tk *orbit.Toolkit, chartsDir string) *http.Server {
	mcpServer := mcp.NewServer(&mcp.Implementation{Name: "orbit-mcp", Version: "0.1.0"}, nil)
	tk.Attach(mcpServer)

	mux := http.NewServeMux()
	mux.Handle("/mcp", mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return mcpServer }, nil))
	mux.Handle("/charts/", chartsHandler(chartsDir))
	mux.HandleFunc("/api/status", statusAPI(tk))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})

	return &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
}

// statusAPI — JSON для страницы статуса: инструменты, карты, health движка.
func statusAPI(tk *orbit.Toolkit) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		status, err := tk.Svc.Status(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"tools":   tk.Defs(),
			"status":  status,
			"charts":  tk.Svc.ChartsList(),
			"renders": tk.Svc.RenderedCharts(),
		}); err != nil {
			log.Printf("status: %v", err)
		}
	}
}

// chartsHandler отдаёт отрисованные SVG-колёса из chartsDir. Отдаём только
// файлы .svg: другой контент из каталога наружу не попадает.
func chartsHandler(dir string) http.Handler {
	// Префикс /charts/ срезается: FileServer отображает остаток пути
	// на файлы внутри dir.
	fs := http.StripPrefix("/charts/", http.FileServer(http.Dir(dir)))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.EqualFold(filepath.Ext(r.URL.Path), ".svg") {
			http.NotFound(w, r)
			return
		}
		// Проставляем тип заранее: http.ServeContent его не перезапишет.
		w.Header().Set("Content-Type", "image/svg+xml")
		w.Header().Set("Cache-Control", "no-store")
		fs.ServeHTTP(w, r)
	})
}
