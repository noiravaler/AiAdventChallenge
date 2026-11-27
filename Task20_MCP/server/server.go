// Package server — режим Streamable HTTP: MCP-эндпоинт /mcp (для хаба
// Task16/17/18/19, Claude Desktop и т.п.) плюс мини-страница статуса на /.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"ai-chat-go/affirm"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

//go:embed static/index.html
var indexHTML []byte

// New создаёт HTTP-сервер с MCP-эндпоинтом и страницей статуса.
func New(addr string, tk *affirm.Toolkit) *http.Server {
	mcpServer := mcp.NewServer(&mcp.Implementation{Name: "affirm-mcp", Version: "0.1.0"}, nil)
	tk.Attach(mcpServer)

	mux := http.NewServeMux()
	mux.Handle("/mcp", mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return mcpServer }, nil))
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

// statusAPI — JSON для страницы статуса: инструменты и health провайдеров.
func statusAPI(tk *affirm.Toolkit) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"tools":  tk.Defs(),
			"status": tk.Status(ctx),
			"now":    time.Now().Format(time.RFC3339),
		}); err != nil {
			log.Printf("status: %v", err)
		}
	}
}
