// Command app — точка входа Affirm MCP: отдельный MCP-сервер аффирмаций
// (Task20_MCP). Функциональность вынесена из Task17_MCP (orbit): клиент
// affirmations.dev и перевод через Google Translate (gtx).
//
// Режимы:
//
//	go run ./app                      — MCP over stdio (Claude Desktop, любой MCP-клиент)
//	go run ./app -http 127.0.0.1:5200 — MCP over Streamable HTTP (/mcp) + страница статуса (/)
//	go run ./app -list                — список инструментов
//	go run ./app -probe daily_affirmation
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ai-chat-go/affirm"
	"ai-chat-go/affirmations"
	"ai-chat-go/config"
	"ai-chat-go/server"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("affirm ")

	cfg, err := config.Load(os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}

	cl := affirmations.New(cfg.AffirmURL, cfg.TranslateURL, cfg.AffirmTimeout)
	tk := affirm.NewToolkit(cl)

	switch {
	case cfg.List:
		for _, d := range tk.Defs() {
			fmt.Printf("%s\n    %s\n", d.Name, d.Description)
		}
		return

	case cfg.Probe != "":
		runProbe(cfg, tk)

	case cfg.HTTPAddr != "":
		runHTTP(cfg, tk)

	default:
		runStdio(cfg, tk)
	}
}

// runStdio — классический режим MCP-сервера поверх stdio.
func runStdio(cfg *config.Config, tk *affirm.Toolkit) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := mcp.NewServer(&mcp.Implementation{Name: "affirm-mcp", Version: "0.1.0"}, nil)
	tk.Attach(srv)
	log.Printf("MCP over stdio: аффирмации %s, переводчик %s", cfg.AffirmURL, cfg.TranslateURL)
	if err := srv.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Printf("stdio завершён: %v", err)
	}
}

// runHTTP — Streamable HTTP (/mcp) + статус-страница (/).
func runHTTP(cfg *config.Config, tk *affirm.Toolkit) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	hs := server.New(cfg.HTTPAddr, tk)
	go func() {
		<-ctx.Done()
		shCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = hs.Shutdown(shCtx)
	}()

	log.Printf("Affirm MCP: http://%s/mcp (Streamable HTTP), статус: http://%s/", cfg.HTTPAddr, cfg.HTTPAddr)
	log.Printf("аффирмации %s, переводчик %s", cfg.AffirmURL, cfg.TranslateURL)
	if err := hs.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

// runProbe — прямой вызов инструмента без MCP-транспорта (тесты и отладка).
func runProbe(cfg *config.Config, tk *affirm.Toolkit) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	raw := json.RawMessage(cfg.Args)
	payload, err := tk.Invoke(ctx, cfg.Probe, raw)
	out := map[string]any{"tool": cfg.Probe}
	if err != nil {
		out["error"] = err.Error()
	} else {
		out["result"] = payload
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
	if err != nil {
		os.Exit(1)
	}
}
