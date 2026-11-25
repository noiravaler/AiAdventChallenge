// Command app — точка входа Orbit MCP: MCP-сервер натальных карт и
// гороскопов (движок: self-hosted Astrologer-API, гео/погода: Open-Meteo).
//
// Режимы:
//
//	go run ./app                 — MCP over stdio (Claude Desktop, любой MCP-клиент)
//	go run ./app -http 127.0.0.1:5199 — MCP over Streamable HTTP (/mcp) + страница статуса (/)
//	go run ./app -list           — список инструментов
//	go run ./app -probe natal_build_chart -args '{"name":"Anna","birth_date":"1995-07-21","birth_time":"14:30","city":"Казань"}'
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
	_ "time/tzdata" // встроенная база таймзон: сервер не зависит от ОС

	"ai-chat-go/astrologer"
	"ai-chat-go/config"
	"ai-chat-go/orbit"
	"ai-chat-go/server"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("orbit ")

	cfg, err := config.Load(os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}

	svc := orbit.New(cfg)
	if st, err := orbit.NewStore(cfg.StoreFile); err == nil {
		svc = svc.WithStore(st)
	} else {
		log.Printf("хранилище карт не загружено (%v) — стартуем пустым", err)
	}
	if jr, err := orbit.NewJournal(cfg.JournalFile); err == nil {
		svc = svc.WithJournal(jr)
	} else {
		log.Printf("дневник наблюдений не загружен (%v) — стартуем пустым", err)
	}
	if ps, err := orbit.NewPulseStore(cfg.PulseDB); err == nil {
		svc = svc.WithPulse(ps)
		defer ps.Close()
	} else {
		log.Printf("база пульса не открыта (%v) — пульс недоступен", err)
	}
	tk := orbit.NewToolkit(svc)
	eng := newEngineSupervisor(cfg)

	switch {
	case cfg.List:
		for _, d := range tk.Defs() {
			fmt.Printf("%s\n    %s\n", d.Name, d.Description)
		}
		return

	case cfg.Probe != "":
		ensureEngine(cfg, eng) // probe ждёт готовности движка
		runProbe(cfg, tk)

	case cfg.HTTPAddr != "":
		runHTTP(cfg, tk, eng)

	default:
		runStdio(cfg, tk, eng)
	}
}

// newEngineSupervisor возвращает супервизор движка или nil, если автозапуск
// выключен (ORBIT_ENGINE_AUTOSTART / ORBIT_ASTRO_DIR, см. config).
func newEngineSupervisor(cfg *config.Config) *astrologer.Supervisor {
	if !cfg.EngineAutostartEnabled() {
		return nil
	}
	return astrologer.NewSupervisor(astrologer.EngineOptions{
		BaseURL: cfg.AstroURL,
		Dir:     cfg.AstroDir,
		Timeout: cfg.EngineStartTimeout,
		LogFile: cfg.EngineLog,
	})
}

// ensureEngine блокирующе поднимает движок (режим -probe).
func ensureEngine(cfg *config.Config, eng *astrologer.Supervisor) {
	if eng == nil {
		return
	}
	log.Printf("автозапуск движка: %s (каталог %s)", cfg.AstroURL, cfg.AstroDir)
	if st, err := eng.Ensure(context.Background()); err != nil {
		log.Printf("движок не поднят: %v", err)
	} else {
		log.Printf("движок: %s", st)
	}
}

// ensureEngineAsync поднимает движок в фоне: сервер стартует сразу,
// а статус подъёма виден в журнале и на странице / (режимы stdio и -http).
func ensureEngineAsync(cfg *config.Config, eng *astrologer.Supervisor, ctx context.Context) {
	log.Printf("движок не отвечает — автозапуск: %s (каталог %s)", cfg.AstroURL, cfg.AstroDir)
	st, err := eng.Ensure(ctx)
	if err != nil {
		log.Printf("автозапуск движка не удался: %v", err)
		return
	}
	log.Printf("движок: %s", st)
}

// runStdio — классический режим MCP-сервера поверх stdio.
func runStdio(cfg *config.Config, tk *orbit.Toolkit, eng *astrologer.Supervisor) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := mcp.NewServer(&mcp.Implementation{Name: "orbit-mcp", Version: "0.1.0"}, nil)
	tk.Attach(srv)
	if eng != nil {
		go ensureEngineAsync(cfg, eng, ctx)
	}
	log.Printf("MCP over stdio: движок %s/api/%s, карты → %s", cfg.AstroURL, cfg.AstroVersion, cfg.StoreFile)
	if err := srv.Run(ctx, &mcp.StdioTransport{}); err != nil {
		log.Printf("stdio завершён: %v", err)
	}
	if eng != nil {
		eng.Stop() // останавливаем только тот движок, что подняли сами
	}
}

// runHTTP — Streamable HTTP (/mcp) + статус-страница (/).
func runHTTP(cfg *config.Config, tk *orbit.Toolkit, eng *astrologer.Supervisor) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Ссылки на отрисованные карты в результатах инструментов строятся
	// от публичного адреса HTTP-режима.
	tk.Svc.PublicURL = "http://" + cfg.HTTPAddr
	hs := server.New(cfg.HTTPAddr, tk, tk.Svc.ChartsDir)
	go func() {
		<-ctx.Done()
		// Движок гасим параллельно с HTTP-shutdown, а не после него:
		// SIGTERM уходит в группу движка сразу, поэтому полная остановка
		// укладывается в льготный период остановившего нас процесса.
		done := make(chan struct{})
		if eng != nil {
			go func() {
				defer close(done)
				eng.Stop()
			}()
		} else {
			close(done)
		}
		shCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = hs.Shutdown(shCtx)
		<-done // дожидаемся полной остановки движка перед выходом
	}()

	if eng != nil {
		go ensureEngineAsync(cfg, eng, ctx)
	}
	log.Printf("Orbit MCP: http://%s/mcp (Streamable HTTP), статус: http://%s/, карты: http://%s/charts/", cfg.HTTPAddr, cfg.HTTPAddr, cfg.HTTPAddr)
	log.Printf("движок %s/api/%s, карты → %s, SVG → %s", cfg.AstroURL, cfg.AstroVersion, cfg.StoreFile, tk.Svc.ChartsDir)
	if err := hs.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		if eng != nil {
			eng.Stop() // выходим через log.Fatal мимо defer — гасим движок сами
		}
		log.Fatal(err)
	}
}

// runProbe — прямой вызов инструмента без MCP-транспорта (тесты и отладка).
func runProbe(cfg *config.Config, tk *orbit.Toolkit) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	raw := json.RawMessage(cfg.Args)
	payload, err := tk.Invoke(ctx, cfg.Probe, raw)
	out := map[string]any{"tool": cfg.Probe}
	if err != nil {
		out["error"] = err.Error()
	} else if rr, ok := payload.(*orbit.RenderResult); ok {
		// SVG не печатаем — только сводка для модели и размер картинки.
		out["result"] = rr.Model
		out["image"] = map[string]any{"mime": "image/svg+xml", "bytes": len(rr.SVG), "file": rr.FileName}
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
