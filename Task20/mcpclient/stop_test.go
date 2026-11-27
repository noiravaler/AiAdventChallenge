package mcpclient

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ai-chat-go/api"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// startTestMCPServer поднимает настоящий MCP-сервер (Streamable HTTP) с одним
// инструментом ping и возвращает тестовый сервер httptest.
func startTestMCPServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := mcp.NewServer(&mcp.Implementation{Name: "test-mcp", Version: "0.0.1"}, nil)
	mcp.AddTool(srv, &mcp.Tool{Name: "ping", Description: "отвечает pong"},
		func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "pong"}}}, nil, nil
		})
	ts := httptest.NewServer(mcp.NewStreamableHTTPHandler(
		func(*http.Request) *mcp.Server { return srv }, nil))
	t.Cleanup(ts.Close)
	return ts
}

// TestStopKeepsToolsForModel — семантика кнопки «стоп» (симуляция отказа
// сервиса): после остановки инструменты остаются в списке, передаваемом
// модели, а вызов падает с ошибкой про остановку; тумблер же убирает
// инструменты из списка. «Поднять» (Resume + Dial) возвращает рабочий режим.
func TestStopKeepsToolsForModel(t *testing.T) {
	ts := startTestMCPServer(t)
	hub := NewHub([]ServerDef{{Slug: "t", Name: "Test", URL: ts.URL + "/mcp"}}, 5*time.Second)
	defer hub.Close()

	var exposed []api.Tool
	hub.SetOnToolsChange(func(list []api.Tool) { exposed = list })
	ctx := context.Background()
	if err := hub.Dial(ctx, "t"); err != nil {
		t.Fatalf("dial: %v", err)
	}
	if len(exposed) != 1 || exposed[0].Function.Name != "ping" {
		t.Fatalf("после Dial модель должна видеть ping, получили %v", exposed)
	}

	// Стоп: процесс остаётся в списке модели, вызов падает с ошибкой.
	if err := hub.Stop("t"); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if len(exposed) != 1 || exposed[0].Function.Name != "ping" {
		t.Fatalf("после Stop инструменты должны остаться у модели, получили %v", exposed)
	}
	if _, _, _, _, err := hub.Execute(ctx, "ping", nil); err == nil ||
		!strings.Contains(err.Error(), "остановлен") {
		t.Fatalf("вызов на остановленном сервере должен падать с «остановлен», получили: %v", err)
	}

	// Авто-реконнект не должен поднимать остановленный сервер.
	hub.StartAutoReconnect(time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	if info := hub.Infos()[0]; !info.Stopped || info.Connected {
		t.Fatalf("авто-реконнект не должен трогать остановленный сервер: %+v", info)
	}

	// Тумблер, в отличие от «стопа», убирает инструменты из списка.
	if err := hub.SetEnabled(ctx, "t", false); err != nil {
		t.Fatalf("toggle off: %v", err)
	}
	if len(exposed) != 0 {
		t.Fatalf("выключенный тумблером сервер не должен передавать инструменты, получили %v", exposed)
	}
	// Включение тумблера восстанавливает соединение (в юнит-тесте процесс
	// жив; в приложении после «стопа» процесс мёртв и Dial не succeed'ится —
	// поднять его может только кнопка «поднять» через mcpboot).
	if err := hub.SetEnabled(ctx, "t", true); err != nil {
		t.Fatalf("toggle on: %v", err)
	}
	if info := hub.Infos()[0]; info.Stopped || !info.Connected {
		t.Fatalf("успешный Dial должен снять «остановлен»: %+v", info)
	}
	if len(exposed) != 1 || exposed[0].Function.Name != "ping" {
		t.Fatalf("после включения модель должна видеть ping, получили %v", exposed)
	}
	text, _, _, _, err := hub.Execute(ctx, "ping", nil)
	if err != nil || text != "pong" {
		t.Fatalf("после включения вызов должен вернуть pong, получили %q, %v", text, err)
	}
}
