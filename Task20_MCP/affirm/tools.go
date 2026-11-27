package affirm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolDef — описание инструмента для -list и status-страницы.
type ToolDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type invoker func(ctx context.Context, raw json.RawMessage) (any, error)

// add регистрирует один инструмент.
func (t *Toolkit) add(name, description string, fn invoker) {
	t.defs = append(t.defs, ToolDef{Name: name, Description: description})
	t.invokers[name] = fn
}

// Defs — список описаний (для -list и status-страницы).
func (t *Toolkit) Defs() []ToolDef { return t.defs }

// Invoke вызывает инструмент по имени (используется режимом -probe).
func (t *Toolkit) Invoke(ctx context.Context, name string, raw json.RawMessage) (any, error) {
	fn, ok := t.invokers[name]
	if !ok {
		return nil, fmt.Errorf("неизвестный инструмент %q; доступно: %v", name, names(t.defs))
	}
	return fn(ctx, raw)
}

// Attach регистрирует инструменты на MCP-сервере. Аргументы типизированы:
// SDK сам строит input-схему из json/jsonschema-тегов и валидирует вход.
func (t *Toolkit) Attach(server *mcp.Server) {
	mcp.AddTool(server, t.tool("daily_affirmation"), func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "daily_affirmation", func() (any, error) { return t.Daily(ctx) })
	})
	mcp.AddTool(server, t.tool("translate_text"), func(ctx context.Context, _ *mcp.CallToolRequest, in TranslateArgs) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "translate_text", func() (any, error) { return t.TranslateText(ctx, in) })
	})
	mcp.AddTool(server, t.tool("affirmation_status"), func(ctx context.Context, _ *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
		return t.call(ctx, "affirmation_status", func() (any, error) {
			res := t.Status(ctx)
			return res, nil
		})
	})
}

// tool — описание инструмента по имени.
func (t *Toolkit) tool(name string) *mcp.Tool {
	for _, d := range t.defs {
		if d.Name == name {
			return &mcp.Tool{Name: d.Name, Description: d.Description}
		}
	}
	return &mcp.Tool{Name: name}
}

// call — общий каркас: логика → результат MCP, ошибка → IsError-результат.
func (t *Toolkit) call(ctx context.Context, name string, fn func() (any, error)) (*mcp.CallToolResult, any, error) {
	payload, err := fn()
	if err != nil {
		log.Printf("tool %s: %v", name, err)
		return errorResult(err), nil, nil
	}
	return okResult(payload), payload, nil
}

// decode разбирает JSON-аргументы инструмента.
func decode(raw json.RawMessage, into any) error {
	if len(raw) == 0 {
		return nil
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("неверные аргументы: %w", err)
	}
	return nil
}

// okResult — текстовый результат с компактным JSON-payload.
func okResult(payload any) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		Content: []mcp.Content{&mcp.TextContent{Text: marshal(payload)}},
	}
}

// errorResult — ошибка инструмента (IsError=true): модель увидит текст.
func errorResult(err error) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: "⚠️ " + err.Error()}},
	}
}

// marshal — JSON без лишних экранирований, читаемый в логах.
func marshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf(`{"error":"payload: %v"}`, err)
	}
	return string(b)
}

func names(defs []ToolDef) []string {
	out := make([]string, 0, len(defs))
	for _, d := range defs {
		out = append(out, d.Name)
	}
	return out
}
