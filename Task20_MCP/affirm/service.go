package affirm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"ai-chat-go/affirmations"
)

// Toolkit — логика всех инструментов Affirm MCP и их MCP-привязка.
// Сервер без состояния: все инструменты — чистые вызовы внешних API,
// между вызовами ничего не накапливается.
type Toolkit struct {
	Affirm   *affirmations.Client
	invokers map[string]invoker
	defs     []ToolDef
}

// NewToolkit регистрирует все инструменты Affirm.
func NewToolkit(cl *affirmations.Client) *Toolkit {
	t := &Toolkit{Affirm: cl, invokers: map[string]invoker{}}

	// Имя намеренно совпадает с инструментом Task17_MCP (orbit): оба сервера
	// в одном чате дают потенциальный конфликт имён, который хаб Task16/17/18
	// разрешает префиксом slug__tool → orbit__daily_affirmation /
	// affirm__daily_affirmation.
	t.add("daily_affirmation",
		"Аффирмация дня с affirmations.dev на русском языке со случайным волшебным смайликом. Используйте как доброе напутствие в конце ответа или дополните брифинг vibe_today.",
		func(ctx context.Context, _ json.RawMessage) (any, error) {
			res, err := t.Daily(ctx)
			return res, err
		})

	t.add("translate_text",
		"Перевод произвольного текста через бесплатный эндпоинт Google Translate (gtx, без ключа). По умолчанию en→ru; направление задаётся аргументами source/target (коды языков).",
		func(ctx context.Context, raw json.RawMessage) (any, error) {
			var in TranslateArgs
			if err := decode(raw, &in); err != nil {
				return nil, err
			}
			return t.TranslateText(ctx, in)
		})

	t.add("affirmation_status",
		"Диагностика сервера Affirm: доступность affirmations.dev и переводчика Google Translate (gtx).",
		func(ctx context.Context, _ json.RawMessage) (any, error) {
			return t.Status(ctx), nil
		})

	return t
}

// Daily — daily_affirmation: аффирмация дня (оригинал en + перевод ru)
// со случайным волшебным смайликом. Сбой переводчика не роняет инструмент —
// вернётся оригинал.
func (t *Toolkit) Daily(ctx context.Context) (affirmations.Result, error) {
	return t.Affirm.Russian(ctx)
}

// TranslateArgs — translate_text.
type TranslateArgs struct {
	Text   string `json:"text" jsonschema:"текст для перевода"`
	Source string `json:"source,omitempty" jsonschema:"язык источника, напр. en; пусто — en"`
	Target string `json:"target,omitempty" jsonschema:"язык перевода, напр. ru; пусто — ru"`
}

// TranslateText — translate_text: перевод через Google Translate (gtx).
func (t *Toolkit) TranslateText(ctx context.Context, in TranslateArgs) (map[string]any, error) {
	text := strings.TrimSpace(in.Text)
	if text == "" {
		return nil, fmt.Errorf("пустой текст для перевода")
	}
	out, err := t.Affirm.Translate(ctx, text, in.Source, in.Target)
	if err != nil {
		return nil, err
	}
	sl, tl := in.Source, in.Target
	if strings.TrimSpace(sl) == "" {
		sl = "en"
	}
	if strings.TrimSpace(tl) == "" {
		tl = "ru"
	}
	return map[string]any{
		"translated": out,
		"original":   text,
		"source":     sl,
		"target":     tl,
	}, nil
}

// Status — affirmation_status: доступность провайдеров сервера.
func (t *Toolkit) Status(ctx context.Context) map[string]any {
	return map[string]any{
		"server":    "affirm-mcp",
		"version":   "0.1.0",
		"providers": t.Affirm.Status(ctx),
	}
}
