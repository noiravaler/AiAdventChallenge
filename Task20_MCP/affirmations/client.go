// Package affirmations — клиенты бесплатных внешних сервисов для
// MCP-сервера Affirm (Task20_MCP): affirmations.dev (аффирмация дня, без
// ключа) и перевод через бесплатный эндпоинт Google Translate (gtx, тоже
// без ключа). Пакет вынесен из Task17_MCP: там он обслуживал инструмент
// daily_affirmation (и им пользуются journal_*/elect_*), здесь — ядро
// отдельного сервера.
package affirmations

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// magicEmoji — набор волшебных смайликов для аффирмации.
var magicEmoji = []string{"✨", "🌟", "⭐", "💫", "🔮", "🌙", "🌠", "🪄", "🌌", "🎆"}

// Result — аффирмация дня: текст на русском (или оригинал, если перевод
// не удался) и случайный волшебный смайлик.
type Result struct {
	Affirmation string `json:"affirmation"`
	Emoji       string `json:"emoji"`
}

// Client — affirmations.dev + переводчик.
type Client struct {
	base         string
	translateURL string
	hc           *http.Client
}

// New создаёт клиент. translateURL — полный URL эндпоинта перевода
// (по умолчанию Google Translate gtx); пустая строка отключает перевод.
func New(base, translateURL string, timeout time.Duration) *Client {
	return &Client{
		base:         trimRight(base),
		translateURL: trimRight(translateURL),
		hc:           &http.Client{Timeout: timeout},
	}
}

// Russian — случайная аффирмация с affirmations.dev на русском языке.
// Источник отдаёт только английский, поэтому текст переводится через gtx;
// при сбое переводчика (вторичная зависимость) возвращается оригинал.
func (c *Client) Russian(ctx context.Context) (Result, error) {
	original, err := c.fetch(ctx)
	if err != nil {
		return Result{}, err
	}
	affirmation := original
	if c.translateURL != "" {
		if ru, err := c.translate(ctx, original); err == nil {
			affirmation = ru
		}
	}
	return Result{Affirmation: affirmation, Emoji: pickEmoji()}, nil
}

// Translate — перевод произвольного текста через бесплатный эндпоинт
// Google Translate (client=gtx, без ключа). sl/tl — коды языков («en»,
// «ru», …); пустое значение трактуется как en→ru.
func (c *Client) Translate(ctx context.Context, text, sl, tl string) (string, error) {
	if strings.TrimSpace(sl) == "" {
		sl = "en"
	}
	if strings.TrimSpace(tl) == "" {
		tl = "ru"
	}
	q := url.Values{}
	q.Set("client", "gtx")
	q.Set("sl", sl)
	q.Set("tl", tl)
	q.Set("dt", "t")
	q.Set("q", text)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.translateURL+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	raw, err := c.do(req, "переводчик")
	if err != nil {
		return "", err
	}
	return parseGTX(raw)
}

// Status — диагностика доступности обоих провайдеров. Ошибка одного
// сервиса не роняет проверку другого.
func (c *Client) Status(ctx context.Context) map[string]any {
	out := map[string]any{}
	if _, err := c.fetch(ctx); err != nil {
		out["affirmations"] = map[string]any{"url": c.base, "status": "error", "error": err.Error()}
	} else {
		out["affirmations"] = map[string]any{"url": c.base, "status": "ok", "comment": "affirmations.dev, без ключа"}
	}
	switch {
	case c.translateURL == "":
		out["translate"] = map[string]any{"status": "off", "comment": "переводчик не настроен — только оригинал"}
	default:
		if _, err := c.translate(ctx, "hello"); err != nil {
			out["translate"] = map[string]any{"url": c.translateURL, "status": "error", "error": err.Error()}
		} else {
			out["translate"] = map[string]any{"url": c.translateURL, "status": "ok", "comment": "Google Translate (gtx), без ключа"}
		}
	}
	return out
}

// pickEmoji — случайный смайлик из набора.
func pickEmoji() string { return magicEmoji[rand.IntN(len(magicEmoji))] }

// fetch — GET affirmations.dev → {"affirmation": "..."}.
func (c *Client) fetch(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/json")
	raw, err := c.do(req, "аффирмации")
	if err != nil {
		return "", err
	}
	var parsed struct {
		Affirmation string `json:"affirmation"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("некорректный ответ affirmations.dev: %w", err)
	}
	parsed.Affirmation = strings.TrimSpace(parsed.Affirmation)
	if parsed.Affirmation == "" {
		return "", fmt.Errorf("affirmations.dev вернул пустую аффирмацию")
	}
	return parsed.Affirmation, nil
}

// translate — перевод en→ru (направление по умолчанию для аффирмаций).
func (c *Client) translate(ctx context.Context, text string) (string, error) {
	return c.Translate(ctx, text, "en", "ru")
}

// parseGTX разбирает ответ gtx вида [[["перевод","оригинал",…],…],null,"en",…]:
// первый блок — сегменты перевода, склеиваем их в одну строку.
func parseGTX(raw []byte) (string, error) {
	var outer []json.RawMessage
	if err := json.Unmarshal(raw, &outer); err != nil || len(outer) == 0 {
		return "", fmt.Errorf("некорректный ответ переводчика")
	}
	var segments [][]any
	if err := json.Unmarshal(outer[0], &segments); err != nil {
		return "", fmt.Errorf("некорректный блок перевода: %w", err)
	}
	var b strings.Builder
	for _, seg := range segments {
		if len(seg) == 0 {
			continue
		}
		text, ok := seg[0].(string) // [перевод, оригинал, null, null, 10]
		if ok {
			b.WriteString(text)
		}
	}
	out := strings.TrimSpace(b.String())
	if out == "" {
		return "", fmt.Errorf("переводчик вернул пустой результат")
	}
	return out, nil
}

// do — общий каркас запроса: контекст, лимит тела, статус.
func (c *Client) do(req *http.Request, what string) ([]byte, error) {
	res, err := c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d: %s", what, res.StatusCode, firstRunes(string(raw), 120))
	}
	return raw, nil
}

func trimRight(s string) string { return strings.TrimRight(strings.TrimSpace(s), "/") }

func firstRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n]) + "…"
	}
	return string(r)
}
