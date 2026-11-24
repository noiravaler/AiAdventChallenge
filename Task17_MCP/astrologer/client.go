// Package astrologer — HTTP-клиент self-hosted Astrologer-API
// (github.com/g-battaglia/Astrologer-API, движок Kerykeion / Swiss Ephemeris).
// Все методы возвращают сырой ответ как map[string]any: форму chart_data
// сервер не нормализует, разбор ответственных полей — в пакете orbit.
package astrologer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Subject — модель участника Astrologer-API. Сервер валидирует строго
// (лишние поля = 422), поэтому omitempty обязателен для опциональных полей.
type Subject struct {
	Name       string  `json:"name"`
	Year       int     `json:"year"`
	Month      int     `json:"month"`
	Day        int     `json:"day"`
	Hour       int     `json:"hour"`
	Minute     int     `json:"minute"`
	Second     int     `json:"second,omitempty"`
	City       string  `json:"city"`
	Nation     string  `json:"nation,omitempty"` // ISO-3166 alpha-2, опционально
	Longitude  float64 `json:"longitude"`
	Latitude   float64 `json:"latitude"`
	Timezone   string  `json:"timezone"`                 // IANA, напр. Europe/Moscow
	ZodiacType string  `json:"zodiac_type"`              // Tropical | Sidereal
	Houses     string  `json:"houses_system_identifier"` // P = Placidus и т.д.
}

// Client — тонкий клиент к REST-движку.
type Client struct {
	base    string
	version string
	key     string // X-RapidAPI-Key для hosted-варианта; пусто для self-host
	hc      *http.Client
}

// New создаёт клиент. base — без хвостового «/», version — «v6».
func New(base, version, key string, timeout time.Duration) *Client {
	return &Client{
		base:    strings.TrimRight(base, "/"),
		version: strings.TrimPrefix(version, "v"),
		key:     key,
		hc:      &http.Client{Timeout: timeout},
	}
}

// Health проверяет доступность движка (GET /health).
func (c *Client) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/health", nil)
	if err != nil {
		return err
	}
	c.auth(req)
	res, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("статус %d", res.StatusCode)
	}
	return nil
}

// BirthChartContext — натальная карта: POST /api/vN/context/birth-chart.
// Возвращает {status, context (XML для LLM), chart_data}.
func (c *Client) BirthChartContext(ctx context.Context, s *Subject) (map[string]any, error) {
	return c.post(ctx, "context/birth-chart", map[string]any{"subject": s})
}

// BirthChartSVG — натальная карта с рендером: POST /api/vN/chart/birth-chart.
// opts — дополнительные параметры рендера (theme, language, style, custom_title).
func (c *Client) BirthChartSVG(ctx context.Context, s *Subject, opts map[string]any) (map[string]any, error) {
	body := map[string]any{"subject": s}
	merge(body, opts)
	return c.post(ctx, "chart/birth-chart", body)
}

// TransitContext — транзиты к наталу: POST /api/vN/context/transit.
func (c *Client) TransitContext(ctx context.Context, natal, transit *Subject) (map[string]any, error) {
	return c.post(ctx, "context/transit", map[string]any{
		"first_subject":   natal,
		"transit_subject": transit,
	})
}

// TransitData — транзиты без рендера: POST /api/vN/chart-data/transit.
func (c *Client) TransitData(ctx context.Context, natal, transit *Subject) (map[string]any, error) {
	return c.post(ctx, "chart-data/transit", map[string]any{
		"first_subject":   natal,
		"transit_subject": transit,
	})
}

// SynastryContext — синастрия пары: POST /api/vN/context/synastry
// (со скором совместимости и попаданием планет в дома партнёра).
func (c *Client) SynastryContext(ctx context.Context, first, second *Subject) (map[string]any, error) {
	return c.post(ctx, "context/synastry", map[string]any{
		"first_subject":              first,
		"second_subject":             second,
		"include_house_comparison":   true,
		"include_relationship_score": true,
	})
}

// SynastrySVG — синастрия с рендером: POST /api/vN/chart/synastry.
func (c *Client) SynastrySVG(ctx context.Context, first, second *Subject, opts map[string]any) (map[string]any, error) {
	body := map[string]any{
		"first_subject":  first,
		"second_subject": second,
	}
	merge(body, opts)
	return c.post(ctx, "chart/synastry", body)
}

// TransitSVG — транзиты с рендером: POST /api/vN/chart/transit.
func (c *Client) TransitSVG(ctx context.Context, natal, transit *Subject, opts map[string]any) (map[string]any, error) {
	body := map[string]any{
		"first_subject":   natal,
		"transit_subject": transit,
	}
	merge(body, opts)
	return c.post(ctx, "chart/transit", body)
}

// SolarReturnContext — соляр на год: POST /api/vN/context/solar-return.
func (c *Client) SolarReturnContext(ctx context.Context, s *Subject, year int) (map[string]any, error) {
	return c.post(ctx, "context/solar-return", map[string]any{
		"subject": s,
		"year":    year,
	})
}

// MoonPhaseContext — фаза Луны + XML-контекст: POST /api/vN/moon-phase/context.
// Запрос без subject-обёртки: только дата/время и координаты.
func (c *Client) MoonPhaseContext(ctx context.Context, at time.Time, latitude, longitude float64, timezone string) (map[string]any, error) {
	return c.post(ctx, "moon-phase/context", map[string]any{
		"year":      at.Year(),
		"month":     int(at.Month()),
		"day":       at.Day(),
		"hour":      at.Hour(),
		"minute":    at.Minute(),
		"second":    0,
		"latitude":  latitude,
		"longitude": longitude,
		"timezone":  timezone,
	})
}

// post выполняет POST c JSON-телом и декодирует ответ.
// Ошибкой считается и HTTP != 200, и {"status":"ERROR"} в теле.
func (c *Client) post(ctx context.Context, suffix string, body any) (map[string]any, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := c.base + "/api/v" + c.version + "/" + strings.TrimPrefix(suffix, "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.auth(req)

	res, err := c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	payload, err := io.ReadAll(io.LimitReader(res.Body, 32<<20))
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, fmt.Errorf("некорректный ответ движка (HTTP %d): %s", res.StatusCode, firstRunes(string(payload), 200))
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", res.StatusCode, errMessage(out, firstRunes(string(payload), 200)))
	}
	if st, _ := out["status"].(string); st == "ERROR" {
		return nil, fmt.Errorf("%s", errMessage(out, "движок вернул ERROR"))
	}
	return out, nil
}

// auth добавляет RapidAPI-заголовки, когда задан ключ (hosted-вариант).
func (c *Client) auth(req *http.Request) {
	if c.key == "" {
		return
	}
	req.Header.Set("X-RapidAPI-Key", c.key)
	req.Header.Set("X-RapidAPI-Host", req.URL.Host)
}

// errMessage достаёт сообщение об ошибке из типовых форм ответа API.
func errMessage(out map[string]any, def string) string {
	if m, ok := out["message"].(string); ok && m != "" {
		return m
	}
	if d, ok := out["detail"]; ok {
		if b, err := json.Marshal(d); err == nil {
			return string(b)
		}
	}
	return def
}

// firstRunes обрезает строку для безопасного включения в сообщение об ошибке.
func firstRunes(s string, n int) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) > n {
		return string(r[:n]) + "…"
	}
	return string(r)
}

// merge добавляет пары из src в dst (без перезаписи непустых значений dst).
func merge(dst, src map[string]any) {
	for k, v := range src {
		if v != nil {
			dst[k] = v
		}
	}
}
