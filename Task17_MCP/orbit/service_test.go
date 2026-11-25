package orbit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-chat-go/config"
	"ai-chat-go/openmeteo"
)

// testLocation — фиксированный результат геокодинга для тестов.
func testLocation() openmeteo.Location {
	return openmeteo.Location{
		Name: "Kazan", Country: "Russia", CountryCode: "RU",
		Latitude: 55.79, Longitude: 49.11, Timezone: "Europe/Moscow", Population: 1243756,
	}
}

// fakes — фейковые Astrologer-API и Open-Meteo на httptest.
type fakes struct {
	astro    *httptest.Server
	geo      *httptest.Server
	weather  *httptest.Server
	svc      *Service
	mu       sync.Mutex
	lastSubj map[string]any // последнее тело POST в движок (по пути)
	astroHit map[string]int // число запросов по пути
}

func newFakes(t *testing.T) *fakes {
	t.Helper()
	f := &fakes{lastSubj: map[string]any{}, astroHit: map[string]int{}}

	f.astro = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.astroHit[r.URL.Path]++
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.lastSubj[r.URL.Path] = body
		f.mu.Unlock()

		switch {
		case r.URL.Path == "/health":
			_, _ = w.Write([]byte(`{"status":"OK"}`))
		case r.URL.Path == "/api/v6/context/birth-chart":
			writeJSON(w, map[string]any{
				"status":  "OK",
				"context": "<chart_analysis type=\"Natal\"><positions/></chart_analysis>",
				"chart_data": map[string]any{
					"aspects": []any{
						map[string]any{"p1_name": "Sun", "p2_name": "Moon", "aspect": "conjunction", "orbit": 2.5},
						map[string]any{"p1_name": "Venus", "p2_name": "Mars", "aspect": "trine", "orbit": 1.0},
					},
					"element_distribution": map[string]any{"fire": 5.0, "water": 2.0},
					"quality_distribution": map[string]any{"cardinal": 3.0},
				},
			})
		case r.URL.Path == "/api/v6/chart-data/transit":
			writeJSON(w, map[string]any{
				"status": "OK",
				"chart_data": map[string]any{
					"aspects": []any{
						map[string]any{"p1_name": "Transit Jupiter", "p2_name": "Natal Sun", "aspect": "trine", "orbit": 1.4},
					},
				},
			})
		case r.URL.Path == "/api/v6/context/transit":
			writeJSON(w, map[string]any{
				"status":  "OK",
				"context": "<chart_analysis type=\"Transit\"/>",
				"chart_data": map[string]any{
					"aspects": []any{
						map[string]any{"p1_name": "Transit Saturn", "p2_name": "Natal Sun", "aspect": "square", "orbit": 0.8},
					},
				},
			})
		case r.URL.Path == "/api/v6/context/synastry":
			writeJSON(w, map[string]any{
				"status":  "OK",
				"context": "<chart_analysis type=\"Synastry\"/>",
				"chart_data": map[string]any{
					"aspects": []any{
						map[string]any{"p1_name": "Sun", "p2_name": "Moon", "aspect": "conjunction", "orbit": 1.34},
					},
					"relationship_score": map[string]any{
						"score_value":       18.0,
						"score_description": "Very Important",
						"is_destiny_sign":   true,
						"score_breakdown": []any{
							map[string]any{"rule": "sun_moon_conjunction", "description": "Sun-Moon conjunction (high precision)", "points": 11.0},
						},
					},
				},
			})
		case r.URL.Path == "/api/v6/chart/birth-chart" || r.URL.Path == "/api/v6/chart/synastry":
			writeJSON(w, map[string]any{"status": "OK", "chart": "<svg><circle/></svg>"})
		case r.URL.Path == "/api/v6/moon-phase/context":
			writeJSON(w, map[string]any{
				"status": "OK",
				"moon_phase_overview": map[string]any{
					"moon": map[string]any{
						"phase_name": "Waxing Gibbous", "major_phase": "First Quarter", "stage": "waxing",
						"illumination": "68%", "age_days": 9.0, "emoji": "🌔",
						"zodiac":             map[string]any{"moon_sign": "Psc", "sun_sign": "Aqu"},
						"next_lunar_eclipse": map[string]any{"type": "Total Lunar Eclipse", "datestamp": "Sat, 03 Mar 2026"},
					},
					"sun": map[string]any{
						"sunrise_timestamp": "07:52", "sunset_timestamp": "17:20",
						"next_solar_eclipse": map[string]any{"type": "Annular Solar Eclipse", "datestamp": "Sun, 17 Feb 2026"},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.astro.Close)

	f.geo = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/search" {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, map[string]any{"results": []any{testLocation()}})
	}))
	t.Cleanup(f.geo.Close)

	f.weather = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{
			"current": map[string]any{
				"temperature_2m": -3.2, "apparent_temperature": -8.1,
				"weather_code": 3, "wind_speed_10m": 34.0,
			},
			"daily": map[string]any{
				"temperature_2m_max":            []float64{1.0},
				"temperature_2m_min":            []float64{-9.0},
				"precipitation_probability_max": []int{70},
				"precipitation_sum":             []float64{2.4},
			},
		})
	}))
	t.Cleanup(f.weather.Close)

	cfg := &config.Config{
		AstroURL: f.astro.URL, AstroVersion: "v6", AstroTimeout: 5 * time.Second,
		GeoURL: f.geo.URL, GeoTimeout: 2 * time.Second,
		OpenMeteoURL: f.weather.URL, OpenMeteoTimeout: 2 * time.Second,
		DefaultCity: "Москва", Lang: "RU", Theme: "classic", Houses: "P",
		StoreFile: filepath.Join(t.TempDir(), "charts.json"),
		ChartsDir: filepath.Join(t.TempDir(), "svg"),
	}
	f.svc = New(cfg).WithStore(mustStore(t, cfg.StoreFile))
	f.svc.Now = func() time.Time { return time.Date(2026, 2, 10, 9, 0, 0, 0, time.UTC) }
	return f
}

func mustStore(t *testing.T, path string) *Store {
	t.Helper()
	st, err := NewStore(path)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (f *fakes) hits(path string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.astroHit[path]
}

// TestBuildNatalFlow — полный поток: геокодинг → subject → карта → хранилище.
func TestBuildNatalFlow(t *testing.T) {
	f := newFakes(t)
	payload, err := f.svc.BuildNatal(context.Background(), NatalArgs{
		Name: "Anna", BirthDate: "1995-07-21", BirthTime: "14:30", City: "Казань",
	})
	if err != nil {
		t.Fatal(err)
	}

	// Subject уходит в движок с координатами и таймзоной от геокодинга.
	f.mu.Lock()
	subj := mapAny(f.lastSubj["/api/v6/context/birth-chart"])
	s := mapAny(subj["subject"])
	f.mu.Unlock()
	if s["timezone"] != "Europe/Moscow" || s["nation"] != "RU" || s["houses_system_identifier"] != "P" {
		t.Fatalf("subject собран неверно: %v", s)
	}
	if s["latitude"].(float64) != 55.79 || s["longitude"].(float64) != 49.11 {
		t.Fatalf("координаты неверны: %v", s)
	}

	// Аспекты отсортированы по силе, распределения на месте.
	aspects, _ := payload["aspects"].([]Aspect)
	if len(aspects) != 2 || aspects[0].P1 != "Venus" {
		t.Fatalf("аспекты не отсортированы: %+v", aspects)
	}
	if payload["context"] == nil {
		t.Fatal("XML-контекст движка должен попадать в payload")
	}

	// Карта сохранена и стала «последней».
	ch, err := f.svc.Store.Get("")
	if err != nil || ch.Name != "Anna" {
		t.Fatalf("карта не сохранена: %v (%v)", ch, err)
	}

	// Повторный вызов с тем же городом не должен ходить в геокодинг заново.
	if _, err := f.svc.BuildNatal(context.Background(), NatalArgs{
		Name: "Ann2", BirthDate: "1990-01-01", BirthTime: "08:00", City: "Казань",
	}); err != nil {
		t.Fatal(err)
	}
}

// TestTransitTodayAndForecast — транзиты дня и календарь.
func TestTransitTodayAndForecast(t *testing.T) {
	f := newFakes(t)
	st := f.svc.Store
	if err := st.Save(fixtureChart("Anna")); err != nil {
		t.Fatal(err)
	}

	today, err := f.svc.TransitToday(context.Background(), ChartArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if today["chart"] != "Anna" || today["context"] == nil {
		t.Fatalf("транзиты дня неполные: %+v", today)
	}

	fc, err := f.svc.TransitForecast(context.Background(), ForecastArgs{Days: intPtr(3)})
	if err != nil {
		t.Fatal(err)
	}
	rows, _ := fc["days"].([]map[string]any)
	if len(rows) != 3 {
		t.Fatalf("ожидались 3 дня, получено %d", len(rows))
	}
	if rows[0]["date"] != "2026-02-10" || rows[2]["date"] != "2026-02-12" {
		t.Fatalf("даты посчитаны неверно: %v..%v", rows[0]["date"], rows[2]["date"])
	}
	if len(rows[0]["aspects"].([]Aspect)) != 1 {
		t.Fatalf("аспекты дня потерялись: %+v", rows[0]["aspects"])
	}
}

// TestSynastryFlow — скор и аспекты пары.
func TestSynastryFlow(t *testing.T) {
	f := newFakes(t)
	_ = f.svc.Store.Save(fixtureChart("Anna"))
	_ = f.svc.Store.Save(fixtureChart("Bob"))

	payload, err := f.svc.Synastry(context.Background(), SynastryArgs{First: "Anna", Second: "Bob"})
	if err != nil {
		t.Fatal(err)
	}
	score, _ := payload["score"].(map[string]any)
	if score["score"] != 18.0 || score["destiny_sign"] != true {
		t.Fatalf("скор разобран неверно: %+v", score)
	}
	bd, _ := payload["score_breakdown"].([]map[string]any)
	if len(bd) != 1 || bd[0]["rule"] != "sun_moon_conjunction" {
		t.Fatalf("breakdown потерялся: %+v", bd)
	}
}

// TestMoonPhaseCache — фаза Луны кэшируется на день.
func TestMoonPhaseCache(t *testing.T) {
	f := newFakes(t)
	first, err := f.svc.MoonPhase(context.Background(), MoonArgs{City: "Казань"})
	if err != nil {
		t.Fatal(err)
	}
	phase, _ := first["phase"].(map[string]any)
	if phase["name"] != "Waxing Gibbous" || phase["emoji"] != "🌔" {
		t.Fatalf("фаза разобрана неверно: %+v", phase)
	}
	if first["next_lunar_eclipse"] == nil {
		t.Fatal("ближайшее затмение должно попадать в ответ")
	}
	if _, err := f.svc.MoonPhase(context.Background(), MoonArgs{City: "Казань"}); err != nil {
		t.Fatal(err)
	}
	if n := f.hits("/api/v6/moon-phase/context"); n != 1 {
		t.Fatalf("кэш не сработал: %d запросов вместо 1", n)
	}
}

// TestVibeToday — брифинг: погода + Луна + транзиты + подсказки.
func TestVibeToday(t *testing.T) {
	f := newFakes(t)
	_ = f.svc.Store.Save(fixtureChart("Anna"))

	payload, err := f.svc.VibeToday(context.Background(), VibeArgs{City: "Казань"})
	if err != nil {
		t.Fatal(err)
	}
	w, ok := payload["weather"].(openmeteo.Weather)
	if !ok || w.Temperature != -3.2 || w.Description != "облачно" {
		t.Fatalf("погода разобрана неверно: %+v", payload["weather"])
	}
	if _, ok := payload["moon"]; !ok {
		t.Fatal("Луна должна присутствовать в брифинге")
	}
	tr, _ := payload["transits"].(map[string]any)
	if tr["chart"] != "Anna" {
		t.Fatalf("транзиты не привязаны к карте: %+v", tr)
	}
	hints, _ := payload["hints"].([]string)
	joined := fmt.Sprint(hints)
	if !strings.Contains(joined, "зонт") || !strings.Contains(joined, "ветер") {
		t.Fatalf("подсказки про осадки и ветер потерялись: %v", hints)
	}
}

// TestRenderChartNatal — SVG уходит в файл, а не в результат инструмента.
func TestRenderChartNatal(t *testing.T) {
	f := newFakes(t)
	_ = f.svc.Store.Save(fixtureChart("Anna"))

	rr, err := f.svc.RenderChart(context.Background(), RenderArgs{Type: "natal", First: "Anna", Theme: "dark"})
	if err != nil {
		t.Fatal(err)
	}
	// Модель не получает ни SVG, ни путей, ни ссылок.
	for _, key := range []string{"svg", "file", "url"} {
		if _, ok := rr.Model[key]; ok {
			t.Fatalf("ключ %q не должен попадать в результат для модели: %+v", key, rr.Model)
		}
	}
	if bytes, _ := rr.Model["bytes"].(int); bytes == 0 {
		t.Fatalf("размер не вернулся: %+v", rr.Model)
	}
	if !strings.Contains(string(rr.SVG), "<svg") {
		t.Fatal("в SVG-байтах нет разметки <svg")
	}
	// Файл сохранён в галерею сервера под детерминированным именем.
	if rr.FileName != "anna-natal.svg" {
		t.Fatalf("неожиданное имя файла: %s", rr.FileName)
	}
	raw, rerr := os.ReadFile(filepath.Join(f.svc.ChartsDir, rr.FileName))
	if rerr != nil {
		t.Fatal(rerr)
	}
	if !strings.Contains(string(raw), "<svg") {
		t.Fatalf("в файле нет SVG: %s", rr.FileName)
	}
	// Рендер виден в списке для страницы статуса.
	if rs := f.svc.RenderedCharts(); len(rs) != 1 {
		t.Fatalf("ожидался 1 рендер, получено %d", len(rs))
	}

	// Повторный рендер той же карты перезаписывает тот же файл.
	rr2, err := f.svc.RenderChart(context.Background(), RenderArgs{Type: "natal", First: "Anna"})
	if err != nil {
		t.Fatal(err)
	}
	if rr2.FileName != rr.FileName {
		t.Fatalf("имя файла должно быть детерминированным: %s != %s", rr2.FileName, rr.FileName)
	}

	if _, err := f.svc.RenderChart(context.Background(), RenderArgs{Type: "synastry", First: "Anna"}); err == nil {
		t.Fatal("synastry без second должен давать ошибку")
	}
	if _, err := f.svc.RenderChart(context.Background(), RenderArgs{Type: "nope", First: "Anna"}); err == nil {
		t.Fatal("неизвестный type должен давать ошибку")
	}
}

// TestRenderChartSynastryFile — имя файла синастрии содержит обе карты
// (first-kind-second), а second попадает в сводку для модели.
func TestRenderChartSynastryFile(t *testing.T) {
	f := newFakes(t)
	_ = f.svc.Store.Save(fixtureChart("Anna"))
	_ = f.svc.Store.Save(fixtureChart("Bob"))

	sec := "Bob"
	rr, err := f.svc.RenderChart(context.Background(), RenderArgs{Type: "synastry", First: "Anna", Second: &sec})
	if err != nil {
		t.Fatal(err)
	}
	if rr.FileName != "anna-synastry-bob.svg" {
		t.Fatalf("неожиданное имя файла: %s", rr.FileName)
	}
	if second, _ := rr.Model["second"].(string); second != "Bob" {
		t.Fatalf("second не попал в сводку: %+v", rr.Model)
	}
}

// TestStatusHealth — orbit_status видит недоступный движок.
func TestStatusHealth(t *testing.T) {
	f := newFakes(t)
	f.astro.Close() // движок «упал»

	payload, err := f.svc.Status(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	prov, _ := payload["providers"].(map[string]any)
	astro, _ := prov["astrologer"].(map[string]any)
	if astro["status"] != "down" {
		t.Fatalf("недоступный движок должен быть down: %+v", astro)
	}
}

// TestToolkitInvoke — режим -probe: разбор аргументов и неизвестные имена.
func TestToolkitInvoke(t *testing.T) {
	f := newFakes(t)
	tk := NewToolkit(f.svc)

	if _, err := tk.Invoke(context.Background(), "no_such_tool", nil); err == nil {
		t.Fatal("неизвестный инструмент должен давать ошибку")
	}
	if _, err := tk.Invoke(context.Background(), "transit_today", json.RawMessage(`{"days":`)); err == nil {
		t.Fatal("битый JSON должен давать ошибку")
	}
	if _, err := tk.Invoke(context.Background(), "orbit_status", nil); err != nil {
		t.Fatalf("orbit_status должен работать без аргументов: %v", err)
	}
	if got := len(tk.Defs()); got != 16 {
		t.Fatalf("должно быть 16 инструментов, зарегистрировано %d", got)
	}
}

// TestGeocodeNotFound — город, которого нет в геокодинге.
func TestGeocodeNotFound(t *testing.T) {
	f := newFakes(t)
	f.geo.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{})
	})
	_, err := f.svc.resolveLocation(context.Background(), "Атлантида")
	if err == nil || !strings.Contains(err.Error(), "не найден") {
		t.Fatalf("ожидалась ошибка «не найден», получено %v", err)
	}
}

func intPtr(v int) *int { return &v }
