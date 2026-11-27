// Package config собирает конфигурацию сервера Orbit MCP из .env,
// переменных окружения и флагов (приоритет: флаги > окружение > .env).
package config

import (
	"flag"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Config — все настройки Orbit MCP.
type Config struct {
	// Astro — self-hosted Astrologer-API (github.com/g-battaglia/Astrologer-API).
	AstroURL     string        // базовый URL, напр. http://127.0.0.1:8000
	AstroVersion string        // версия API: v5/v6 (суффикс /api/<version>/…)
	AstroKey     string        // X-RapidAPI-Key — только для hosted-варианта
	AstroTimeout time.Duration // лимит одного запроса к астро-движку

	GeoURL           string        // Open-Meteo Geocoding (город → координаты + IANA-таймзона)
	GeoTimeout       time.Duration // лимит запроса геокодинга
	OpenMeteoURL     string        // Open-Meteo Weather API (погода)
	OpenMeteoTimeout time.Duration // лимит запроса погоды

	DefaultCity string // город по умолчанию (когда не задан city и нет сохранённых карт)
	Lang        string // язык SVG-карт (RU, EN, …)
	Theme       string // тема SVG-карт (classic, dark, …)
	Houses      string // система домов по умолчанию (P = Placidus, W = Whole Sign, …)

	StoreFile string // файл персистентности натальных карт (JSON)
	ChartsDir string // каталог для отрисованных SVG-колёс (chart_render)

	JournalFile string // файл дневника наблюдений (journal_capture/read/summary)
	PulseDB     string // SQLite-база пульса неба (pulse_capture/read/summary)

	// Автозапуск self-hosted движка Astrologer-API.
	AstroDir           string        // каталог клона движка (включает автозапуск в режиме auto)
	EngineAutostart    string        // auto | on | off
	EngineStartTimeout time.Duration // сколько ждать /health после запуска движка
	EngineLog          string        // файл для stdout/stderr движка

	// Режимы запуска (флаги).
	HTTPAddr string // -http: режим Streamable HTTP по этому адресу; пусто = stdio
	List     bool   // -list: показать список инструментов и выйти
	Probe    string // -probe NAME: вызвать инструмент напрямую и выйти
	Args     string // -args JSON: аргументы для -probe
}

// Load читает .env (если есть), окружение и флаги argv.
func Load(argv []string) (*Config, error) {
	loadDotEnv(".env")

	c := &Config{
		AstroURL:         env("ORBIT_ASTRO_URL", "http://127.0.0.1:8000"),
		AstroVersion:     env("ORBIT_ASTRO_VERSION", "v6"),
		AstroKey:         os.Getenv("ORBIT_ASTRO_KEY"),
		AstroTimeout:     envDur("ORBIT_ASTRO_TIMEOUT", 30*time.Second),
		GeoURL:           env("ORBIT_GEO_URL", "https://geocoding-api.open-meteo.com"),
		GeoTimeout:       envDur("ORBIT_GEO_TIMEOUT", 15*time.Second),
		OpenMeteoURL:     env("ORBIT_OPENMETEO_URL", "https://api.open-meteo.com"),
		OpenMeteoTimeout: envDur("ORBIT_OPENMETEO_TIMEOUT", 15*time.Second),
		DefaultCity:      env("ORBIT_DEFAULT_CITY", "Москва"),
		Lang:             env("ORBIT_LANG", "RU"),
		Theme:            env("ORBIT_THEME", "classic"),
		Houses:           env("ORBIT_HOUSES", "P"),
		StoreFile:        env("ORBIT_STORE_FILE", "orbit-charts.json"),
		ChartsDir:        env("ORBIT_CHARTS_DIR", ""),
		JournalFile:      env("ORBIT_JOURNAL_FILE", "orbit-journal.json"),
		PulseDB:          env("ORBIT_PULSE_DB", "orbit-pulse.db"),

		AstroDir:           env("ORBIT_ASTRO_DIR", ""),
		EngineAutostart:    env("ORBIT_ENGINE_AUTOSTART", "auto"),
		EngineStartTimeout: envDur("ORBIT_ENGINE_START_TIMEOUT", 2*time.Minute),
		EngineLog:          env("ORBIT_ENGINE_LOG", "bin/astrologer.log"),
	}

	fs := flag.NewFlagSet("orbit-mcp", flag.ContinueOnError)
	fs.StringVar(&c.HTTPAddr, "http", "", "запустить в режиме Streamable HTTP по адресу (напр. 127.0.0.1:5199); по умолчанию stdio")
	fs.BoolVar(&c.List, "list", false, "показать список инструментов и выйти")
	fs.StringVar(&c.Probe, "probe", "", "вызвать инструмент по имени (без MCP-транспорта) и выйти")
	fs.StringVar(&c.Args, "args", "{}", "JSON-аргументы для -probe")
	if err := fs.Parse(argv); err != nil {
		return nil, err
	}

	c.AstroURL = strings.TrimRight(c.AstroURL, "/")
	c.AstroVersion = "v" + strings.Trim(strings.TrimSpace(c.AstroVersion), "/v ")
	if c.AstroVersion == "v" {
		c.AstroVersion = "v6"
	}
	c.DefaultCity = strings.TrimSpace(c.DefaultCity)
	if c.DefaultCity == "" {
		c.DefaultCity = "Москва"
	}
	if c.ChartsDir == "" {
		c.ChartsDir = filepath.Join(filepath.Dir(c.StoreFile), "charts")
	}
	return c, nil
}

// EngineAutostartEnabled — поднимать ли движок при старте Orbit.
// on — всегда; off — никогда; auto (по умолчанию) — когда задан каталог
// клона (ORBIT_ASTRO_DIR) и ORBIT_ASTRO_URL указывает на localhost.
func (c *Config) EngineAutostartEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(c.EngineAutostart)) {
	case "on", "1", "true", "yes":
		return true
	case "off", "0", "false", "no":
		return false
	}
	return c.AstroDir != "" && isLocalURL(c.AstroURL)
}

// isLocalURL сообщает, указывает ли URL на локальную машину.
func isLocalURL(raw string) bool {
	lower := strings.ToLower(raw)
	for _, marker := range []string{"localhost", "127.0.0.1", "[::1]"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// AstroEndpoint возвращает полный путь к эндпоинту движка, напр.
// http://127.0.0.1:8000/api/v6/context/birth-chart.
func (c *Config) AstroEndpoint(suffix string) string {
	return c.AstroURL + "/api/" + c.AstroVersion + "/" + strings.TrimPrefix(suffix, "/")
}

// env читает переменную окружения с значением по умолчанию.
func env(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// envDur читает длительность (секунды, "30s", "1m" и т.п.).
func envDur(key string, def time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	if secs, err := strconv.ParseFloat(raw, 64); err == nil && secs > 0 {
		return time.Duration(secs * float64(time.Second))
	}
	return def
}

// loadDotEnv подгружает пары KEY=VALUE из файла .env, не перезаписывая
// уже установленные переменные. Ошибки чтения игнорируются: файла может не быть.
func loadDotEnv(path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
}
