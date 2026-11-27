// Package config собирает конфигурацию MCP-сервера Affirm из .env,
// переменных окружения и флагов (приоритет: флаги > окружение > .env).
package config

import (
	"flag"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config — все настройки Affirm MCP.
type Config struct {
	AffirmURL     string        // affirmations.dev — аффирмация дня (без ключа)
	AffirmTimeout time.Duration // лимит запроса аффирмаций и перевода
	TranslateURL  string        // Google Translate (gtx) — перевод на русский (без ключа)

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
		AffirmURL:     env("AFFIRM_URL", "https://www.affirmations.dev"),
		AffirmTimeout: envDur("AFFIRM_TIMEOUT", 10*time.Second),
		TranslateURL:  env("AFFIRM_TRANSLATE_URL", "https://translate.googleapis.com/translate_a/single"),
	}

	fs := flag.NewFlagSet("affirm-mcp", flag.ContinueOnError)
	fs.StringVar(&c.HTTPAddr, "http", "", "запустить в режиме Streamable HTTP по адресу (напр. 127.0.0.1:5200); по умолчанию stdio")
	fs.BoolVar(&c.List, "list", false, "показать список инструментов и выйти")
	fs.StringVar(&c.Probe, "probe", "", "вызвать инструмент по имени (без MCP-транспорта) и выйти")
	fs.StringVar(&c.Args, "args", "{}", "JSON-аргументы для -probe")
	if err := fs.Parse(argv); err != nil {
		return nil, err
	}
	return c, nil
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
