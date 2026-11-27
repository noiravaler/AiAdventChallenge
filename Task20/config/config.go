// Package config loads chat settings from environment variables and a .env file.
package config

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	DefaultBaseURL = "https://api.openai.com/v1"
	DefaultModel   = "gpt-4o-mini"
	DefaultTimeout = 5 * time.Minute

	// DefaultReserveBaseURL is the fallback provider used by the [reserve]
	// model: an external OpenAI-compatible endpoint independent of the main
	// GPUStack API (OpenCode Zen).
	DefaultReserveBaseURL = "https://opencode.ai/zen/v1"
	// DefaultReserveModel is the free model of the reserve provider.
	DefaultReserveModel = "glm-5.3-flash-free"
)

// DefaultModels is the chat model list used when AI_MODELS is not set.
var DefaultModels = []string{
	"glm-5.3-flash",
	"default-chat",
	"default-coding",
	"default-pr-review",
	"gpt-4o",
	"qwen3.8-27b",
	"deepseek-v4-flash",
	"deepseek-v4.1-flash",
}

// MCPServer describes one remote MCP endpoint in the chat configuration.
type MCPServer struct {
	Slug string // латинский идентификатор (префикс имён инструментов, API)
	Name string // отображаемое название в UI
	URL  string // endpoint (Streamable HTTP)
}

// DefaultMCPServers is the MCP server list used when AI_MCP_URLS is not set.
var DefaultMCPServers = []MCPServer{
	{Slug: "dubai", Name: "Dubai Grocery Prices", URL: "https://prices.brainy.ae/mcp"},
	// Локальные серверы: mcpboot сам их поднимает при старте приложения.
	{Slug: "orbit", Name: "Orbit MCP", URL: "http://127.0.0.1:5199/mcp"},  // Task17_MCP
	{Slug: "affirm", Name: "Affirm MCP", URL: "http://127.0.0.1:5200/mcp"}, // Task20_MCP
}

// knownMCPServerNames maps slugs to display names for user-supplied lists.
var knownMCPServerNames = map[string]string{
	"dubai":  "Dubai Grocery Prices",
	"orbit":  "Orbit MCP",
	"affirm": "Affirm MCP",
}

// Config holds all settings needed to talk to an OpenAI-compatible API.
type Config struct {
	APIKey    string
	BaseURL   string
	Model     string
	System    string
	Timeout   time.Duration
	MaxTokens int
	Models    []string

	// Reserve is an independent fallback provider bypassing the main API: when
	// the main provider is unavailable, the chat can switch to it.
	ReserveBaseURL string
	ReserveAPIKey  string
	ReserveModel   string

	// MCPServers is the list of remote MCP endpoints (empty = MCP disabled).
	MCPServers []MCPServer
	// MCPTimeout bounds a single MCP tool call (and the session handshake).
	MCPTimeout time.Duration

	// MCPAutostart enables booting local MCP servers at chat startup: if an
	// endpoint does not answer, the app spawns it from its directory. Only the
	// "orbit" slug (Task17_MCP) and the "affirm" slug (Task20_MCP) are
	// supported; orbit in turn boots the Astrologer-API engine itself.
	MCPAutostart    bool
	MCPDir          string        // каталог Task17_MCP (автодетект рядом с приложением)
	MCPBin          string        // готовый бинарник; пусто → go run ./app
	MCPStartTimeout time.Duration // сколько ждать готовности сервера
	MCPLog          string        // файл логов сервера (относительные пути — внутри MCPDir)

	// Affirm (Task20_MCP) имеет собственные каталог, бинарник и лог: параметры
	// независимы от orbit, чтобы серверы можно было поднимать по отдельности.
	MCPAffirmDir  string // каталог Task20_MCP (автодетект рядом с приложением)
	MCPAffirmBin  string // готовый бинарник; пусто → go run ./app
	MCPAffirmLog  string // файл логов сервера (относительные пути — внутри MCPAffirmDir)

	// Scheduler runs scheduled agent jobs in the background (a ticker
	// goroutine inside this process): jobs fire only while the app is up,
	// missed runs are skipped. Jobs persist to the SQLite database
	// SchedulerDB.
	Scheduler   bool
	SchedulerDB string
}

// Load reads .env (if present) and then the environment variables.
func Load() (Config, error) {
	if err := loadDotEnv(".env"); err != nil {
		return Config{}, fmt.Errorf("read .env: %w", err)
	}

	cfg := Config{
		APIKey:  strings.TrimSpace(os.Getenv("AI_API_KEY")),
		BaseURL: strings.TrimRight(strings.TrimSpace(envOr("AI_BASE_URL", DefaultBaseURL)), "/"),
		Model:   strings.TrimSpace(envOr("AI_MODEL", DefaultModel)),
		System:  strings.TrimSpace(os.Getenv("AI_SYSTEM")),
		Timeout: DefaultTimeout,
	}

	if v := strings.TrimSpace(os.Getenv("AI_TIMEOUT")); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return Config{}, fmt.Errorf("AI_TIMEOUT: %w", err)
		}
		if d <= 0 {
			return Config{}, errors.New("AI_TIMEOUT must be positive")
		}
		cfg.Timeout = d
	}

	if v := strings.TrimSpace(os.Getenv("AI_MAX_TOKENS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return Config{}, errors.New("AI_MAX_TOKENS must be a non-negative integer")
		}
		cfg.MaxTokens = n
	}

	cfg.Models = parseModelList(envOr("AI_MODELS", strings.Join(DefaultModels, ",")))
	if len(cfg.Models) == 0 {
		cfg.Models = append([]string(nil), DefaultModels...)
	}

	// Reserve provider (active when both a URL and a key are configured).
	cfg.ReserveBaseURL = strings.TrimRight(strings.TrimSpace(envOr("AI_RESERVE_BASE_URL", DefaultReserveBaseURL)), "/")
	cfg.ReserveAPIKey = strings.TrimSpace(os.Getenv("AI_RESERVE_API_KEY"))
	cfg.ReserveModel = strings.TrimSpace(envOr("AI_RESERVE_MODEL", DefaultReserveModel))
	if cfg.ReserveModel == "" {
		cfg.ReserveModel = DefaultReserveModel
	}

	// Remote MCP servers (optional; empty disables MCP integration).
	// Format: comma-separated slug=URL pairs, e.g.
	//   AI_MCP_URLS="dubai=https://prices.brainy.ae/mcp,orbit=http://127.0.0.1:5199/mcp"
	cfg.MCPServers = parseMCPServers(os.Getenv("AI_MCP_URLS"))

	// Per-call MCP timeout (default 30s): a hanging or unreachable server must
	// not stall the chat for the whole request timeout.
	cfg.MCPTimeout = 30 * time.Second
	if v := strings.TrimSpace(os.Getenv("AI_MCP_TIMEOUT")); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return Config{}, errors.New("AI_MCP_TIMEOUT must be a positive duration, e.g. 30s")
		}
		cfg.MCPTimeout = d
	}

	// Local MCP autostart (see mcpboot): disable with AI_MCP_AUTOSTART=0.
	cfg.MCPAutostart = envBool("AI_MCP_AUTOSTART", true)
	cfg.MCPDir = envOr("AI_MCP_DIR", detectMCPDir("Task17_MCP"))
	cfg.MCPBin = strings.TrimSpace(os.Getenv("AI_MCP_BIN"))
	cfg.MCPStartTimeout = 90 * time.Second
	if v := strings.TrimSpace(os.Getenv("AI_MCP_START_TIMEOUT")); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return Config{}, errors.New("AI_MCP_START_TIMEOUT must be a positive duration, e.g. 90s")
		}
		cfg.MCPStartTimeout = d
	}
	cfg.MCPLog = envOr("AI_MCP_LOG", "bin/orbit-mcp.log")

	// Affirm MCP (Task20_MCP): отдельные каталог/бинарник/лог, тот же таймаут
	// старта, что и у orbit.
	cfg.MCPAffirmDir = envOr("AI_MCP_AFFIRM_DIR", detectMCPDir("Task20_MCP"))
	cfg.MCPAffirmBin = strings.TrimSpace(os.Getenv("AI_MCP_AFFIRM_BIN"))
	cfg.MCPAffirmLog = envOr("AI_MCP_AFFIRM_LOG", "bin/affirm-mcp.log")

	// Background scheduler (see package scheduler): disable with
	// AI_SCHEDULER=0. Jobs persist to the SQLite database AI_SCHEDULER_DB.
	cfg.Scheduler = envBool("AI_SCHEDULER", true)
	cfg.SchedulerDB = envOr("AI_SCHEDULER_DB", "scheduler.db")

	return cfg, nil
}

// detectMCPDir looks for the given server directory next to the app
// (./<name> or ../<name>); an explicit AI_MCP_DIR / AI_MCP_AFFIRM_DIR wins. A
// candidate counts only if it contains go.mod (guards against stray empty
// directories), and the result is resolved to an absolute path so spawned
// processes are independent of the app's working directory.
func detectMCPDir(name string) string {
	for _, cand := range []string{name, filepath.Join("..", name)} {
		if isServerDir(cand) {
			if abs, err := filepath.Abs(cand); err == nil {
				return abs
			}
			return cand
		}
	}
	return ""
}

// isServerDir reports whether dir looks like the MCP server home: it exists
// and contains go.mod (spurious or empty directories are rejected).
func isServerDir(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, "go.mod"))
	return err == nil && !st.IsDir()
}

func envBool(key string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "0", "off", "false", "no":
		return false
	case "1", "on", "true", "yes":
		return true
	}
	return fallback
}

// parseMCPServers parses a comma-separated list of slug=URL pairs; an empty
// value yields the default server list. Unknown slugs get a capitalised name.
func parseMCPServers(v string) []MCPServer {
	v = strings.TrimSpace(v)
	if v == "" {
		return append([]MCPServer(nil), DefaultMCPServers...)
	}
	var out []MCPServer
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		slug, url, _ := strings.Cut(part, "=")
		slug, url = strings.TrimSpace(slug), strings.TrimRight(strings.TrimSpace(url), "/")
		if slug == "" || url == "" {
			continue
		}
		name := knownMCPServerNames[strings.ToLower(slug)]
		if name == "" {
			name = strings.ToUpper(slug[:1]) + slug[1:]
		}
		out = append(out, MCPServer{Slug: slug, Name: name, URL: url})
	}
	return out
}

// ReserveConfigured reports whether the reserve provider can actually serve
// requests: a key is present (or the endpoint is local and needs none).
func (c Config) ReserveConfigured() bool {
	return c.ReserveAPIKey != "" || isLocalURL(c.ReserveBaseURL)
}

// Validate checks that the configuration can be used for a request.
func (c Config) Validate() error {
	if c.BaseURL == "" {
		return errors.New("base URL is empty")
	}
	if c.Model == "" {
		return errors.New("model is empty")
	}
	if c.Timeout <= 0 {
		return errors.New("timeout must be positive")
	}
	if c.APIKey == "" && !isLocalURL(c.BaseURL) {
		return errors.New("API key is missing: set AI_API_KEY (or point AI_BASE_URL at a local server)")
	}
	return nil
}

func isLocalURL(raw string) bool {
	lower := strings.ToLower(raw)
	for _, marker := range []string{"localhost", "127.0.0.1", "[::1]", "0.0.0.0"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func envOr(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

// parseModelList splits a comma-separated model list after trimming spaces.
func parseModelList(v string) []string {
	var out []string
	for _, part := range strings.Split(v, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// loadDotEnv reads simple KEY=VALUE lines. Existing environment variables win.
func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') ||
				(value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}
	return scanner.Err()
}
