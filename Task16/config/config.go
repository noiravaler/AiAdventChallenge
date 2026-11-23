// Package config loads chat settings from environment variables and a .env file.
package config

import (
	"bufio"
	"errors"
	"fmt"
	"os"
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
	{Slug: "vkusvill", Name: "ВкусВилл", URL: "https://mcp.vkusvill.ru/mcp"},
	{Slug: "dubai", Name: "Dubai Grocery Prices", URL: "https://prices.brainy.ae/mcp"},
	{Slug: "supershopping", Name: "SuperShopping", URL: "https://supershopping-mcp.atdev.ai/mcp"},
}

// knownMCPServerNames maps slugs to display names for user-supplied lists.
var knownMCPServerNames = map[string]string{
	"vkusvill":      "ВкусВилл",
	"dubai":         "Dubai Grocery Prices",
	"supershopping": "SuperShopping",
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
	//   AI_MCP_URLS="vkusvill=https://mcp.vkusvill.ru/mcp,dubai=https://prices.brainy.ae/mcp"
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

	return cfg, nil
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
