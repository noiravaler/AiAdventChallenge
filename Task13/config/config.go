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

	// defaultHistoryFile is the legacy JSON path of the web chat context (now
	// only the migration source); per-model files live next to it inside the
	// dedicated "history" folder (see history.go modelHistoryPath).
	defaultHistoryFile = "history/.chat-history.json"

	// defaultDBFile is the SQLite database that persists the conversation and
	// the long-term memory (AI_DB_FILE). Empty disables the persistence.
	defaultDBFile = "history/chat.db"
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

	// HistoryFile is the path to the legacy JSON context file. It is no longer
	// written to; existing files are migrated into the SQLite database on
	// start. Empty disables nothing on its own (see DBFile).
	HistoryFile string

	// DBFile is the SQLite database that stores the conversation (short-term
	// and working memory) and the long-term memory. Empty disables the
	// persistence entirely.
	DBFile string
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

	// Conversation persistence for the web chat: SQLite by default; override
	// with AI_DB_FILE (empty value disables the persistence). The legacy
	// AI_HISTORY_FILE only names the JSON files to migrate from.
	cfg.HistoryFile = strings.TrimSpace(envOr("AI_HISTORY_FILE", defaultHistoryFile))
	cfg.DBFile = strings.TrimSpace(envOr("AI_DB_FILE", defaultDBFile))

	return cfg, nil
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
