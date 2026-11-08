// Package config loads chat settings from environment variables and a .env file.
package config

import (
	"bufio"
	"encoding/json"
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

	// DefaultSmallModel is the low-cost "Маленькая модель" (GPT-5 Nano) served
	// by the provider defined in the OpenCode config.
	DefaultSmallModel = "gpt-5-nano"
	// DefaultFlagshipModel is the top-tier "Флагманская модель" (DeepSeek-V4-Pro)
	// served by the reserve provider.
	DefaultFlagshipModel = "deepseek-v4-pro"
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

	// Small is the "Маленькая модель" (GPT-5 Nano): a low-cost provider whose
	// settings (base URL, API key, model id) come from the OpenCode config.
	SmallBaseURL string
	SmallAPIKey  string
	SmallModel   string

	// FlagshipModel is the "Флагманская модель" (DeepSeek-V4-Pro), served by
	// the reserve provider.
	FlagshipModel string
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

	// Flagship model (DeepSeek-V4-Pro) — served by the reserve provider.
	cfg.FlagshipModel = strings.TrimSpace(envOr("AI_FLAGSHIP_MODEL", DefaultFlagshipModel))
	if cfg.FlagshipModel == "" {
		cfg.FlagshipModel = DefaultFlagshipModel
	}

	// Small model (GPT-5 Nano): settings + apiKey come from the OpenCode config.
	// Explicit AI_SMALL_* env vars win over the OpenCode config.
	cfg.SmallBaseURL = strings.TrimSpace(os.Getenv("AI_SMALL_BASE_URL"))
	cfg.SmallAPIKey = strings.TrimSpace(os.Getenv("AI_SMALL_API_KEY"))
	cfg.SmallModel = strings.TrimSpace(os.Getenv("AI_SMALL_MODEL"))
	if sb, sak, sm, found := smallModelFromOpencode(); found {
		if cfg.SmallBaseURL == "" {
			cfg.SmallBaseURL = strings.TrimRight(strings.TrimSpace(sb), "/")
		}
		if cfg.SmallAPIKey == "" {
			cfg.SmallAPIKey = sak
		}
		if cfg.SmallModel == "" {
			cfg.SmallModel = sm
		}
	}
	if cfg.SmallModel == "" {
		cfg.SmallModel = DefaultSmallModel
	}

	return cfg, nil
}

// smallModelFromOpencode extracts the small (GPT-5 Nano) model settings from the
// OpenCode config file. It scans the providers for the "gpt-5-nano" model, then
// reads the provider's baseURL/apiKey (either at provider level or in
// "options"), resolving "{env:NAME}" API-key placeholders. Returns the resolved
// base URL, API key and model id, and whether the model was found.
func smallModelFromOpencode() (baseURL, apiKey, modelID string, ok bool) {
	path, err := opencodeConfigPath()
	if err != nil {
		return "", "", "", false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", "", "", false
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", "", "", false
	}
	providers, _ := cfg["provider"].(map[string]any)
	for _, pv := range providers {
		p, _ := pv.(map[string]any)
		if p == nil {
			continue
		}
		models, _ := p["models"].(map[string]any)
		if models == nil {
			continue
		}
		if _, has := models[DefaultSmallModel]; !has {
			continue
		}
		baseURL = nestedString(p, "baseURL")
		if baseURL == "" {
			baseURL = nestedString(p, "options", "baseURL")
		}
		apiKey = nestedString(p, "apiKey")
		if apiKey == "" {
			apiKey = nestedString(p, "options", "apiKey")
		}
		apiKey = expandEnvRef(apiKey)
		return strings.TrimRight(strings.TrimSpace(baseURL), "/"), apiKey, DefaultSmallModel, true
	}
	return "", "", "", false
}

// opencodeConfigPath returns the OpenCode config path, honoring OPENCODE_CONFIG
// or defaulting to ~/.config/opencode/opencode.json.
func opencodeConfigPath() (string, error) {
	if p := strings.TrimSpace(os.Getenv("OPENCODE_CONFIG")); p != "" {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "opencode", "opencode.json"), nil
}

// expandEnvRef resolves OpenCode's "{env:NAME}" API-key placeholders.
func expandEnvRef(v string) string {
	if strings.HasPrefix(v, "{env:") && strings.HasSuffix(v, "}") {
		return os.Getenv(strings.TrimSuffix(strings.TrimPrefix(v, "{env:"), "}"))
	}
	return v
}

// nestedString reads a string value following the given key path.
func nestedString(m map[string]any, keys ...string) string {
	cur := m
	for i, k := range keys {
		v, ok := cur[k]
		if !ok {
			return ""
		}
		if i == len(keys)-1 {
			s, _ := v.(string)
			return s
		}
		next, ok := v.(map[string]any)
		if !ok {
			return ""
		}
		cur = next
	}
	return ""
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
