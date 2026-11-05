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
)

// Config holds all settings needed to talk to an OpenAI-compatible API.
type Config struct {
	APIKey    string
	BaseURL   string
	Model     string
	System    string
	Timeout   time.Duration
	MaxTokens int
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

	return cfg, nil
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
