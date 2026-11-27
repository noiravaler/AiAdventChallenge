package config

import (
	"testing"
	"time"
)

// TestDefaults — дефолтная конфигурация без окружения и флагов.
func TestDefaults(t *testing.T) {
	c, err := Load([]string{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.AffirmURL != "https://www.affirmations.dev" {
		t.Errorf("AffirmURL = %q", c.AffirmURL)
	}
	if c.AffirmTimeout != 10*time.Second {
		t.Errorf("AffirmTimeout = %v", c.AffirmTimeout)
	}
	if c.TranslateURL != "https://translate.googleapis.com/translate_a/single" {
		t.Errorf("TranslateURL = %q", c.TranslateURL)
	}
	if c.HTTPAddr != "" {
		t.Errorf("по умолчанию режим stdio, HTTPAddr = %q", c.HTTPAddr)
	}
}

// TestFlags — флаги перекрывают значения по умолчанию.
func TestFlags(t *testing.T) {
	c, err := Load([]string{"-http", "127.0.0.1:5200", "-probe", "daily_affirmation", "-args", `{"text":"hi"}`})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.HTTPAddr != "127.0.0.1:5200" {
		t.Errorf("HTTPAddr = %q", c.HTTPAddr)
	}
	if c.Probe != "daily_affirmation" {
		t.Errorf("Probe = %q", c.Probe)
	}
	if c.Args != `{"text":"hi"}` {
		t.Errorf("Args = %q", c.Args)
	}
}

// TestEnvOverride — переменные окружения перекрывают дефолты.
func TestEnvOverride(t *testing.T) {
	t.Setenv("AFFIRM_URL", "http://127.0.0.1:9/src")
	t.Setenv("AFFIRM_TIMEOUT", "3s")
	t.Setenv("AFFIRM_TRANSLATE_URL", "http://127.0.0.1:9/translate_a/single")
	c, err := Load([]string{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.AffirmURL != "http://127.0.0.1:9/src" {
		t.Errorf("AffirmURL = %q", c.AffirmURL)
	}
	if c.AffirmTimeout != 3*time.Second {
		t.Errorf("AffirmTimeout = %v", c.AffirmTimeout)
	}
	if c.TranslateURL != "http://127.0.0.1:9/translate_a/single" {
		t.Errorf("TranslateURL = %q", c.TranslateURL)
	}
}
