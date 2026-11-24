package config

import "testing"

func TestEngineAutostartEnabled(t *testing.T) {
	tests := []struct {
		name           string
		autostart, dir string
		astroURL       string
		want           bool
	}{
		{"auto: dir задан и URL локальный", "auto", "/path/Astrologer-API", "http://127.0.0.1:8000", true},
		{"auto: dir не задан", "auto", "", "http://127.0.0.1:8000", false},
		{"auto: hosted URL", "auto", "/path/Astrologer-API", "https://astrologer.p.rapidapi.com", false},
		{"on: принудительно даже без dir", "on", "", "http://127.0.0.1:8000", true},
		{"off: выключено", "off", "/path/Astrologer-API", "http://127.0.0.1:8000", false},
		{"off в виде 0", "0", "/path", "http://localhost:8000", false},
		{"on в виде 1", "1", "", "http://localhost:8000", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Config{EngineAutostart: tt.autostart, AstroDir: tt.dir, AstroURL: tt.astroURL}
			if got := c.EngineAutostartEnabled(); got != tt.want {
				t.Fatalf("EngineAutostartEnabled() = %v, хочу %v", got, tt.want)
			}
		})
	}
}
