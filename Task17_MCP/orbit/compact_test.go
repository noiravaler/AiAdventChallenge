package orbit

import (
	"strings"
	"testing"
)

// TestAspectsSortingAndLimits — сортировка по орбису, фильтр и обрезка.
func TestAspectsSortingAndLimits(t *testing.T) {
	cd := map[string]any{
		"aspects": []any{
			map[string]any{"p1_name": "Sun", "p2_name": "Moon", "aspect": "conjunction", "orbit": 4.2},
			map[string]any{"p1_name": "Mars", "p2_name": "Venus", "aspect": "square", "orbit": 1.1},
			map[string]any{"p1_name": "Jupiter", "p2_name": "Saturn", "aspect": "trine", "orbit": 2.8},
			map[string]any{"p1_name": "Mercury", "p2_name": "Pluto", "aspect": "opposition", "orbit": 9.9},
			map[string]any{"p1_name": "", "p2_name": "Sun", "aspect": "conjunction", "orbit": 0.5}, // мусор
		},
	}
	got := Aspects(cd, 0, 3)
	if len(got) != 3 {
		t.Fatalf("хотим 3 аспекта, получено %d", len(got))
	}
	if got[0].P1 != "Mars" || got[0].Orbit != 1.1 {
		t.Fatalf("первым должен быть самый тесный аспект (Mars 1.1°), получено %+v", got[0])
	}
	if got[2].P1 != "Sun" {
		t.Fatalf("лишние аспекты не должны попадать в лимит, получено %+v", got[2])
	}

	tight := Aspects(cd, 3.0, 0)
	if len(tight) != 2 {
		t.Fatalf("фильтр орбиса <= 3° должен оставить 2 аспекта, получено %d", len(tight))
	}
}

// TestScoreAndBreakdown — скор совместимости и его расшифровка.
func TestScoreAndBreakdown(t *testing.T) {
	cd := map[string]any{
		"relationship_score": map[string]any{
			"score_value":       18.0,
			"description_str":   "",
			"score_description": "Very Important",
			"is_destiny_sign":   true,
			"score_breakdown": []any{
				map[string]any{"rule": "sun_moon_conjunction", "description": "Sun-Moon conjunction", "points": 11.0},
				map[string]any{"rule": "venus_mars", "description": "Venus-Mars aspect", "points": 7.0},
			},
		},
	}
	score := Score(cd)
	if score["score"] != 18.0 || score["destiny_sign"] != true {
		t.Fatalf("скор разобран неверно: %+v", score)
	}
	bd := Breakdown(cd, 5)
	if len(bd) != 2 || bd[0]["rule"] != "sun_moon_conjunction" {
		t.Fatalf("breakdown должен быть отсортирован по очкам: %+v", bd)
	}
}

// TestParseDateAndTime — строгость парсинга даты/времени рождения.
func TestParseDateAndTime(t *testing.T) {
	y, m, d, err := parseDate("1995-07-21")
	if err != nil || y != 1995 || m != 7 || d != 21 {
		t.Fatalf("дата не распознана: %v %v", []int{y, m, d}, err)
	}
	if _, _, _, err := parseDate("21.07.1995"); err == nil {
		t.Fatal("дата в неверном формате должна давать ошибку")
	}
	hh, mm, err := parseTime("14:30")
	if err != nil || hh != 14 || mm != 30 {
		t.Fatalf("время не распознано: %d:%d %v", hh, mm, err)
	}
	if hh, mm, err := parseTime("09:05:30"); err != nil || hh != 9 || mm != 5 {
		t.Fatalf("HH:MM:SS должен приниматься: %d:%d %v", hh, mm, err)
	}
}

// TestChartBirthLabel — человекочитаемая строка рождения.
func TestChartBirthLabel(t *testing.T) {
	ch := fixtureChart("Anna")
	if got := ch.BirthLabel(); !strings.Contains(got, "Anna") || !strings.Contains(got, "Kazan (RU)") {
		t.Fatalf("неожиданный label: %q", got)
	}
}
