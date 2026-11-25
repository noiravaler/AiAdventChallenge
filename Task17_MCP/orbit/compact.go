package orbit

import (
	"fmt"
	"sort"
	"strconv"
)

// Aspect — компактная проекция аспекта из chart_data Astrologer-API.
type Aspect struct {
	P1     string  `json:"p1"`
	P2     string  `json:"p2"`
	Aspect string  `json:"aspect"`
	Orbit  float64 `json:"orbit"` // орбис в градусах, чем меньше — тем сильнее
}

// mapAny — безопасное приведение к map[string]any.
func mapAny(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// String — человекочитаемая запись аспекта: «Sun trine Moon».
func (a Aspect) String() string {
	return fmt.Sprintf("%s %s %s", a.P1, a.Aspect, a.P2)
}

// ChartData достаёт chart_data из ответа движка.
func ChartData(resp map[string]any) map[string]any {
	return mapAny(resp["chart_data"])
}

// Aspects извлекает аспекты из chart_data: фильтр по орбису, сортировка
// по силе (минимальный орбис первый), обрезка до limit. maxOrbit <= 0 —
// без фильтра.
func Aspects(chartData map[string]any, maxOrbit float64, limit int) []Aspect {
	raw, _ := chartData["aspects"].([]any)
	out := make([]Aspect, 0, len(raw))
	for _, item := range raw {
		m := mapAny(item)
		if m == nil {
			continue
		}
		a := Aspect{
			P1:     str(m["p1_name"]),
			P2:     str(m["p2_name"]),
			Aspect: str(m["aspect"]),
			Orbit:  num(m["orbit"]),
		}
		if a.P1 == "" || a.P2 == "" || a.Aspect == "" {
			continue
		}
		if maxOrbit > 0 && a.Orbit > maxOrbit {
			continue
		}
		out = append(out, a)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Orbit < out[j].Orbit })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// Distribution возвращает распределение стихий/качеств (element_distribution,
// quality_distribution) в исходном виде.
func Distribution(chartData map[string]any, key string) map[string]any {
	d := mapAny(chartData[key])
	if d == nil {
		return map[string]any{}
	}
	return d
}

// Score — скор совместимости из синастрии (метод Ciro Discepolo).
func Score(chartData map[string]any) map[string]any {
	rs := mapAny(chartData["relationship_score"])
	if rs == nil {
		return map[string]any{}
	}
	out := map[string]any{
		"score":        rs["score_value"],
		"description":  rs["score_description"],
		"destiny_sign": rs["is_destiny_sign"],
	}
	return out
}

// Breakdown — расшифровка скора: правила с наибольшим числом очков.
func Breakdown(chartData map[string]any, limit int) []map[string]any {
	rs := mapAny(chartData["relationship_score"])
	if rs == nil {
		return nil
	}
	raw, _ := rs["score_breakdown"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		m := mapAny(item)
		if m == nil {
			continue
		}
		out = append(out, map[string]any{
			"rule":        str(m["rule"]),
			"description": str(m["description"]),
			"points":      m["points"],
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		return num(out[i]["points"]) > num(out[j]["points"])
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// str приводит any к строке (включая числа от JSON-декодера).
func str(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(t)
	default:
		return ""
	}
}

// num приводит any к float64.
func num(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case string:
		f, _ := strconv.ParseFloat(t, 64)
		return f
	default:
		return 0
	}
}
