package orbit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"ai-chat-go/astrologer"
	"ai-chat-go/config"
	"ai-chat-go/openmeteo"
)

// Service — логика всех инструментов Orbit MCP. Методы возвращают payload
// (map/срез), который уходит агенту как JSON; thin-обёртки в tools.go
// превращают их в MCP-результаты.
type Service struct {
	Cfg     *config.Config
	Astro   *astrologer.Client
	Geo     *openmeteo.GeoClient
	Weather *openmeteo.WeatherClient
	Store   *Store
	Cache   *Cache
	Now     func() time.Time // для тестов; nil → time.Now

	ChartsDir string // каталог для отрисованных SVG (пусто — рендер в файл отключён)
	PublicURL string // базовый URL HTTP-режима для ссылок в результатах; пусто = stdio
}

// New собирает сервис из конфигурации.
func New(cfg *config.Config) *Service {
	return &Service{
		Cfg:       cfg,
		Astro:     astrologer.New(cfg.AstroURL, cfg.AstroVersion, cfg.AstroKey, cfg.AstroTimeout),
		Geo:       openmeteo.NewGeoClient(cfg.GeoURL, cfg.GeoTimeout),
		Weather:   openmeteo.NewWeatherClient(cfg.OpenMeteoURL, cfg.OpenMeteoTimeout),
		Store:     &Store{},
		Cache:     NewCache(),
		Now:       time.Now,
		ChartsDir: cfg.ChartsDir,
	}
}

// WithStore подменяет хранилище (тесты, нестандартный путь).
func (s *Service) WithStore(st *Store) *Service {
	s.Store = st
	return s
}

// ---------- Аргументы инструментов ----------

// NatalArgs — natal_build_chart.
type NatalArgs struct {
	Name        string `json:"name" jsonschema:"имя для сохранения карты, напр. Anna"`
	BirthDate   string `json:"birth_date" jsonschema:"дата рождения YYYY-MM-DD"`
	BirthTime   string `json:"birth_time" jsonschema:"местное время рождения HH:MM"`
	City        string `json:"city" jsonschema:"место рождения — город"`
	HouseSystem string `json:"house_system,omitempty" jsonschema:"система домов: P (Placidus, по умолчанию), W (Whole Sign), K (Koch), R (Regiomontanus)"`
}

// ChartArgs — инструменты, выбирающие сохранённую карту (transit_today,
// transit_forecast, solar_return, chart_render).
type ChartArgs struct {
	Name string `json:"name,omitempty" jsonschema:"имя сохранённой карты; пусто — последняя построенная"`
}

// ForecastArgs — transit_forecast.
type ForecastArgs struct {
	Name string `json:"name,omitempty" jsonschema:"имя сохранённой карты; пусто — последняя построенная"`
	Days *int   `json:"days,omitempty" jsonschema:"на сколько дней вперёд (1–31, по умолчанию 7)"`
}

// SynastryArgs — synastry_compare.
type SynastryArgs struct {
	First  string `json:"first" jsonschema:"имя первой карты"`
	Second string `json:"second" jsonschema:"имя второй карты"`
}

// SolarArgs — solar_return.
type SolarArgs struct {
	Name string `json:"name,omitempty" jsonschema:"имя сохранённой карты; пусто — последняя построенная"`
	Year *int   `json:"year,omitempty" jsonschema:"год соляра (по умолчанию текущий)"`
}

// MoonArgs — moon_phase.
type MoonArgs struct {
	Date string `json:"date,omitempty" jsonschema:"дата YYYY-MM-DD; пусто — сегодня"`
	City string `json:"city,omitempty" jsonschema:"город; пусто — город последней карты"`
}

// VibeArgs — vibe_today.
type VibeArgs struct {
	City string `json:"city,omitempty" jsonschema:"город; пусто — город последней карты"`
}

// RenderArgs — chart_render.
type RenderArgs struct {
	Type   string  `json:"type" jsonschema:"тип карты: natal, transit или synastry"`
	First  string  `json:"first" jsonschema:"имя первой (натальной) карты"`
	Second *string `json:"second,omitempty" jsonschema:"имя второй карты (для synastry обязательно)"`
	Theme  string  `json:"theme,omitempty" jsonschema:"тема: classic (по умолчанию), dark, light, strawberry"`
	Style  string  `json:"style,omitempty" jsonschema:"стиль колеса: classic или modern"`
	Title  string  `json:"title,omitempty" jsonschema:"заголовок на карте (до 40 символов)"`
}

// ---------- Вспомогательные ----------

func (s *Service) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// resolveLocation определяет место: явный city → город последней карты →
// город по умолчанию. Результаты геокодинга кэшируются на 72 часа.
func (s *Service) resolveLocation(ctx context.Context, city string) (openmeteo.Location, error) {
	if strings.TrimSpace(city) == "" {
		if last := s.Store.Last(); last != nil {
			return last.Location, nil
		}
		city = s.Cfg.DefaultCity
	}
	city = strings.TrimSpace(city)
	cacheKey := "geo:" + strings.ToLower(city)
	if v, ok := s.Cache.Get(cacheKey); ok {
		if loc, ok := v.(openmeteo.Location); ok {
			return loc, nil
		}
	}
	loc, err := s.Geo.Geocode(ctx, city)
	if err != nil {
		return openmeteo.Location{}, err
	}
	s.Cache.Set(cacheKey, loc, ttlGeo)
	return loc, nil
}

// buildSubject собирает Subject из карты или места рождения пользователя.
func (s *Service) buildSubject(ctx context.Context, name string, birthDate, birthTime, city, houseSystem string) (astrologer.Subject, openmeteo.Location, error) {
	if strings.TrimSpace(name) == "" {
		return astrologer.Subject{}, openmeteo.Location{}, fmt.Errorf("задайте имя карты")
	}
	if strings.TrimSpace(city) == "" {
		return astrologer.Subject{}, openmeteo.Location{}, fmt.Errorf("задайте город рождения")
	}
	y, m, d, err := parseDate(birthDate)
	if err != nil {
		return astrologer.Subject{}, openmeteo.Location{}, fmt.Errorf("birth_date: %w", err)
	}
	hh, mm, err := parseTime(birthTime)
	if err != nil {
		return astrologer.Subject{}, openmeteo.Location{}, fmt.Errorf("birth_time: %w", err)
	}

	loc, err := s.resolveLocation(ctx, city)
	if err != nil {
		return astrologer.Subject{}, openmeteo.Location{}, err
	}

	hs := strings.TrimSpace(houseSystem)
	if hs == "" {
		hs = s.Cfg.Houses
	}
	subject := astrologer.Subject{
		Name:       strings.TrimSpace(name),
		Year:       y,
		Month:      m,
		Day:        d,
		Hour:       hh,
		Minute:     mm,
		City:       loc.Name,
		Nation:     loc.CountryCode,
		Longitude:  loc.Longitude,
		Latitude:   loc.Latitude,
		Timezone:   loc.Timezone,
		ZodiacType: "Tropical",
		Houses:     hs,
	}
	return subject, loc, nil
}

// transitSubject — субъект «текущего момента» в точке рождения карты.
func transitSubject(ch *Chart, t time.Time) astrologer.Subject {
	return astrologer.Subject{
		Name:       "Transit",
		Year:       t.Year(),
		Month:      int(t.Month()),
		Day:        t.Day(),
		Hour:       t.Hour(),
		Minute:     t.Minute(),
		City:       ch.Subject.City,
		Nation:     ch.Subject.Nation,
		Longitude:  ch.Subject.Longitude,
		Latitude:   ch.Subject.Latitude,
		Timezone:   ch.Subject.Timezone,
		ZodiacType: ch.Subject.ZodiacType,
		Houses:     ch.Subject.Houses,
	}
}

// chartLoc загружает карту из хранилища и её таймзону (UTC при сбое).
func (s *Service) chartLoc(ch *Chart) *time.Location {
	loc, err := time.LoadLocation(ch.Subject.Timezone)
	if err != nil {
		return time.UTC
	}
	return loc
}

func parseDate(s string) (int, int, int, error) {
	t, err := time.Parse("2006-01-02", strings.TrimSpace(s))
	if err != nil {
		return 0, 0, 0, fmt.Errorf("ожидается формат YYYY-MM-DD, получено %q", s)
	}
	return t.Year(), int(t.Month()), t.Day(), nil
}

func parseTime(s string) (int, int, error) {
	raw := strings.TrimSpace(s)
	if t, err := time.Parse("15:04:05", raw); err == nil {
		return t.Hour(), t.Minute(), nil
	}
	t, err := time.Parse("15:04", raw)
	if err != nil {
		return 0, 0, fmt.Errorf("ожидается формат HH:MM, получено %q", s)
	}
	return t.Hour(), t.Minute(), nil
}

// ---------- Инструменты ----------

// BuildNatal строит и сохраняет натальную карту.
func (s *Service) BuildNatal(ctx context.Context, in NatalArgs) (map[string]any, error) {
	subject, loc, err := s.buildSubject(ctx, in.Name, in.BirthDate, in.BirthTime, in.City, in.HouseSystem)
	if err != nil {
		return nil, err
	}

	resp, err := s.Astro.BirthChartContext(ctx, &subject)
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}
	cd := ChartData(resp)

	chart := &Chart{
		Name:        subject.Name,
		Subject:     subject,
		Location:    loc,
		Aspects:     Aspects(cd, 0, 12),
		ElementDist: Distribution(cd, "element_distribution"),
		QualityDist: Distribution(cd, "quality_distribution"),
	}
	if err := s.Store.Save(chart); err != nil {
		return nil, err
	}

	return map[string]any{
		"chart":                chart.Name,
		"birth":                chart.BirthLabel(),
		"location":             loc,
		"houses_system":        subject.Houses,
		"aspects":              chart.Aspects,
		"element_distribution": chart.ElementDist,
		"quality_distribution": chart.QualityDist,
		// XML-контекст движка — факты о позициях и домах, готовые для LLM.
		"context": resp["context"],
		"next":    "карта сохранена; доступны transit_today, transit_forecast, synastry_compare, solar_return, chart_render по имени " + chart.Name,
	}, nil
}

// TransitToday — персональные транзиты дня к натальной карте.
func (s *Service) TransitToday(ctx context.Context, in ChartArgs) (map[string]any, error) {
	ch, err := s.Store.Get(in.Name)
	if err != nil {
		return nil, err
	}
	now := s.now().In(s.chartLoc(ch))
	resp, err := s.Astro.TransitContext(ctx, &ch.Subject, subjectPtr(transitSubject(ch, now)))
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}
	cd := ChartData(resp)
	return map[string]any{
		"chart":   ch.Name,
		"moment":  now.Format("2006-01-02 15:04 (MST)"),
		"aspects": Aspects(cd, 0, 12),
		"context": resp["context"],
	}, nil
}

// TransitForecast — календарь транзитов на окно дней (по одному запросу
// в день к self-host движку — бесплатно и без лимитов).
func (s *Service) TransitForecast(ctx context.Context, in ForecastArgs) (map[string]any, error) {
	days := 7
	if in.Days != nil {
		days = *in.Days
	}
	if days < 1 {
		days = 1
	}
	if days > 31 {
		days = 31
	}
	ch, err := s.Store.Get(in.Name)
	if err != nil {
		return nil, err
	}
	loc := s.chartLoc(ch)
	start := s.now().In(loc)

	rows := make([]map[string]any, 0, days)
	for i := 0; i < days; i++ {
		day := start.AddDate(0, 0, i)
		noon := time.Date(day.Year(), day.Month(), day.Day(), 12, 0, 0, 0, loc)
		resp, err := s.Astro.TransitData(ctx, &ch.Subject, subjectPtr(transitSubject(ch, noon)))
		if err != nil {
			return nil, fmt.Errorf("прогноз, день %s: %w", day.Format("2006-01-02"), err)
		}
		rows = append(rows, map[string]any{
			"date":    day.Format("2006-01-02"),
			"aspects": Aspects(ChartData(resp), 3.0, 4), // только тесные орбисы
		})
	}
	return map[string]any{
		"chart": ch.Name,
		"days":  rows,
		"note":  "в каждом дне — аспекты транзитов к наталу с орбисом до 3° (чем меньше орбис, тем сильнее)",
	}, nil
}

// Synastry сравнивает две сохранённые карты.
func (s *Service) Synastry(ctx context.Context, in SynastryArgs) (map[string]any, error) {
	first, err := s.Store.Get(in.First)
	if err != nil {
		return nil, err
	}
	second, err := s.Store.Get(in.Second)
	if err != nil {
		return nil, err
	}
	resp, err := s.Astro.SynastryContext(ctx, &first.Subject, &second.Subject)
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}
	cd := ChartData(resp)
	return map[string]any{
		"first":           first.Name,
		"second":          second.Name,
		"score":           Score(cd),
		"score_breakdown": Breakdown(cd, 6),
		"aspects":         Aspects(cd, 0, 12),
		"context":         resp["context"],
	}, nil
}

// SolarReturn — солярная карта на год.
func (s *Service) SolarReturn(ctx context.Context, in SolarArgs) (map[string]any, error) {
	ch, err := s.Store.Get(in.Name)
	if err != nil {
		return nil, err
	}
	year := s.now().Year()
	if in.Year != nil && *in.Year > 1900 {
		year = *in.Year
	}
	resp, err := s.Astro.SolarReturnContext(ctx, &ch.Subject, year)
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}
	cd := ChartData(resp)
	return map[string]any{
		"chart":   ch.Name,
		"year":    year,
		"aspects": Aspects(cd, 0, 8),
		"context": resp["context"],
	}, nil
}

// MoonPhase — фаза Луны, затмения и световой день для места и даты.
func (s *Service) MoonPhase(ctx context.Context, in MoonArgs) (map[string]any, error) {
	loc, err := s.resolveLocation(ctx, in.City)
	if err != nil {
		return nil, err
	}
	tz, tzErr := time.LoadLocation(loc.Timezone)
	if tzErr != nil {
		tz = time.UTC
	}
	when := s.now().In(tz)
	if strings.TrimSpace(in.Date) != "" {
		y, m, d, perr := parseDate(in.Date)
		if perr != nil {
			return nil, perr
		}
		when = time.Date(y, time.Month(m), d, 12, 0, 0, 0, tz)
	}

	cacheKey := fmt.Sprintf("moon:%s:%s", strings.ToLower(loc.Name), when.Format("2006-01-02"))
	if v, ok := s.Cache.Get(cacheKey); ok {
		if payload, ok := v.(map[string]any); ok {
			return payload, nil
		}
	}

	resp, err := s.Astro.MoonPhaseContext(ctx, when, loc.Latitude, loc.Longitude, loc.Timezone)
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}
	overview := mapAny(resp["moon_phase_overview"])
	moon := mapAny(overview["moon"])
	sun := mapAny(overview["sun"])

	payload := map[string]any{
		"location": loc.Label(),
		"date":     when.Format("2006-01-02"),
		"phase": map[string]any{
			"name":         moon["phase_name"],
			"major":        moon["major_phase"],
			"stage":        moon["stage"],
			"illumination": moon["illumination"],
			"age_days":     moon["age_days"],
			"emoji":        moon["emoji"],
			"moon_sign":    mapAny(moon["zodiac"])["moon_sign"],
			"sun_sign":     mapAny(moon["zodiac"])["sun_sign"],
		},
		"next_lunar_eclipse": eclipse(moon, "next_lunar_eclipse"),
		"next_solar_eclipse": eclipse(sun, "next_solar_eclipse"),
		"sunrise":            mapAny(sun)["sunrise_timestamp"],
		"sunset":             mapAny(sun)["sunset_timestamp"],
		"context":            resp["context"],
	}
	s.Cache.Set(cacheKey, payload, ttlMoon)
	return payload, nil
}

// eclipse — компактная сводка о затмении из блока overview.
func eclipse(m map[string]any, key string) map[string]any {
	e := mapAny(m[key])
	if e == nil {
		return nil
	}
	return map[string]any{
		"type": e["type"],
		"date": e["datestamp"],
	}
}

// VibeToday — брифинг дня: погода + Луна + транзиты к последней карте.
func (s *Service) VibeToday(ctx context.Context, in VibeArgs) (map[string]any, error) {
	loc, err := s.resolveLocation(ctx, in.City)
	if err != nil {
		return nil, err
	}
	tz, tzErr := time.LoadLocation(loc.Timezone)
	if tzErr != nil {
		tz = time.UTC
	}
	now := s.now().In(tz)

	payload := map[string]any{
		"city": loc.Label(),
		"date": now.Format("2006-01-02"),
	}

	// Погода (кэш 30 минут).
	wKey := fmt.Sprintf("weather:%s:%s", strings.ToLower(loc.Name), now.Format("2006-01-02 15"))
	if v, ok := s.Cache.Get(wKey); ok {
		payload["weather"] = v
	} else {
		w, werr := s.Weather.Current(ctx, loc.Latitude, loc.Longitude, loc.Timezone)
		if werr != nil {
			payload["weather"] = map[string]any{"error": werr.Error()}
		} else {
			s.Cache.Set(wKey, w, ttlWeather)
			payload["weather"] = w
		}
	}

	// Луна — переиспользуем MoonPhase без города (resolveLocation отдаст loc).
	moon, merr := s.MoonPhase(ctx, MoonArgs{})
	if merr == nil {
		payload["moon"] = moon["phase"]
		if e := moon["next_lunar_eclipse"]; e != nil {
			payload["next_lunar_eclipse"] = e
		}
	}

	// Транзиты — только если натальная карта уже построена.
	if last := s.Store.Last(); last != nil {
		resp, terr := s.Astro.TransitContext(ctx, &last.Subject, subjectPtr(transitSubject(last, now)))
		if terr == nil {
			payload["transits"] = map[string]any{
				"chart":   last.Name,
				"aspects": Aspects(ChartData(resp), 6.0, 6),
			}
		} else {
			payload["transits"] = map[string]any{"error": terr.Error()}
		}
	} else {
		payload["transits"] = map[string]any{
			"note": "натальная карта не построена — вызовите natal_build_chart, чтобы включить персональные транзиты",
		}
	}

	payload["hints"] = hints(payload)
	return payload, nil
}

// hints — детерминированные практические подсказки (без астро-трактовок:
// их даёт модель поверх фактов).
func hints(payload map[string]any) []string {
	var out []string
	if w, ok := payload["weather"].(openmeteo.Weather); ok {
		if w.PrecipProb >= 50 {
			out = append(out, fmt.Sprintf("вероятность осадков %d%% — стоит взять зонт", w.PrecipProb))
		}
		if w.TempMax-w.TempMin >= 10 {
			out = append(out, fmt.Sprintf("большой перепад температур за день (%.0f…%.0f °C) — одевайтесь слоями", w.TempMin, w.TempMax))
		}
		if w.WindSpeed >= 30 {
			out = append(out, fmt.Sprintf("сильный ветер (%.0f км/ч)", w.WindSpeed))
		}
	}
	if e, ok := payload["next_lunar_eclipse"].(map[string]any); ok && e != nil && e["date"] != nil {
		out = append(out, fmt.Sprintf("ближайшее лунное затмение: %v (%v)", e["date"], e["type"]))
	}
	if len(out) == 0 {
		out = append(out, "без ярких погодных событий — день под контролем")
	}
	return out
}

// RenderResult — результат chart_render. Модель видит только Model
// (компактная сводка без путей, ссылок и SVG); сам SVG уходит клиенту
// отдельным image-контентом MCP, чтобы тот показал картинку пользователю.
// FileName — имя в галерее сервера, никуда дальше не передаётся.
type RenderResult struct {
	Model    map[string]any
	SVG      []byte
	FileName string
}

// RenderChart — SVG-колесо карты (natal | transit | synastry).
func (s *Service) RenderChart(ctx context.Context, in RenderArgs) (*RenderResult, error) {
	kind := strings.ToLower(strings.TrimSpace(in.Type))
	first, err := s.Store.Get(in.First)
	if err != nil {
		return nil, err
	}

	opts := map[string]any{
		"theme":    s.Cfg.Theme,
		"language": s.Cfg.Lang,
		"style":    "classic",
	}
	if strings.TrimSpace(in.Theme) != "" {
		opts["theme"] = in.Theme
	}
	if strings.TrimSpace(in.Style) != "" {
		opts["style"] = in.Style
	}
	if title := strings.TrimSpace(in.Title); title != "" {
		if r := []rune(title); len(r) > 40 {
			title = string(r[:40])
		}
		opts["custom_title"] = title
	}

	var resp map[string]any
	var secondName string
	switch kind {
	case "natal":
		resp, err = s.Astro.BirthChartSVG(ctx, &first.Subject, opts)
	case "transit":
		second := transitSubject(first, s.now().In(s.chartLoc(first)))
		secondName = second.Name
		resp, err = s.Astro.TransitSVG(ctx, &first.Subject, &second, opts)
	case "synastry":
		if in.Second == nil || strings.TrimSpace(*in.Second) == "" {
			return nil, fmt.Errorf("для synastry укажите second — имя второй карты")
		}
		second, serr := s.Store.Get(*in.Second)
		if serr != nil {
			return nil, serr
		}
		secondName = second.Name
		resp, err = s.Astro.SynastrySVG(ctx, &first.Subject, &second.Subject, opts)
	default:
		return nil, fmt.Errorf("type должен быть natal, transit или synastry, получено %q", in.Type)
	}
	if err != nil {
		return nil, fmt.Errorf("астро-движок: %w", err)
	}

	svg, _ := resp["chart"].(string)
	if svg == "" {
		return nil, fmt.Errorf("движок не вернул SVG (проверьте type и параметры рендера)")
	}

	// Сырой SVG в текстовый результат для модели не попадает: он раздувает
	// контекст и не отображается в чате. Файл нужен только галерее
	// статус-страницы, а пользователю картинку доставляет клиент
	// (image-контент MCP → системный просмотрщик).
	file, ferr := s.saveSVG(kind, first.Name, secondName, svg)
	if ferr != nil {
		return nil, fmt.Errorf("не удалось сохранить SVG: %w", ferr)
	}
	out := map[string]any{
		"type":  kind,
		"chart": first.Name,
		"bytes": len(svg),
		"note":  "SVG-колесо доставлено клиенту как изображение (image/svg+xml) и показано пользователю; также доступно в галерее статус-страницы сервера. Не вставляйте в ответ SVG, пути к файлам и ссылки — просто подтвердите, что карта отрисована.",
	}
	if in.Second != nil && kind == "synastry" {
		out["second"] = *in.Second
	}
	return &RenderResult{
		Model:    out,
		SVG:      []byte(svg),
		FileName: filepath.Base(file),
	}, nil
}

// saveSVG записывает колесо в ChartsDir под детерминированным именем
// (повторный рендер той же карты перезаписывает файл).
func (s *Service) saveSVG(kind, firstName, secondName, svg string) (string, error) {
	if strings.TrimSpace(s.ChartsDir) == "" {
		return "", fmt.Errorf("каталог для SVG не задан (ORBIT_CHARTS_DIR)")
	}
	name := safeName(firstName) + "-" + kind
	if kind == "synastry" && secondName != "" {
		name += "-" + safeName(secondName)
	}
	path := filepath.Join(s.ChartsDir, name+".svg")
	if err := os.MkdirAll(s.ChartsDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(svg), 0o644); err != nil {
		return "", err
	}
	if abs, aerr := filepath.Abs(path); aerr == nil {
		path = abs
	}
	return path, nil
}

// safeName превращает имя карты в безопасную часть имени файла:
// буквы и цифры остаются, всё прочее заменяется на '-'.
func safeName(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		out = "chart"
	}
	return out
}

// RenderedCharts — отрисованные SVG из ChartsDir (для страницы статуса).
func (s *Service) RenderedCharts() []map[string]any {
	if strings.TrimSpace(s.ChartsDir) == "" {
		return nil
	}
	entries, err := os.ReadDir(s.ChartsDir)
	if err != nil {
		return nil
	}
	out := []map[string]any{}
	for _, e := range entries {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".svg") {
			continue
		}
		info, ierr := e.Info()
		if ierr != nil {
			continue
		}
		item := map[string]any{
			"file": filepath.Join(s.ChartsDir, e.Name()),
			"url":  "/charts/" + url.PathEscape(e.Name()),
			"size": info.Size(),
		}
		if s.PublicURL != "" {
			item["absolute_url"] = s.PublicURL + "/charts/" + url.PathEscape(e.Name())
		}
		out = append(out, item)
	}
	return out
}

// Status — диагностика: провайдеры и сохранённые карты.
func (s *Service) Status(ctx context.Context) (map[string]any, error) {
	astroOK, astroErr := "ok", ""
	{
		hctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		if err := s.Astro.Health(hctx); err != nil {
			astroOK, astroErr = "down", err.Error()
		}
	}
	charts := s.Store.List()
	list := make([]map[string]any, 0, len(charts))
	for _, ch := range charts {
		list = append(list, map[string]any{
			"name":  ch.Name,
			"birth": ch.BirthLabel(),
		})
	}
	return map[string]any{
		"service": "orbit-mcp",
		"providers": map[string]any{
			"astrologer": map[string]any{
				"url":     s.Cfg.AstroURL + "/api/" + s.Cfg.AstroVersion,
				"status":  astroOK,
				"error":   astroErr,
				"comment": "self-hosted Astrologer-API; запустите: uv run poe start (см. README)",
			},
			"geocoding": map[string]any{"url": s.Cfg.GeoURL, "status": "ok", "comment": "Open-Meteo Geocoding, без ключа"},
			"weather":   map[string]any{"url": s.Cfg.OpenMeteoURL, "status": "ok", "comment": "Open-Meteo Weather, без ключа"},
		},
		"charts":     list,
		"store_file": s.Cfg.StoreFile,
	}, nil
}

// ChartsList — список сохранённых карт (для status-страницы и probe).
func (s *Service) ChartsList() []map[string]any {
	charts := s.Store.List()
	out := make([]map[string]any, 0, len(charts))
	for _, ch := range charts {
		out = append(out, map[string]any{
			"name":     ch.Name,
			"birth":    ch.BirthLabel(),
			"built_at": ch.BuiltAt.Format(time.RFC3339),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return fmt.Sprint(out[i]["name"]) < fmt.Sprint(out[j]["name"]) })
	return out
}

// summaryContext — короткая строка-напоминание о карте (без тяжёлого XML).
func (ch *Chart) summaryContext() string {
	b, _ := json.Marshal(map[string]any{
		"name":     ch.Name,
		"birth":    ch.BirthLabel(),
		"location": ch.Location.Label(),
		"aspects":  ch.Aspects,
	})
	return string(b)
}

// subjectPtr — хелпер для вызовов, где движок ждёт указатель.
func subjectPtr(s astrologer.Subject) *astrologer.Subject { return &s }
