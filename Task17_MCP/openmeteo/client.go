// Package openmeteo — клиенты бесплатных API Open-Meteo (без ключей):
// Geocoding (город → координаты + IANA-таймзона) и Weather (текущая погода).
package openmeteo

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Location — результат геокодинга: всё, что нужно для построения карты.
type Location struct {
	Name        string  `json:"name"`
	Admin1      string  `json:"admin1,omitempty"` // регион
	Country     string  `json:"country,omitempty"`
	CountryCode string  `json:"country_code,omitempty"` // ISO-3166 alpha-2 (nation для Astrologer)
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	Timezone    string  `json:"timezone"` // IANA
	Population  int     `json:"population,omitempty"`
}

// Label — человекочитаемое название места («Казань, Татарстан, RU»).
func (l Location) Label() string {
	s := l.Name
	if l.Admin1 != "" && l.Admin1 != l.Name {
		s += ", " + l.Admin1
	}
	if l.CountryCode != "" {
		s += " (" + l.CountryCode + ")"
	}
	return s
}

// Weather — текущая погода и дневной прогноз для брифинга vibe_today.
type Weather struct {
	Temperature   float64 `json:"temperature"` // °C
	Apparent      float64 `json:"apparent"`    // ощущается, °C
	WindSpeed     float64 `json:"wind_speed"`  // км/ч
	Code          int     `json:"code"`        // WMO weather code
	Description   string  `json:"description"` // человекочитаемо (RU)
	TempMax       float64 `json:"temp_max"`    // прогноз на день
	TempMin       float64 `json:"temp_min"`
	PrecipProb    int     `json:"precip_probability"` // вероятность осадков, %
	Precipitation float64 `json:"precipitation"`      // сумма осадков, мм
}

// GeoClient — geocoding-api.open-meteo.com.
type GeoClient struct {
	base string
	hc   *http.Client
}

// NewGeoClient создаёт клиент геокодинга.
func NewGeoClient(base string, timeout time.Duration) *GeoClient {
	return &GeoClient{base: trimBase(base), hc: &http.Client{Timeout: timeout}}
}

// Geocode ищет город и возвращает первый (обычно самый населённый) результат.
func (c *GeoClient) Geocode(ctx context.Context, city string) (Location, error) {
	q := url.Values{}
	q.Set("name", city)
	q.Set("count", "5")
	q.Set("language", "ru")
	q.Set("format", "json")

	raw, err := c.get(ctx, "/v1/search?"+q.Encode())
	if err != nil {
		return Location{}, err
	}
	var parsed struct {
		Results []Location `json:"results"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Location{}, fmt.Errorf("некорректный ответ геокодинга: %w", err)
	}
	if len(parsed.Results) == 0 {
		return Location{}, fmt.Errorf("город %q не найден", city)
	}
	// Среди совпадений предпочитаем самый крупный населённый пункт.
	best := parsed.Results[0]
	for _, r := range parsed.Results[1:] {
		if r.Population > best.Population {
			best = r
		}
	}
	if best.Timezone == "" {
		best.Timezone = "UTC"
	}
	return best, nil
}

func (c *GeoClient) get(ctx context.Context, pathAndQuery string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+pathAndQuery, nil)
	if err != nil {
		return nil, err
	}
	res, err := c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 4<<20))
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("геокодинг: HTTP %d: %s", res.StatusCode, firstRunes(string(raw), 160))
	}
	return raw, nil
}

// WeatherClient — api.open-meteo.com (прогноз погоды).
type WeatherClient struct {
	base string
	hc   *http.Client
}

// NewWeatherClient создаёт клиент погоды.
func NewWeatherClient(base string, timeout time.Duration) *WeatherClient {
	return &WeatherClient{base: trimBase(base), hc: &http.Client{Timeout: timeout}}
}

// Current запрашивает текущую погоду и сводку на день для координат.
// timezone — IANA-зона, чтобы «сегодня» считалось по местному времени.
func (c *WeatherClient) Current(ctx context.Context, latitude, longitude float64, timezone string) (Weather, error) {
	q := url.Values{}
	q.Set("latitude", strconv.FormatFloat(latitude, 'f', 4, 64))
	q.Set("longitude", strconv.FormatFloat(longitude, 'f', 4, 64))
	q.Set("current", "temperature_2m,apparent_temperature,weather_code,wind_speed_10m")
	q.Set("daily", "temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum")
	q.Set("forecast_days", "1")
	q.Set("timezone", timezone)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/v1/forecast?"+q.Encode(), nil)
	if err != nil {
		return Weather{}, err
	}
	res, err := c.hc.Do(req)
	if err != nil {
		return Weather{}, err
	}
	defer res.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(res.Body, 4<<20))
	if err != nil {
		return Weather{}, err
	}
	if res.StatusCode != http.StatusOK {
		return Weather{}, fmt.Errorf("погода: HTTP %d: %s", res.StatusCode, firstRunes(string(raw), 160))
	}

	var parsed struct {
		Current struct {
			Temperature float64 `json:"temperature_2m"`
			Apparent    float64 `json:"apparent_temperature"`
			WeatherCode int     `json:"weather_code"`
			WindSpeed   float64 `json:"wind_speed_10m"`
		} `json:"current"`
		Daily struct {
			TempMax    []float64 `json:"temperature_2m_max"`
			TempMin    []float64 `json:"temperature_2m_min"`
			PrecipProb []int     `json:"precipitation_probability_max"`
			PrecipSum  []float64 `json:"precipitation_sum"`
		} `json:"daily"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Weather{}, fmt.Errorf("некорректный ответ погоды: %w", err)
	}

	w := Weather{
		Temperature: parsed.Current.Temperature,
		Apparent:    parsed.Current.Apparent,
		WindSpeed:   parsed.Current.WindSpeed,
		Code:        parsed.Current.WeatherCode,
		Description: DescribeCode(parsed.Current.WeatherCode),
	}
	if len(parsed.Daily.TempMax) > 0 {
		w.TempMax = parsed.Daily.TempMax[0]
	}
	if len(parsed.Daily.TempMin) > 0 {
		w.TempMin = parsed.Daily.TempMin[0]
	}
	if len(parsed.Daily.PrecipProb) > 0 {
		w.PrecipProb = parsed.Daily.PrecipProb[0]
	}
	if len(parsed.Daily.PrecipSum) > 0 {
		w.Precipitation = parsed.Daily.PrecipSum[0]
	}
	return w, nil
}

// DescribeCode — расшифровка WMO weather code по-русски (основные группы).
func DescribeCode(code int) string {
	switch {
	case code == 0:
		return "ясно"
	case code == 1 || code == 2:
		return "переменная облачность"
	case code == 3:
		return "облачно"
	case code == 45 || code == 48:
		return "туман"
	case code >= 51 && code <= 57:
		return "морось"
	case code >= 61 && code <= 67:
		return "дождь"
	case code >= 71 && code <= 77:
		return "снег"
	case code >= 80 && code <= 82:
		return "ливни"
	case code >= 85 && code <= 86:
		return "снегопад"
	case code >= 95:
		return "гроза"
	default:
		return "переменчиво"
	}
}

func trimBase(s string) string {
	if len(s) > 0 && s[len(s)-1] == '/' {
		return s[:len(s)-1]
	}
	return s
}

func firstRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n]) + "…"
	}
	return string(r)
}
