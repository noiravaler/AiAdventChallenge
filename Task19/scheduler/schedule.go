// Package scheduler runs scheduled agent jobs in the background: a ticker
// goroutine fires due jobs and executes their prompt through the regular
// agent pipeline (model + tools + MCP), provided by the Runner callback.
package scheduler

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// minInterval is the smallest allowed "@every" period: the scheduler makes
// paid LLM calls, so sub-minute hammering is rejected at parse time.
const minInterval = time.Minute

// maxTimesPerDay bounds the number of comma-separated HH:MM triggers.
const maxTimesPerDay = 24

// TimeOfDay is a minute-of-day trigger (HH:MM in the job's timezone).
type TimeOfDay struct {
	Hour int
	Min  int
}

// Schedule describes when a job fires. Exactly one of Interval / Times is
// meaningful; Weekdays additionally filters the Times triggers (nil = every
// day). Schedules are value objects: parse once, then query Next.
type Schedule struct {
	Interval time.Duration // >0: fire every Interval, counted from the previous run end
	Times    []TimeOfDay   // daily triggers at fixed clock times
	Weekdays []time.Weekday // optional filter for Times (nil = every day)
}

// ParseSchedule parses a schedule spec:
//
//	@every 30m | @every 2h        — interval between runs
//	09:00 | 09:00,18:30           — daily at fixed clock times
//	[пн,ср,пт] 09:00              — weekly filter (Russian or English day names)
//
// Times are interpreted in the job's timezone (Job.TZ, default local).
// The minimum interval is one minute; unknown syntax is an error.
func ParseSchedule(spec string) (Schedule, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return Schedule{}, fmt.Errorf("расписание пустое; примеры: @every 1h, 09:00, [пн,ср,пт] 09:00")
	}

	// "@every <duration>" — fixed interval between runs.
	if strings.HasPrefix(strings.ToLower(spec), "@every ") {
		d, err := time.ParseDuration(strings.TrimSpace(spec[len("@every "):]))
		if err != nil {
			return Schedule{}, fmt.Errorf("расписание %q: ожидается @every <длительность>, например @every 30m", spec)
		}
		if d < minInterval {
			return Schedule{}, fmt.Errorf("расписание %q: минимальный интервал — %s", spec, minInterval)
		}
		return Schedule{Interval: d}, nil
	}

	// Optional "[пн,ср,пт]" weekday prefix.
	var weekdays []time.Weekday
	rest := spec
	if strings.HasPrefix(spec, "[") {
		end := strings.Index(spec, "]")
		if end < 0 {
			return Schedule{}, fmt.Errorf("расписание %q: не закрыта скобка дней недели", spec)
		}
		for _, day := range strings.Split(spec[1:end], ",") {
			wd, ok := parseWeekday(day)
			if !ok {
				return Schedule{}, fmt.Errorf("расписание %q: неизвестный день недели %q", spec, strings.TrimSpace(day))
			}
			weekdays = append(weekdays, wd)
		}
		rest = spec[end+1:]
	}

	// "HH:MM[,HH:MM...]" clock times.
	var times []TimeOfDay
	for _, part := range strings.Split(rest, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			return Schedule{}, fmt.Errorf("расписание %q: пустое время (ожидается HH:MM)", spec)
		}
		hh, mm, err := parseClock(part)
		if err != nil {
			return Schedule{}, fmt.Errorf("расписание %q: %v", spec, err)
		}
		times = append(times, TimeOfDay{Hour: hh, Min: mm})
	}
	if len(times) > maxTimesPerDay {
		return Schedule{}, fmt.Errorf("расписание %q: слишком много моментов (максимум %d)", spec, maxTimesPerDay)
	}
	return Schedule{Times: times, Weekdays: weekdays}, nil
}

// Next returns the first fire time strictly after "after", in loc.
// For an interval schedule it is simply after+Interval (the engine re-arms
// after each run finishes, so a long run never collides with the next one).
// For clock times it scans the following 8 days for the earliest matching
// moment; a schedule that never fires (impossible weekday filter) returns
// the zero time.
func (s Schedule) Next(after time.Time, loc *time.Location) time.Time {
	if s.Interval > 0 {
		return after.Add(s.Interval)
	}
	if len(s.Times) == 0 {
		return time.Time{}
	}
	after = after.In(loc)
	var best time.Time
	for day := 0; day < 8; day++ {
		base := after.AddDate(0, 0, day)
		if len(s.Weekdays) > 0 && !s.matchesWeekday(base.Weekday()) {
			continue
		}
		for _, t := range s.Times {
			cand := time.Date(base.Year(), base.Month(), base.Day(), t.Hour, t.Min, 0, 0, loc)
			if !cand.After(after) {
				continue
			}
			if best.IsZero() || cand.Before(best) {
				best = cand
			}
		}
	}
	return best
}

// String renders the schedule back to a spec-like form (for logs/UI).
func (s Schedule) String() string {
	if s.Interval > 0 {
		return "@every " + s.Interval.String()
	}
	parts := make([]string, 0, len(s.Times))
	for _, t := range s.Times {
		parts = append(parts, fmt.Sprintf("%02d:%02d", t.Hour, t.Min))
	}
	out := strings.Join(parts, ",")
	if len(s.Weekdays) > 0 {
		names := make([]string, 0, len(s.Weekdays))
		for _, wd := range s.Weekdays {
			names = append(names, weekdayName(wd))
		}
		out = "[" + strings.Join(names, ",") + "] " + out
	}
	return out
}

func (s Schedule) matchesWeekday(wd time.Weekday) bool {
	for _, d := range s.Weekdays {
		if d == wd {
			return true
		}
	}
	return false
}

// parseClock parses "HH:MM" with strict ranges.
func parseClock(v string) (int, int, error) {
	hhStr, mmStr, ok := strings.Cut(v, ":")
	if !ok {
		return 0, 0, fmt.Errorf("ожидается HH:MM, получено %q", v)
	}
	hh, err := strconv.Atoi(strings.TrimSpace(hhStr))
	if err != nil {
		return 0, 0, fmt.Errorf("час %q не число", hhStr)
	}
	mm, err := strconv.Atoi(strings.TrimSpace(mmStr))
	if err != nil {
		return 0, 0, fmt.Errorf("минуты %q не число", mmStr)
	}
	if hh < 0 || hh > 23 || mm < 0 || mm > 59 {
		return 0, 0, fmt.Errorf("время %q вне диапазона 00:00–23:59", v)
	}
	return hh, mm, nil
}

// weekdayNames maps accepted day tokens to time.Weekday (Sunday = 0).
var weekdayNames = map[string]time.Weekday{
	"пн": time.Monday, "вт": time.Tuesday, "ср": time.Wednesday,
	"чт": time.Thursday, "пт": time.Friday, "сб": time.Saturday, "вс": time.Sunday,
	"mon": time.Monday, "tue": time.Tuesday, "wed": time.Wednesday,
	"thu": time.Thursday, "fri": time.Friday, "sat": time.Saturday, "sun": time.Sunday,
}

func parseWeekday(v string) (time.Weekday, bool) {
	wd, ok := weekdayNames[strings.ToLower(strings.TrimSpace(v))]
	return wd, ok
}

func weekdayName(wd time.Weekday) string {
	for name, d := range weekdayNames {
		if d == wd && len([]rune(name)) == 2 {
			return name
		}
	}
	return wd.String()
}
