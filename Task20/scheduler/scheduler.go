package scheduler

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"strconv"
	"sync"
	"time"
)

// Runner executes a job's prompt through the agent pipeline (model + tools +
// MCP). The engine knows nothing about the chat internals — wiring happens in
// app/main.go, so scheduled runs get exactly the same capabilities as a chat.
type Runner func(ctx context.Context, job Job) (answer string, err error)

// Job is one scheduled task. Configuration fields come from the user (via
// the API/CLI); runtime fields are maintained by the engine and persisted.
type Job struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Prompt   string `json:"prompt"`
	Tool     string `json:"tool,omitempty"` // MCP tool name: direct call, no model round-trip
	System   string `json:"system,omitempty"`
	Schedule string `json:"schedule"`
	TZ       string `json:"tz,omitempty"` // IANA name; empty = local time
	Model    string `json:"model,omitempty"`
	Enabled  bool   `json:"enabled"`

	// Runtime state (managed by the engine).
	NextRun   time.Time `json:"next_run,omitempty"`
	LastRun   time.Time `json:"last_run,omitempty"`
	LastOK    bool      `json:"last_ok"`
	LastError string    `json:"last_error,omitempty"`
	LastDurMS int64     `json:"last_dur_ms,omitempty"`
	Running   bool      `json:"running,omitempty"`
}

// Result is one execution record of a job (the engine keeps a ring of the
// most recent ones per job).
type Result struct {
	At     time.Time `json:"at"`
	OK     bool      `json:"ok"`
	DurMS  int64     `json:"dur_ms"`
	Answer string    `json:"answer,omitempty"`
	Error  string    `json:"error,omitempty"`
}

// Options tune the engine; the zero value is production-ready.
type Options struct {
	DB         string           // SQLite database path; empty = in-memory database
	Tick       time.Duration    // how often due jobs are checked (default 30s)
	RunTimeout time.Duration    // per-run context timeout (default 5m)
	MaxResults int              // kept results per job (default 20)
	MaxAnswer  int              // answer bytes kept per result (default 4000)
	Now        func() time.Time // clock override for tests
	Logf       func(format string, args ...any) // logger override for tests
}

// jobState couples a job with its parsed schedule and result ring.
type jobState struct {
	job     Job
	sched   Schedule
	results []Result
}

// Scheduler is the background engine: a ticker goroutine fires due jobs, each
// run executes in its own goroutine; a job never runs twice concurrently, and
// a manual run (RunNow) never disturbs the regular schedule.
type Scheduler struct {
	runner Runner
	opt    Options
	db     *sql.DB // SQLite store; nil = работает без хранения (открытие не удалось)

	mu   sync.Mutex
	jobs []*jobState

	started bool
	stop    chan struct{}
	wake    chan struct{}
	done    chan struct{}
	wg      sync.WaitGroup // in-flight job runs
}

// Defaults applied when the corresponding Options field is zero.
const (
	defaultTick       = 30 * time.Second
	defaultRunTimeout = 5 * time.Minute
	defaultMaxResults = 20
	defaultMaxAnswer  = 4000
)

// New creates a scheduler around the given runner (nil runner makes every
// run fail with an explicit error — handy for UI-only usage).
func New(runner Runner, opt Options) *Scheduler {
	s := &Scheduler{runner: runner, opt: opt}
	if s.opt.Tick <= 0 {
		s.opt.Tick = defaultTick
	}
	if s.opt.RunTimeout <= 0 {
		s.opt.RunTimeout = defaultRunTimeout
	}
	if s.opt.MaxResults <= 0 {
		s.opt.MaxResults = defaultMaxResults
	}
	if s.opt.MaxAnswer <= 0 {
		s.opt.MaxAnswer = defaultMaxAnswer
	}
	if s.opt.Now == nil {
		s.opt.Now = time.Now
	}
	if s.opt.Logf == nil {
		s.opt.Logf = log.Printf
	}
	s.stop = make(chan struct{})
	s.wake = make(chan struct{}, 1)
	s.done = make(chan struct{})
	// Хранилище открываем сразу (ленивая схема внутри); пустой путь — база
	// в памяти. Ошибка открытия не мешает работе: планировщик продолжит без
	// персистентности.
	db, err := openStore(s.opt.DB)
	if err != nil {
		s.opt.Logf("scheduler: sqlite: %v — работаю без хранения", err)
	} else {
		s.db = db
	}
	return s
}

// ---------------------------------------------------------------- persistence

type fileState struct {
	Jobs []Job `json:"jobs"`
}

// Load reads jobs (and their recent results) from the SQLite store. A fresh
// database is not an error (empty state); NextRun of enabled jobs is
// recomputed from "now", so runs missed while the app was down are skipped,
// never replayed. If the store is empty and a legacy scheduler.json exists
// next to the database, jobs are imported from it once.
func (s *Scheduler) Load() error {
	now := s.opt.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return nil
	}
	importLegacyJSON(s.db, legacyJSONPath(s.opt.DB), s.opt.Logf)
	jobs, err := loadJobs(s.db)
	if err != nil {
		return fmt.Errorf("load jobs: %w", err)
	}
	s.jobs = s.jobs[:0]
	for _, j := range jobs {
		sched, err := ParseSchedule(j.Schedule)
		if err != nil {
			s.opt.Logf("scheduler: задача %q пропущена: %v", j.Name, err)
			continue
		}
		j.Running = false
		if j.Enabled {
			j.NextRun = sched.Next(now, s.loc(j))
		} else {
			j.NextRun = time.Time{}
		}
		results, err := loadResults(s.db, j.ID, s.opt.MaxResults)
		if err != nil {
			s.opt.Logf("scheduler: результаты %q: %v", j.Name, err)
		}
		s.jobs = append(s.jobs, &jobState{job: j, sched: sched, results: results})
	}
	return nil
}

// persistJob writes one job row (definitions + last-run summary). Best-effort:
// callers log a warning on error. Called without s.mu held.
func (s *Scheduler) persistJob(j Job) {
	if s.db == nil {
		return
	}
	if err := saveJob(s.db, j); err != nil {
		s.opt.Logf("scheduler: save: %v", err)
	}
}

// persistDelete removes the job row and its results. Best-effort.
func (s *Scheduler) persistDelete(id string) {
	if s.db == nil {
		return
	}
	if err := deleteJob(s.db, id); err != nil {
		s.opt.Logf("scheduler: delete: %v", err)
	}
}

// persistResult appends one result to the journal and trims the ring to
// MaxResults rows. Best-effort.
func (s *Scheduler) persistResult(jobID string, res Result) {
	if s.db == nil {
		return
	}
	if err := insertResult(s.db, jobID, res, s.opt.MaxResults); err != nil {
		s.opt.Logf("scheduler: result: %v", err)
	}
}

// ------------------------------------------------------------------- CRUD API

// Add validates and appends a job; ID is assigned here. Enabled jobs get an
// immediate NextRun (a future moment — missed runs are never replayed).
func (s *Scheduler) Add(j Job) (Job, error) {
	sched, loc, err := s.validate(j)
	if err != nil {
		return Job{}, err
	}
	j.ID = newID()
	if j.Enabled {
		j.NextRun = sched.Next(s.opt.Now(), loc)
	}
	st := &jobState{job: j, sched: sched}
	s.mu.Lock()
	s.jobs = append(s.jobs, st)
	s.mu.Unlock()
	s.wakeLoop()
	s.persistJob(j)
	return j, nil
}

// Update replaces the configuration of an existing job (matched by ID),
// keeping its results.
func (s *Scheduler) Update(j Job) (Job, error) {
	sched, loc, err := s.validate(j)
	if err != nil {
		return Job{}, err
	}
	s.mu.Lock()
	st := s.byIDLocked(j.ID)
	if st == nil {
		s.mu.Unlock()
		return Job{}, fmt.Errorf("задача %s не найдена", j.ID)
	}
	j.Running = st.job.Running
	j.LastRun, j.LastOK, j.LastError, j.LastDurMS = st.job.LastRun, st.job.LastOK, st.job.LastError, st.job.LastDurMS
	if j.Enabled {
		j.NextRun = sched.Next(s.opt.Now(), loc)
	} else {
		j.NextRun = time.Time{}
	}
	st.job, st.sched = j, sched
	s.mu.Unlock()
	s.wakeLoop()
	s.persistJob(j)
	return j, nil
}

// Remove deletes a job; a currently running instance finishes on its own but
// its result goes nowhere (the job no longer exists).
func (s *Scheduler) Remove(id string) error {
	s.mu.Lock()
	idx := -1
	for i, st := range s.jobs {
		if st.job.ID == id {
			idx = i
			break
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		return fmt.Errorf("задача %s не найдена", id)
	}
	s.jobs = append(s.jobs[:idx], s.jobs[idx+1:]...)
	s.mu.Unlock()
	s.persistDelete(id)
	return nil
}

// Toggle switches a job on/off; enabling re-arms NextRun from "now".
func (s *Scheduler) Toggle(id string, enabled bool) (Job, error) {
	s.mu.Lock()
	st := s.byIDLocked(id)
	if st == nil {
		s.mu.Unlock()
		return Job{}, fmt.Errorf("задача %s не найдена", id)
	}
	st.job.Enabled = enabled
	if enabled {
		st.job.NextRun = st.sched.Next(s.opt.Now(), s.loc(st.job))
	} else {
		st.job.NextRun = time.Time{}
	}
	job := st.job
	s.mu.Unlock()
	s.wakeLoop()
	s.persistJob(job)
	return job, nil
}

// RunNow starts the job immediately in the background. It does not touch
// NextRun (manual runs do not reset the schedule) and returns an error if the
// job is already running or does not exist.
func (s *Scheduler) RunNow(id string) error {
	s.mu.Lock()
	st := s.byIDLocked(id)
	if st == nil {
		s.mu.Unlock()
		return fmt.Errorf("задача %s не найдена", id)
	}
	if st.job.Running {
		s.mu.Unlock()
		return fmt.Errorf("задача %q уже выполняется", st.job.Name)
	}
	st.job.Running = true
	st.job.LastRun = s.opt.Now()
	s.mu.Unlock()
	s.wg.Add(1)
	go s.runJob(st, true)
	return nil
}

// List returns a snapshot of all jobs in creation order.
func (s *Scheduler) List() []Job {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Job, 0, len(s.jobs))
	for _, st := range s.jobs {
		out = append(out, st.job)
	}
	return out
}

// Results returns the result ring of one job (oldest first).
func (s *Scheduler) Results(id string) []Result {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.byIDLocked(id)
	if st == nil {
		return nil
	}
	return append([]Result(nil), st.results...)
}

// ------------------------------------------------------------------- engine

// Start launches the ticker goroutine. Safe to call once; extra calls are
// no-ops. Without Start the scheduler still serves the CRUD API.
func (s *Scheduler) Start() {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return
	}
	s.started = true
	s.mu.Unlock()
	s.wg.Add(1)
	go s.loop()
}

// Stop stops the ticker and waits for in-flight job runs, then closes the
// SQLite store.
func (s *Scheduler) Stop() {
	s.mu.Lock()
	if !s.started {
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	close(s.stop)
	<-s.done
	s.wg.Wait()
	s.mu.Lock()
	db := s.db
	s.db = nil
	s.mu.Unlock()
	if db != nil {
		db.Close()
	}
}

// wake pokes the loop so freshly added/due jobs are evaluated without
// waiting for the next tick. Non-blocking: the loop also ticks on its own.
func (s *Scheduler) wakeLoop() {
	s.mu.Lock()
	started := s.started
	s.mu.Unlock()
	if !started {
		return
	}
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *Scheduler) loop() {
	defer s.wg.Done()
	defer close(s.done)
	ticker := time.NewTicker(s.opt.Tick)
	defer ticker.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-s.wake:
			s.fireDue()
		case <-ticker.C:
			s.fireDue()
		}
	}
}

// fireDue starts every enabled, idle job whose NextRun has arrived.
func (s *Scheduler) fireDue() {
	now := s.opt.Now()
	var due []*jobState
	s.mu.Lock()
	for _, st := range s.jobs {
		if !st.job.Enabled || st.job.Running || st.job.NextRun.IsZero() {
			continue
		}
		if st.job.NextRun.After(now) {
			continue
		}
		st.job.Running = true
		st.job.LastRun = now
		due = append(due, st)
	}
	s.mu.Unlock()
	for _, st := range due {
		s.wg.Add(1)
		go s.runJob(st, false)
	}
}

// runJob executes one job run, records the result, re-arms NextRun (unless
// manual) and persists. Runs in its own goroutine.
func (s *Scheduler) runJob(st *jobState, manual bool) {
	defer s.wg.Done()

	start := s.opt.Now()
	var (
		answer string
		err    error
	)
	if s.runner == nil {
		err = errors.New("планировщик не подключён к конвейеру агента")
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), s.opt.RunTimeout)
		answer, err = s.runner(ctx, st.job)
		cancel()
	}

	res := Result{
		At:    start,
		OK:    err == nil,
		DurMS: s.opt.Now().Sub(start).Milliseconds(),
	}
	if err != nil {
		res.Error = err.Error()
	} else {
		res.Answer = truncateRunes(answer, s.opt.MaxAnswer)
	}

	name := st.job.Name
	s.mu.Lock()
	st.results = append(st.results, res)
	if over := len(st.results) - s.opt.MaxResults; over > 0 {
		st.results = append([]Result(nil), st.results[over:]...)
	}
	st.job.Running = false
	st.job.LastRun = start
	st.job.LastOK = res.OK
	st.job.LastDurMS = res.DurMS
	st.job.LastError = res.Error
	if !manual {
		st.job.NextRun = st.sched.Next(s.opt.Now(), s.loc(st.job))
	}
	jobID := st.job.ID
	snapshot := st.job
	s.mu.Unlock()

	if err != nil {
		s.opt.Logf("scheduler: %q — ошибка за %d мс: %v", name, res.DurMS, err)
	} else {
		s.opt.Logf("scheduler: %q — выполнена за %d мс", name, res.DurMS)
	}
	s.persistResult(jobID, res)
	s.persistJob(snapshot)
}

// validate parses the schedule and timezone, and requires a name plus either
// a prompt (model turn) or a tool name (direct MCP call).
func (s *Scheduler) validate(j Job) (Schedule, *time.Location, error) {
	if j.Name == "" {
		return Schedule{}, nil, errors.New("укажите имя задачи")
	}
	if j.Tool == "" && j.Prompt == "" {
		return Schedule{}, nil, errors.New("укажите промпт задачи или имя инструмента (tool)")
	}
	sched, err := ParseSchedule(j.Schedule)
	if err != nil {
		return Schedule{}, nil, err
	}
	loc := time.Local
	if tz := j.TZ; tz != "" {
		if loc, err = time.LoadLocation(tz); err != nil {
			return Schedule{}, nil, fmt.Errorf("неизвестная таймзона %q", tz)
		}
	}
	return sched, loc, nil
}

func (s *Scheduler) loc(j Job) *time.Location {
	if j.TZ == "" {
		return time.Local
	}
	if loc, err := time.LoadLocation(j.TZ); err == nil {
		return loc
	}
	return time.Local
}

func (s *Scheduler) byIDLocked(id string) *jobState {
	for _, st := range s.jobs {
		if st.job.ID == id {
			return st
		}
	}
	return nil
}

// newID returns a short random hex id; falling back to a time-based value if
// the system RNG is unavailable.
func newID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err == nil {
		return hex.EncodeToString(b[:])
	}
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

func truncateRunes(v string, max int) string {
	if max <= 0 || len(v) <= max {
		return v
	}
	r := []rune(v)
	if len(r) <= max {
		return v
	}
	return string(r[:max]) + "…"
}

// ------------------------------------------------------------------ default

var (
	defaultMu   sync.Mutex
	defaultSched *Scheduler
)

// Install makes s the process-wide scheduler served by the HTTP API
// (same pattern as mcpclient.Install). nil disables the API.
func Install(s *Scheduler) {
	defaultMu.Lock()
	defaultSched = s
	defaultMu.Unlock()
}

// Default returns the installed scheduler or nil.
func Default() *Scheduler {
	defaultMu.Lock()
	defer defaultMu.Unlock()
	return defaultSched
}
