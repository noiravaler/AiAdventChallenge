package scheduler

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	// Pure-Go SQLite (no cgo): планировщик не тянет за собой C-тулчейн.
	_ "modernc.org/sqlite"
)

// storeSchema — таблицы планировщика: jobs (определения задач + сводка
// последнего запуска) и results (журнал запусков; кольцо ограничивается при
// вставке — последние MaxResults строк на задачу).
const storeSchema = `
CREATE TABLE IF NOT EXISTS jobs (
	id          TEXT PRIMARY KEY,
	name        TEXT NOT NULL,
	prompt      TEXT NOT NULL,
	tool        TEXT NOT NULL DEFAULT '',
	system      TEXT NOT NULL DEFAULT '',
	schedule    TEXT NOT NULL DEFAULT '',
	tz          TEXT NOT NULL DEFAULT '',
	model       TEXT NOT NULL DEFAULT '',
	enabled     INTEGER NOT NULL DEFAULT 1,
	next_run    TEXT NOT NULL DEFAULT '',
	last_run    TEXT NOT NULL DEFAULT '',
	last_ok     INTEGER NOT NULL DEFAULT 0,
	last_error  TEXT NOT NULL DEFAULT '',
	last_dur_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS results (
	id      INTEGER PRIMARY KEY AUTOINCREMENT,
	job_id  TEXT NOT NULL,
	at      TEXT NOT NULL,
	ok      INTEGER NOT NULL DEFAULT 0,
	dur_ms  INTEGER NOT NULL DEFAULT 0,
	answer  TEXT NOT NULL DEFAULT '',
	error   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id, id);
`

// openStore открывает (создавая при необходимости) базу SQLite и
// инициализирует схему. Пустой путь = база в памяти (состояние не переживает
// перезапуск). *sql.DB безопасен для конкурентного использования.
func openStore(path string) (*sql.DB, error) {
	if path == "" {
		path = ":memory:"
	}
	if path != ":memory:" {
		if dir := filepath.Dir(path); dir != "." && dir != "" {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, err
			}
		}
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// SQLite допускает один writer: короткий busy_timeout сглаживает
	// конкуренцию goroutine, WAL ускоряет частые точечные записи.
	for _, pragma := range []string{
		"PRAGMA busy_timeout = 5000",
		"PRAGMA journal_mode = WAL",
	} {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			return nil, fmt.Errorf("sqlite %s: %w", pragma, err)
		}
	}
	if _, err := db.Exec(storeSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("sqlite schema: %w", err)
	}
	// Миграция баз, созданных до появления колонки tool: ошибка означает,
	// что колонка уже существует — это нормально.
	_, _ = db.Exec(`ALTER TABLE jobs ADD COLUMN tool TEXT NOT NULL DEFAULT ''`)
	return db, nil
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// fmtTime кодирует время для хранения; нулевое время — пустая строка.
func fmtTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.Format(time.RFC3339Nano)
}

func parseTime(v string) time.Time {
	if v == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339Nano, v)
	if err != nil {
		return time.Time{}
	}
	return t
}

// saveJob вставляет или обновляет строку задачи вместе с runtime-сводкой
// (next_run, last_run и т.д.).
func saveJob(db *sql.DB, j Job) error {
	_, err := db.Exec(`INSERT INTO jobs
		(id, name, prompt, tool, system, schedule, tz, model, enabled,
		 next_run, last_run, last_ok, last_error, last_dur_ms)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			name=excluded.name, prompt=excluded.prompt, tool=excluded.tool,
			system=excluded.system,
			schedule=excluded.schedule, tz=excluded.tz, model=excluded.model,
			enabled=excluded.enabled, next_run=excluded.next_run,
			last_run=excluded.last_run, last_ok=excluded.last_ok,
			last_error=excluded.last_error, last_dur_ms=excluded.last_dur_ms`,
		j.ID, j.Name, j.Prompt, j.Tool, j.System, j.Schedule, j.TZ, j.Model, boolInt(j.Enabled),
		fmtTime(j.NextRun), fmtTime(j.LastRun), boolInt(j.LastOK), j.LastError, j.LastDurMS)
	return err
}

// deleteJob удаляет задачу и её журнал результатов.
func deleteJob(db *sql.DB, id string) error {
	if _, err := db.Exec(`DELETE FROM results WHERE job_id = ?`, id); err != nil {
		return err
	}
	_, err := db.Exec(`DELETE FROM jobs WHERE id = ?`, id)
	return err
}

// loadJobs возвращает все задачи в порядке создания.
func loadJobs(db *sql.DB) ([]Job, error) {
	rows, err := db.Query(`SELECT id, name, prompt, tool, system, schedule, tz, model,
		enabled, next_run, last_run, last_ok, last_error, last_dur_ms
		FROM jobs ORDER BY rowid`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Job
	for rows.Next() {
		var j Job
		var enabled, lastOK int
		var nextRun, lastRun string
		if err := rows.Scan(&j.ID, &j.Name, &j.Prompt, &j.Tool, &j.System, &j.Schedule, &j.TZ, &j.Model,
			&enabled, &nextRun, &lastRun, &lastOK, &j.LastError, &j.LastDurMS); err != nil {
			return nil, err
		}
		j.Enabled = enabled != 0
		j.LastOK = lastOK != 0
		j.NextRun = parseTime(nextRun)
		j.LastRun = parseTime(lastRun)
		out = append(out, j)
	}
	return out, rows.Err()
}

// insertResult добавляет запись журнала и подрезает кольцо до keep последних
// строк на задачу.
func insertResult(db *sql.DB, jobID string, r Result, keep int) error {
	if _, err := db.Exec(`INSERT INTO results (job_id, at, ok, dur_ms, answer, error)
		VALUES (?, ?, ?, ?, ?, ?)`,
		jobID, fmtTime(r.At), boolInt(r.OK), r.DurMS, r.Answer, r.Error); err != nil {
		return err
	}
	if keep > 0 {
		_, err := db.Exec(`DELETE FROM results WHERE job_id = ? AND id NOT IN
			(SELECT id FROM results WHERE job_id = ? ORDER BY id DESC LIMIT ?)`,
			jobID, jobID, keep)
		return err
	}
	return nil
}

// loadResults возвращает последние limit записей задачи в порядке кольца
// (от старых к новым — как отдает Results()).
func loadResults(db *sql.DB, jobID string, limit int) ([]Result, error) {
	if limit <= 0 {
		return nil, nil
	}
	rows, err := db.Query(`SELECT at, ok, dur_ms, answer, error FROM results
		WHERE job_id = ? ORDER BY id DESC LIMIT ?`, jobID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Result
	for rows.Next() {
		var r Result
		var ok int
		var at string
		if err := rows.Scan(&at, &ok, &r.DurMS, &r.Answer, &r.Error); err != nil {
			return nil, err
		}
		r.At = parseTime(at)
		r.OK = ok != 0
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	// Собрали новыми сверху — разворачиваем в порядок кольца.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, nil
}

// legacyJSONPath — прежний файл состояния рядом с базой: scheduler.db →
// scheduler.json. Используется для одноразового импорта при переходе с JSON.
func legacyJSONPath(dbPath string) string {
	if dbPath == "" || dbPath == ":memory:" {
		return ""
	}
	return strings.TrimSuffix(dbPath, filepath.Ext(dbPath)) + ".json"
}

// importLegacyJSON однократно переносит задачи из старого JSON-файла в базу:
// выполняется, только если таблица jobs пуста. Возвращает число перенесённых
// задач.
func importLegacyJSON(db *sql.DB, legacyPath string, logf func(string, ...any)) int {
	if db == nil || legacyPath == "" {
		return 0
	}
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM jobs`).Scan(&n); err != nil || n > 0 {
		return 0
	}
	b, err := os.ReadFile(legacyPath)
	if err != nil {
		return 0 // нет файла — нечего переносить
	}
	var fs fileState
	if err := json.Unmarshal(b, &fs); err != nil {
		logf("scheduler: импорт %s: %v", legacyPath, err)
		return 0
	}
	imported := 0
	for _, j := range fs.Jobs {
		if _, err := ParseSchedule(j.Schedule); err != nil {
			logf("scheduler: импорт: задача %q пропущена: %v", j.Name, err)
			continue
		}
		if err := saveJob(db, j); err != nil {
			logf("scheduler: импорт: задача %q: %v", j.Name, err)
			continue
		}
		imported++
	}
	if imported > 0 {
		logf("scheduler: перенесено задач из %s: %d", legacyPath, imported)
	}
	return imported
}
