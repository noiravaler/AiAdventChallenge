package storage

// Инварианты ассистента — жёсткие правила, которые модель не может нарушать.
// Таблица invariants живёт вне диалога и вне трёх слоёв памяти: у записей нет
// ключа сессии/модели, «Удалить чат»/закрытие вкладки их не трогают. Каждое
// изменение (создание, правка, вкл/выкл, удаление) пишется в журнал
// invariants_log в одной транзакции с самим изменением: что → на что → когда.

import (
	"database/sql"
	"fmt"
	"time"
)

// Источники изменения инварианта (колонка source журнала).
const (
	InvSourceUI      = "ui"       // панель «Инварианты» в браузере
	InvSourceCLI     = "cli"      // слэш-команды терминального режима
	InvSourceEnvSeed = "env_seed" // автосев из AI_INVARIANTS при первом старте
)

// Действия в журнале (колонка action).
const (
	InvActionCreate  = "create"
	InvActionUpdate  = "update"
	InvActionEnable  = "enable"
	InvActionDisable = "disable"
	InvActionDelete  = "delete"
)

// Invariant is one hard rule the assistant must never violate. Patterns is an
// optional '|'-separated list of regular expressions checked by the
// deterministic (code) stage of the gate before the LLM judge runs.
type Invariant struct {
	ID       int64     `json:"id"`
	Text     string    `json:"text"`
	Patterns string    `json:"patterns,omitempty"`
	Enabled  bool      `json:"enabled"`
	Created  time.Time `json:"created"`
}

// InvariantLog is one audit entry: an explicit user command changed the
// invariant list (what → what, when, from where). Deleted invariants survive
// only here (old_value), so a wording can be restored from the log.
type InvariantLog struct {
	ID       int64     `json:"id"`
	Action   string    `json:"action"`
	OldValue string    `json:"old_value,omitempty"`
	NewValue string    `json:"new_value,omitempty"`
	Source   string    `json:"source"`
	At       time.Time `json:"at"`
}

// ListInvariants returns every invariant ordered by id (creation order).
func (d *DB) ListInvariants() ([]Invariant, error) {
	rows, err := d.db.Query(
		`SELECT id, text, patterns, enabled, created FROM invariants ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("storage: list invariants: %w", err)
	}
	defer rows.Close()
	var out []Invariant
	for rows.Next() {
		var it Invariant
		var enabled int
		var created int64
		if err := rows.Scan(&it.ID, &it.Text, &it.Patterns, &enabled, &created); err != nil {
			return nil, fmt.Errorf("storage: scan invariant: %w", err)
		}
		it.Enabled = enabled != 0
		it.Created = nanoToTime(created)
		out = append(out, it)
	}
	return out, rows.Err()
}

// InvariantCount returns how many invariants exist (the env-seed check).
func (d *DB) InvariantCount() (int, error) {
	var n int
	if err := d.db.QueryRow(`SELECT COUNT(*) FROM invariants`).Scan(&n); err != nil {
		return 0, fmt.Errorf("storage: count invariants: %w", err)
	}
	return n, nil
}

// insertInvariantLog writes one audit row inside the caller's transaction.
func insertInvariantLog(tx *sql.Tx, action, oldValue, newValue, source string) error {
	if _, err := tx.Exec(
		`INSERT INTO invariants_log (action, old_value, new_value, source, at) VALUES (?, ?, ?, ?, ?)`,
		action, oldValue, newValue, source, time.Now().UnixNano(),
	); err != nil {
		return fmt.Errorf("storage: insert invariants_log: %w", err)
	}
	return nil
}

// AddInvariant stores a new invariant and logs the create action in the same
// transaction. The stored record (with its id) is returned.
func (d *DB) AddInvariant(text, patterns string, source string) (Invariant, error) {
	tx, err := d.db.Begin()
	if err != nil {
		return Invariant{}, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	now := time.Now()
	res, err := tx.Exec(
		`INSERT INTO invariants (text, patterns, enabled, created) VALUES (?, ?, 1, ?)`,
		text, patterns, now.UnixNano(),
	)
	if err != nil {
		return Invariant{}, fmt.Errorf("storage: insert invariant: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Invariant{}, fmt.Errorf("storage: invariant id: %w", err)
	}
	if err := insertInvariantLog(tx, InvActionCreate, "", text, source); err != nil {
		return Invariant{}, err
	}
	if err := tx.Commit(); err != nil {
		return Invariant{}, fmt.Errorf("storage: commit: %w", err)
	}
	return Invariant{ID: id, Text: text, Patterns: patterns, Enabled: true, Created: now}, nil
}

// UpdateInvariant replaces the text/patterns of one invariant and logs
// old → new. Returns the updated record.
func (d *DB) UpdateInvariant(id int64, text, patterns, source string) (Invariant, error) {
	tx, err := d.db.Begin()
	if err != nil {
		return Invariant{}, fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var oldText, oldPatterns string
	var enabled int
	err = tx.QueryRow(
		`SELECT text, patterns, enabled FROM invariants WHERE id=?`, id,
	).Scan(&oldText, &oldPatterns, &enabled)
	switch {
	case err == nil:
	case err == sql.ErrNoRows:
		return Invariant{}, fmt.Errorf("storage: инвариант %d не найден", id)
	default:
		return Invariant{}, fmt.Errorf("storage: find invariant: %w", err)
	}
	if _, err := tx.Exec(
		`UPDATE invariants SET text=?, patterns=? WHERE id=?`, text, patterns, id,
	); err != nil {
		return Invariant{}, fmt.Errorf("storage: update invariant: %w", err)
	}
	if err := insertInvariantLog(tx, InvActionUpdate, oldText, text, source); err != nil {
		return Invariant{}, err
	}
	if err := tx.Commit(); err != nil {
		return Invariant{}, fmt.Errorf("storage: commit: %w", err)
	}
	return Invariant{ID: id, Text: text, Patterns: patterns, Enabled: enabled != 0}, nil
}

// SetInvariantEnabled toggles the gate participation of one invariant and
// logs enable/disable. The text itself is never touched here.
func (d *DB) SetInvariantEnabled(id int64, enabled bool, source string) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var text string
	err = tx.QueryRow(`SELECT text FROM invariants WHERE id=?`, id).Scan(&text)
	switch {
	case err == nil:
	case err == sql.ErrNoRows:
		return fmt.Errorf("storage: инвариант %d не найден", id)
	default:
		return fmt.Errorf("storage: find invariant: %w", err)
	}
	if _, err := tx.Exec(`UPDATE invariants SET enabled=? WHERE id=?`, boolToInt(enabled), id); err != nil {
		return fmt.Errorf("storage: toggle invariant: %w", err)
	}
	action := InvActionDisable
	if enabled {
		action = InvActionEnable
	}
	if err := insertInvariantLog(tx, action, text, text, source); err != nil {
		return err
	}
	return tx.Commit()
}

// DeleteInvariant removes one invariant and logs delete with its old text so
// the wording can be restored from the audit journal.
func (d *DB) DeleteInvariant(id int64, source string) error {
	tx, err := d.db.Begin()
	if err != nil {
		return fmt.Errorf("storage: begin: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // read-only after commit
	var text string
	err = tx.QueryRow(`SELECT text FROM invariants WHERE id=?`, id).Scan(&text)
	switch {
	case err == nil:
	case err == sql.ErrNoRows:
		return fmt.Errorf("storage: инвариант %d не найден", id)
	default:
		return fmt.Errorf("storage: find invariant: %w", err)
	}
	if _, err := tx.Exec(`DELETE FROM invariants WHERE id=?`, id); err != nil {
		return fmt.Errorf("storage: delete invariant: %w", err)
	}
	if err := insertInvariantLog(tx, InvActionDelete, text, "", source); err != nil {
		return err
	}
	return tx.Commit()
}

// ListInvariantLog returns the newest-first audit journal, at most limit rows
// (0 = all).
func (d *DB) ListInvariantLog(limit int) ([]InvariantLog, error) {
	q := `SELECT id, action, old_value, new_value, source, at FROM invariants_log ORDER BY id DESC`
	if limit > 0 {
		q += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := d.db.Query(q)
	if err != nil {
		return nil, fmt.Errorf("storage: list invariants_log: %w", err)
	}
	defer rows.Close()
	var out []InvariantLog
	for rows.Next() {
		var l InvariantLog
		var at int64
		if err := rows.Scan(&l.ID, &l.Action, &l.OldValue, &l.NewValue, &l.Source, &at); err != nil {
			return nil, fmt.Errorf("storage: scan invariants_log: %w", err)
		}
		l.At = nanoToTime(at)
		out = append(out, l)
	}
	return out, rows.Err()
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
