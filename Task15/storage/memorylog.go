package storage

// Журнал изменений долговременной памяти: каждая замена, удаление и истечение
// факта фиксируется в таблице memories_log в одной транзакции с самим
// изменением — что было → что стало, когда, откуда (см. storage/invariants.go
// для журнала инвариантов). Для append-фактов журнал дополняет картину
// истории: сами записи-строки хранят время записи, журнал — их судьбу.

import (
	"database/sql"
	"fmt"
	"time"
)

// Действия в журнале памяти (колонка action).
const (
	MemLogInsert  = "insert"  // новый факт
	MemLogAppend  = "append"  // пополнение списка (append-факт)
	MemLogReplace = "replace" // замена значения (latest-факт): old → new
	MemLogDelete  = "delete"  // явное удаление (forget, UI, вытеснение)
	MemLogExpire  = "expire"  // автоудаление по завершении задачи (ttl=task)
	MemLogReset   = "reset"   // массовая очистка слоя (корзина «Удалить …»)
)

// MemoryLog is one audit entry of the long-term memory: a write changed the
// memory of one tab (what → what, when, from where). Replaced values survive
// only here (old_value), so a fact can be restored from the journal.
type MemoryLog struct {
	ID       int64     `json:"id"`
	Session  string    `json:"session"`
	MemoryID string    `json:"memory_id,omitempty"`
	Action   string    `json:"action"`
	Kind     string    `json:"kind,omitempty"`
	Key      string    `json:"key,omitempty"`
	OldValue string    `json:"old_value,omitempty"`
	NewValue string    `json:"new_value,omitempty"`
	Source   string    `json:"source"`
	At       time.Time `json:"at"`
}

// insertMemoryLog writes one audit row inside the caller's transaction.
func insertMemoryLog(tx *sql.Tx, session, memoryID, action, kind, key, oldValue, newValue, source string) error {
	if _, err := tx.Exec(
		`INSERT INTO memories_log (session, memory_id, action, kind, key, old_value, new_value, source, at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		session, memoryID, action, kind, key, oldValue, newValue, source, time.Now().UnixNano(),
	); err != nil {
		return fmt.Errorf("storage: insert memories_log: %w", err)
	}
	return nil
}

// ListMemoryLog returns the newest-first audit journal of one tab's long-term
// memory, at most limit rows (0 = all).
func (d *DB) ListMemoryLog(session string, limit int) ([]MemoryLog, error) {
	q := `SELECT id, session, memory_id, action, kind, key, old_value, new_value, source, at
	      FROM memories_log WHERE session=? ORDER BY id DESC`
	if limit > 0 {
		q += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := d.db.Query(q, session)
	if err != nil {
		return nil, fmt.Errorf("storage: list memories_log: %w", err)
	}
	defer rows.Close()
	var out []MemoryLog
	for rows.Next() {
		var l MemoryLog
		var at int64
		if err := rows.Scan(&l.ID, &l.Session, &l.MemoryID, &l.Action, &l.Kind, &l.Key, &l.OldValue, &l.NewValue, &l.Source, &at); err != nil {
			return nil, fmt.Errorf("storage: scan memories_log: %w", err)
		}
		l.At = nanoToTime(at)
		out = append(out, l)
	}
	return out, rows.Err()
}
