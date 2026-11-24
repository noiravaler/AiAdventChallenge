// Package orbit — ядро MCP-сервера астрологии «Orbit»: хранение натальных
// карт, кэш провайдеров и логика инструментов (natal/transit/synastry/…).
package orbit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"ai-chat-go/astrologer"
	"ai-chat-go/openmeteo"
)

// Chart — построенная и сохранённая натальная карта.
type Chart struct {
	Name        string             `json:"name"`
	Subject     astrologer.Subject `json:"subject"`
	Location    openmeteo.Location `json:"location"`
	Aspects     []Aspect           `json:"aspects"`
	ElementDist map[string]any     `json:"element_distribution,omitempty"`
	QualityDist map[string]any     `json:"quality_distribution,omitempty"`
	BuiltAt     time.Time          `json:"built_at"`
}

// BirthLabel — человекочитаемая строка рождения для списков.
func (c *Chart) BirthLabel() string {
	return fmt.Sprintf("%s, %02d.%02d.%d %02d:%02d, %s",
		c.Subject.Name, c.Subject.Day, c.Subject.Month, c.Subject.Year,
		c.Subject.Hour, c.Subject.Minute, c.Location.Label())
}

// Store — персистентный набор карт (JSON-файл) с указателем «последняя».
// Все методы безопасны для конкурентного использования.
type Store struct {
	path string

	mu     sync.Mutex
	charts map[string]*Chart // ключ — нижний регистр имени
	last   string            // ключ последней использованной карты
}

type storeFile struct {
	Charts []*Chart `json:"charts"`
	Last   string   `json:"last"`
}

// NewStore загружает хранилище с диска (отсутствующий файл — не ошибка).
func NewStore(path string) (*Store, error) {
	s := &Store{path: path, charts: map[string]*Chart{}}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, err
	}
	var f storeFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("битый файл хранилища %s: %w", path, err)
	}
	for _, ch := range f.Charts {
		if ch == nil || strings.TrimSpace(ch.Name) == "" {
			continue
		}
		s.charts[key(ch.Name)] = ch
	}
	s.last = key(f.Last)
	if _, ok := s.charts[s.last]; !ok {
		s.last = ""
	}
	return s, nil
}

// Save добавляет/обновляет карту и делает её «последней».
func (s *Store) Save(ch *Chart) error {
	if ch == nil || strings.TrimSpace(ch.Name) == "" {
		return fmt.Errorf("карта без имени не может быть сохранена")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ch.BuiltAt = time.Now()
	s.charts[key(ch.Name)] = ch
	s.last = key(ch.Name)
	return s.persistLocked()
}

// Get возвращает карту по имени; пустое имя — последняя использованная.
func (s *Store) Get(name string) (*Chart, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := key(name)
	if k == "" {
		k = s.last
	}
	if k == "" {
		return nil, fmt.Errorf("натальная карта не задана: постройте её инструментом natal_build_chart")
	}
	ch, ok := s.charts[k]
	if !ok {
		return nil, fmt.Errorf("карта %q не найдена; сохранены: %s", name, strings.Join(s.namesLocked(), ", "))
	}
	s.last = k
	return ch, nil
}

// Last возвращает последнюю карту или nil, если хранилище пусто.
func (s *Store) Last() *Chart {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.last == "" {
		return nil
	}
	return s.charts[s.last]
}

// List возвращает карты, отсортированные по имени.
func (s *Store) List() []*Chart {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*Chart, 0, len(s.charts))
	for _, ch := range s.charts {
		out = append(out, ch)
	}
	sort.Slice(out, func(i, j int) bool { return key(out[i].Name) < key(out[j].Name) })
	return out
}

func (s *Store) namesLocked() []string {
	names := make([]string, 0, len(s.charts))
	for n := range s.charts {
		names = append(names, s.charts[n].Name)
	}
	sort.Strings(names)
	return names
}

// persistLocked атомарно пишет хранилище на диск. Caller должен держать мьютекс.
func (s *Store) persistLocked() error {
	f := storeFile{Last: s.last}
	for _, ch := range s.charts {
		f.Charts = append(f.Charts, ch)
	}
	raw, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(s.path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// key нормализует имя карты в ключ хранилища.
func key(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}
