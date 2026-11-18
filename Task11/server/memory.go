// Memory layer: the three-tier memory rules live here. The short-term layer
// is the conversation tree itself (server.go), the working layer is the
// per-branch state (Task/Notes/Summary/Facts of branchState), and this file
// implements the long-term layer — a SQLite-backed store of durable facts —
// plus the routing rules that decide what is written where and what each
// layer contributes to the model prompt.
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"

	"ai-chat-go/api"
	"ai-chat-go/storage"
	"ai-chat-go/tools"
)

// Long-term memory tuning.
const (
	// memProfileLines — how many profile entries are always injected.
	memProfileLines = 10
	// memBlockMaxTokens — the prompt budget of the long-term block
	// (estimated; there is no tokenizer in the stdlib).
	memBlockMaxTokens = 600
	// memMaxItems — capacity of the layer per model; eviction drops the
	// "coldest" items (fewest hits, oldest last injection).
	memMaxItems = 200
	// memStableConfirmations — a compression-extracted fact must survive this
	// many re-extractions (Confirmations >= 2) before it is auto-injected;
	// single-pass candidates stay accessible only via the recall tool.
	memStableConfirmations = 2
	// memMinWordLen — minimal keyword length used for relevance matching.
	memMinWordLen = 3
)

// estimateTokens approximates the token count of a text (~4 bytes per token;
// the stdlib has no tokenizer, exact counts only come from the API usage).
func estimateTokens(s string) int { return (len(s) + 3) / 4 }

// memCache is the in-RAM mirror of one model's long-term memory (write-through
// to the database when it is enabled).
type memCache struct {
	items  []storage.MemoryItem
	loaded bool
	nextID int
}

// memoryCacheLocked returns the per-model memory cache, loading it from the
// database on first use. Callers must hold s.mu.
func (s *Server) memoryCacheLocked(model string) *memCache {
	c := s.memCache[model]
	if c == nil {
		c = &memCache{}
		s.memCache[model] = c
	}
	if !c.loaded {
		c.loaded = true
		if s.db != nil {
			items, err := s.db.ListMemories(model)
			if err != nil {
				log.Printf("memory: не удалось прочитать память модели %q: %v", model, err)
			}
			c.items = items
		}
		c.nextID = nextMemoryID(c.items)
	}
	return c
}

// nextMemoryID derives the next sequential id ("k<N>") from stored items.
func nextMemoryID(items []storage.MemoryItem) int {
	next := 0
	for _, it := range items {
		var n int
		if _, err := fmt.Sscanf(it.ID, "k%d", &n); err == nil && n > next {
			next = n
		}
	}
	return next
}

// upsertMemoryLocked adds a fact or re-confirms the existing one with the
// same (kind, key): the text is refreshed and Confirmations is incremented —
// the fact has just been seen again. Callers must hold s.mu.
func (s *Server) upsertMemoryLocked(model string, it storage.MemoryItem) storage.MemoryItem {
	c := s.memoryCacheLocked(model)
	for i := range c.items {
		if c.items[i].Kind == it.Kind && c.items[i].Key == it.Key {
			c.items[i].Text = it.Text
			c.items[i].Source = it.Source
			c.items[i].Confirmations++
			c.items[i].Updated = time.Now()
			s.persistMemoryUpsert(model, c.items[i])
			return c.items[i]
		}
	}
	c.nextID++
	it.ID = fmt.Sprintf("k%d", c.nextID)
	it.Created = time.Now()
	it.Confirmations = max(1, it.Confirmations)
	c.items = append(c.items, it)
	s.persistMemoryUpsert(model, it)
	return it
}

// persistMemoryUpsert writes one item through to the database (best effort).
func (s *Server) persistMemoryUpsert(model string, it storage.MemoryItem) {
	if s.db == nil {
		return
	}
	if _, err := s.db.UpsertMemory(model, it); err != nil {
		log.Printf("memory: не удалось сохранить запись: %v", err)
	}
}

// deleteMemoryLocked removes an item by id; reports whether it existed.
// Callers must hold s.mu.
func (s *Server) deleteMemoryLocked(model, id string) bool {
	c := s.memoryCacheLocked(model)
	for i := range c.items {
		if c.items[i].ID == id {
			c.items = slices.Delete(c.items, i, i+1)
			if s.db != nil {
				if err := s.db.DeleteMemory(model, id); err != nil {
					log.Printf("memory: не удалось удалить запись: %v", err)
				}
			}
			return true
		}
	}
	return false
}

// touchMemoriesLocked bumps the usage statistics of the items that were just
// injected into a prompt (the eviction feedback). Callers must hold s.mu.
func (s *Server) touchMemoriesLocked(model string, ids []string) {
	if len(ids) == 0 {
		return
	}
	c := s.memoryCacheLocked(model)
	for i := range c.items {
		if slices.Contains(ids, c.items[i].ID) {
			c.items[i].Hits++
			c.items[i].LastHit = time.Now()
		}
	}
	if s.db != nil {
		if err := s.db.TouchMemories(model, ids); err != nil {
			log.Printf("memory: не удалось отметить попадание в промпт: %v", err)
		}
	}
}

// ── Prompt rules ────────────────────────────────────────────────────────────

// longTermBlockLocked assembles the long-term memory system block: the user
// profile goes in ALWAYS (up to memProfileLines lines), decisions and
// knowledge facts only when confirmed (Confirmations >= 2) AND their keys
// overlap the keywords of the last user message. The block is capped by
// memBlockMaxTokens. Returns the block text and the ids of the injected items.
// Callers must hold s.mu.
func (s *Server) longTermBlockLocked(model, lastUser string) (string, []string) {
	c := s.memoryCacheLocked(model)
	if len(c.items) == 0 {
		return "", nil
	}
	msgWords := keywords(lastUser)
	var profile, matched []storage.MemoryItem
	for _, it := range c.items {
		switch it.Kind {
		case storage.KindProfile:
			profile = append(profile, it)
		case storage.KindDecision, storage.KindKnowledge:
			if it.Confirmations >= memStableConfirmations && intersects(keywords(it.Key), msgWords) {
				matched = append(matched, it)
			}
		}
	}
	// Freshness order: the least injected (coldest) items lose the budget race.
	sort.SliceStable(matched, func(i, j int) bool { return matched[i].Hits < matched[j].Hits })

	var b strings.Builder
	var ids []string
	writeItem := func(it storage.MemoryItem) {
		if estimateTokens(b.String()) >= memBlockMaxTokens {
			return
		}
		b.WriteString(memoryKindTag(it.Kind) + " " + it.Text + "\n")
		ids = append(ids, it.ID)
	}
	for _, it := range profile[:min(len(profile), memProfileLines)] {
		writeItem(it)
	}
	for _, it := range matched {
		writeItem(it)
	}
	if b.Len() == 0 {
		return "", nil
	}
	return "Долговременная память о пользователе и проекте (учитывай при ответе):\n" +
		strings.TrimRight(b.String(), "\n"), ids
}

// workingBlockLocked assembles the working-memory system block: the current
// task goal and the branch notes. The summary and the sticky-facts block
// reach the model through their own channels (role summary in the path and
// the facts block below). Callers must hold s.mu.
func (s *Server) workingBlockLocked() string {
	b := s.activeBranchLocked()
	if b == nil {
		return ""
	}
	var parts []string
	if t := strings.TrimSpace(b.Task); t != "" {
		parts = append(parts, "Задача: "+t)
	}
	if n := strings.TrimSpace(b.Notes); n != "" {
		parts = append(parts, "Заметки: "+n)
	}
	if len(parts) == 0 {
		return ""
	}
	return "Рабочая память текущей задачи:\n" + strings.Join(parts, "\n")
}

// ── Facts compression → long-term promotion ─────────────────────────────────

// splitTaskFacts parses the compression response: the "ЗАДАЧА: …" line
// rewrites the current task goal (or clears it with "(нет)"), every other
// line is a «ключ: значение» fact of the sticky block.
func splitTaskFacts(text string) (task, facts string) {
	var f strings.Builder
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if t, ok := cutTaskPrefix(line); ok {
			task = t
			continue
		}
		f.WriteString(line + "\n")
	}
	return task, strings.TrimRight(f.String(), "\n")
}

// cutTaskPrefix strips the «Задача: …» prefix (any letter case) the model
// may put on a line; ok=false means the line is an ordinary note or fact.
func cutTaskPrefix(line string) (string, bool) {
	for _, p := range []string{"ЗАДАЧА:", "Задача:", "задача:"} {
		if t, ok := strings.CutPrefix(line, p); ok {
			t = strings.TrimSpace(t)
			if strings.EqualFold(t, "(нет)") {
				t = ""
			}
			return t, true
		}
	}
	return "", false
}

// cutFactPrefix strips the «ФАКТ: …» prefix (any letter case) the summary
// prompt asks the model to put on extracted fact lines.
func cutFactPrefix(line string) (string, bool) {
	for _, p := range []string{"ФАКТ:", "Факт:", "факт:"} {
		if v, ok := strings.CutPrefix(line, p); ok {
			return strings.TrimSpace(v), true
		}
	}
	return "", false
}

// extractFactLines splits the «ФАКТ: …» lines out of a compression response:
// the lines feed the facts block, the remaining lines are the summary text.
func extractFactLines(text string) (facts, rest string) {
	var f, r strings.Builder
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if v, ok := cutFactPrefix(line); ok {
			if v != "" {
				f.WriteString(v + "\n")
			}
			continue
		}
		r.WriteString(line + "\n")
	}
	return strings.TrimRight(f.String(), "\n"), strings.TrimRight(r.String(), "\n")
}

// maxFactLines caps the facts block built by mergeFacts.
const maxFactLines = 15

// mergeFacts appends new fact lines to the block without duplicates
// (case-insensitive compare) and keeps at most maxFactLines most recent
// lines, so the block cannot grow unboundedly across compressions.
func mergeFacts(block, add string) string {
	var lines []string
	seen := map[string]bool{}
	for _, line := range strings.Split(block+"\n"+add, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		key := strings.ToLower(line)
		if seen[key] {
			continue
		}
		seen[key] = true
		lines = append(lines, line)
	}
	if len(lines) > maxFactLines {
		lines = lines[len(lines)-maxFactLines:]
	}
	return strings.Join(lines, "\n")
}

// splitTaskLine extracts the trailing «ЗАДАЧА: …» line the summary prompt
// asks the model to append: the line becomes the working-memory task and is
// removed from the displayed summary text. Only the last non-empty line is
// considered, so a «задача» mentioned inside the summary is left as is.
func splitTaskLine(text string) (task, rest string) {
	lines := strings.Split(text, "\n")
	last := -1
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) != "" {
			last = i
			break
		}
	}
	if last < 0 {
		return "", text
	}
	line := strings.TrimSpace(lines[last])
	for _, p := range []string{"ЗАДАЧА:", "Задача:", "задача:"} {
		if t, ok := strings.CutPrefix(line, p); ok {
			task = strings.TrimSpace(t)
			if strings.EqualFold(task, "(нет)") {
				task = ""
			}
			rest = strings.TrimSpace(strings.Join(lines[:last], "\n"))
			return task, rest
		}
	}
	return "", text
}

// parseFactLine parses one «[тег] ключ: значение» line into a memory item.
// The optional leading tag selects the kind (по умолчанию — знание).
func parseFactLine(line string) (storage.MemoryItem, bool) {
	kind, rest := memoryKindFromTag(line)
	key, value, ok := strings.Cut(rest, ":")
	key = normalizeFactKey(key)
	value = strings.TrimSpace(value)
	if !ok || len([]rune(key)) < 2 || value == "" {
		return storage.MemoryItem{}, false
	}
	return storage.MemoryItem{Kind: kind, Key: key, Text: key + ": " + value}, true
}

// promoteFactsToLongTerm moves every fact line of the sticky block into the
// long-term store (source=compress): a fact seen for the first time lands
// with Confirmations=1, a repeated one gets its counter bumped — two passes
// make it stable enough for automatic injection. Returns the number of
// promoted facts.
func (s *Server) promoteFactsToLongTerm(model, facts string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, line := range strings.Split(facts, "\n") {
		it, ok := parseFactLine(strings.TrimSpace(line))
		if !ok {
			continue
		}
		it.Source = storage.SourceCompress
		s.upsertMemoryLocked(model, it)
		n++
	}
	s.evictMemoryLocked(model)
	return n
}

// evictMemoryLocked keeps the layer within capacity by dropping the coldest
// items (fewest hits, oldest injection). Callers must hold s.mu.
func (s *Server) evictMemoryLocked(model string) {
	c := s.memoryCacheLocked(model)
	for len(c.items) > memMaxItems {
		cold := 0
		for i := range c.items {
			a, b := c.items[i], c.items[cold]
			if a.Hits < b.Hits || (a.Hits == b.Hits && a.LastHit.Before(b.LastHit)) {
				cold = i
			}
		}
		victim := c.items[cold]
		c.items = slices.Delete(c.items, cold, cold+1)
		if s.db != nil {
			if err := s.db.DeleteMemory(model, victim.ID); err != nil {
				log.Printf("memory: не удалось вытеснить запись: %v", err)
			}
		}
	}
}

// ── Fact parsing helpers ────────────────────────────────────────────────────

// memoryKindTag renders the bracket tag a kind is displayed with in prompts.
func memoryKindTag(kind string) string {
	switch kind {
	case storage.KindProfile:
		return "[профиль]"
	case storage.KindDecision:
		return "[решение]"
	default:
		return "[знание]"
	}
}

// memoryKindFromTag recognizes an optional leading "[профиль]/[решение]/
// [знание]" (or English) tag; the default kind is knowledge.
func memoryKindFromTag(line string) (kind, rest string) {
	for _, pair := range []struct{ tag, kind string }{
		{"[профиль]", storage.KindProfile}, {"[profile]", storage.KindProfile},
		{"[решение]", storage.KindDecision}, {"[decision]", storage.KindDecision},
		{"[знание]", storage.KindKnowledge}, {"[knowledge]", storage.KindKnowledge},
	} {
		if rest, ok := strings.CutPrefix(line, pair.tag); ok {
			return pair.kind, strings.TrimSpace(rest)
		}
	}
	return storage.KindKnowledge, line
}

// normalizeMemoryKind maps a requested kind onto the known values.
func normalizeMemoryKind(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "профиль", "profile":
		return storage.KindProfile
	case "решение", "decision":
		return storage.KindDecision
	default:
		return storage.KindKnowledge
	}
}

// normalizeFactKey lowercases a fact key and strips surrounding punctuation
// so that "Имя," and "имя" deduplicate to the same key.
func normalizeFactKey(k string) string {
	return strings.ToLower(strings.Trim(strings.TrimSpace(k), ".,;:!?—–-\"'()"))
}

// userMemoryItem turns an explicit text (user request or remember tool call)
// into a memory item: the «ключ: значение» form is preserved verbatim, an
// arbitrary phrase gets a key made of its first keywords.
func userMemoryItem(text, kind string) storage.MemoryItem {
	kind = normalizeMemoryKind(kind)
	if key, value, ok := strings.Cut(strings.TrimSpace(text), ":"); ok {
		key = normalizeFactKey(key)
		value = strings.TrimSpace(value)
		if len([]rune(key)) >= 2 && value != "" {
			return storage.MemoryItem{Kind: kind, Key: key, Text: key + ": " + value}
		}
	}
	var ks []string
	for w := range keywords(text) {
		ks = append(ks, w)
	}
	sort.Strings(ks)
	if len(ks) > 3 {
		ks = ks[:3]
	}
	key := strings.Join(ks, " ")
	if key == "" {
		key = normalizeFactKey(text[:min(len(text), 30)])
	}
	return storage.MemoryItem{Kind: kind, Key: key, Text: text}
}

// ── Keyword matching (stdlib relevance, no embeddings) ──────────────────────

// memStopWords — служебные слова, не участвующие в подборе релевантности.
var memStopWords = map[string]struct{}{
	"и": {}, "в": {}, "во": {}, "не": {}, "на": {}, "с": {}, "со": {}, "о": {},
	"об": {}, "по": {}, "для": {}, "что": {}, "как": {}, "это": {}, "том": {},
	"там": {}, "мне": {}, "меня": {}, "тебе": {}, "тебя": {}, "него": {}, "нее": {},
	"она": {}, "они": {}, "мы": {}, "вы": {}, "ты": {}, "я": {}, "а": {}, "но": {},
	"да": {}, "же": {}, "ли": {}, "бы": {}, "the": {}, "and": {}, "for": {},
	"you": {}, "are": {}, "was": {}, "его": {}, "её": {}, "ее": {}, "про": {},
	"под": {}, "над": {}, "при": {}, "без": {}, "есть": {}, "был": {}, "была": {},
}

// keywords splits a text into a normalized word set: lowercase letter/digit
// runs without stop words and words shorter than memMinWordLen.
func keywords(s string) map[string]struct{} {
	out := map[string]struct{}{}
	var cur strings.Builder
	flush := func() {
		w := cur.String()
		cur.Reset()
		if len([]rune(w)) < memMinWordLen {
			return
		}
		if _, stop := memStopWords[w]; stop {
			return
		}
		out[w] = struct{}{}
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// intersects reports whether two keyword sets share at least one word.
func intersects(a, b map[string]struct{}) bool {
	for w := range a {
		if _, ok := b[w]; ok {
			return true
		}
	}
	return false
}

// ── Memory tools (remember / recall / forget / task_note) ───────────────────

// toolDefinitions — инструменты, доступные модели: базовые (время, кубик)
// плюс инструменты памяти (remember/recall/forget/task_note).
func (s *Server) toolDefinitions() []api.Tool {
	return append(tools.Definitions(), tools.MemoryDefinitions()...)
}

// memoryToolExec returns the RunLoop executor for the memory tools; every
// other tool falls back to the standard tools.Execute.
func (s *Server) memoryToolExec(model string) tools.ExecFunc {
	return func(name string, args json.RawMessage) (string, error) {
		switch name {
		case "remember":
			return s.toolRemember(model, args)
		case "recall":
			return s.toolRecall(model, args)
		case "forget":
			return s.toolForget(model, args)
		case "task_note":
			return s.toolTaskNote(args)
		}
		return tools.Execute(name, args)
	}
}

// toolRemember stores an explicit long-term fact (kind from the arguments;
// explicit writes start durable with Confirmations=2 via max(1, …) — see
// upsertMemoryLocked, here we pass 2).
func (s *Server) toolRemember(model string, args json.RawMessage) (string, error) {
	var in struct {
		Text string `json:"text"`
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	text := strings.TrimSpace(in.Text)
	if text == "" {
		return "", fmt.Errorf("не указан текст воспоминания")
	}
	it := userMemoryItem(text, in.Kind)
	it.Source = storage.SourceTool
	it.Confirmations = 2
	s.mu.Lock()
	saved := s.upsertMemoryLocked(model, it)
	s.mu.Unlock()
	return fmt.Sprintf("запомнил (%s): %s", saved.Kind, saved.Text), nil
}

// toolRecall searches the long-term memory by keywords (unconfirmed
// compression candidates are visible here too).
func (s *Server) toolRecall(model string, args json.RawMessage) (string, error) {
	var in struct {
		Query string `json:"query"`
		Kind  string `json:"kind"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	s.mu.Lock()
	items := append([]storage.MemoryItem(nil), s.memoryCacheLocked(model).items...)
	s.mu.Unlock()
	kind := normalizeMemoryKind(in.Kind)
	qw := keywords(in.Query)
	var out []string
	for _, it := range items {
		if in.Kind != "" && it.Kind != kind {
			continue
		}
		if len(qw) > 0 && !intersects(keywords(it.Key+" "+it.Text), qw) {
			continue
		}
		out = append(out, fmt.Sprintf("%s %s %s", it.ID, memoryKindTag(it.Kind), it.Text))
		if len(out) >= 20 {
			break
		}
	}
	if len(out) == 0 {
		return "(в долговременной памяти ничего не найдено)", nil
	}
	return strings.Join(out, "\n"), nil
}

// toolForget removes one long-term item by id.
func (s *Server) toolForget(model string, args json.RawMessage) (string, error) {
	var in struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	if in.ID == "" {
		return "", fmt.Errorf("не указан id записи")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.deleteMemoryLocked(model, in.ID) {
		return "", fmt.Errorf("запись %q не найдена", in.ID)
	}
	return "запись удалена из долговременной памяти", nil
}

// toolTaskNote appends a line to the active branch's working notes (the
// working-memory layer).
func (s *Server) toolTaskNote(args json.RawMessage) (string, error) {
	var in struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	text := strings.TrimSpace(in.Text)
	if text == "" {
		return "", fmt.Errorf("пустая заметка")
	}
	// Заметка, формулирующая саму задачу («Задача: …»), переписывает поле
	// задачи рабочей памяти, а не добавляется в список заметок — иначе
	// в панели появляются дубли вида «Задача: …» с пустым полем задачи.
	if task, ok := cutTaskPrefix(text); ok {
		s.mu.Lock()
		s.ensureBranchStateLocked().Task = task
		s.mu.Unlock()
		s.persist()
		if task == "" {
			return "задача рабочей памяти сброшена", nil
		}
		return "задача рабочей памяти обновлена: " + task, nil
	}
	s.mu.Lock()
	b := s.ensureBranchStateLocked()
	if b.Notes != "" {
		b.Notes += "\n"
	}
	b.Notes += text
	s.mu.Unlock()
	s.persist()
	return "заметка добавлена в рабочую память", nil
}

// ── HTTP API ────────────────────────────────────────────────────────────────

// handleMemoryList serves GET /api/memory: the current model's long-term
// memory items plus the current contents of the other two layers — the
// short-term (displayed dialog turns and their tokens) and the working
// memory (task, notes, facts of the active branch). The tokens map reports
// the current size of every layer: short — the API-reported context size of
// the active branch, working/long — estimates (~4 bytes per token).
func (s *Server) handleMemoryList(w http.ResponseWriter, _ *http.Request) {
	model := s.currentModel()
	s.mu.Lock()
	items := append([]storage.MemoryItem(nil), s.memoryCacheLocked(model).items...)
	// Краткосрочная: реплики активной ветки, которые сейчас в контексте.
	messages := 0
	path, _ := s.activeMessagesLocked()
	for _, e := range path {
		if e.Msg.Role == "user" || (e.Msg.Role == "assistant" && strings.TrimSpace(e.Msg.Content) != "") {
			messages++
		}
	}
	// Рабочая и её токены: состояние активной ветки.
	task, notes, facts, tokens := "", "", "", 0
	if b := s.activeBranchLocked(); b != nil {
		task, notes, facts, tokens = b.Task, b.Notes, b.Facts, b.HistoryTokens
	}
	// После сжатия счётчик HistoryTokens обнуляется — тогда оцениваем
	// объём краткосрочного слоя по фактическому содержимому пути (сводка
	// сжатого контекста тоже входит в слой).
	if tokens <= 0 {
		for _, e := range path {
			if e.Msg.Role == "system" {
				continue
			}
			tokens += estimateTokens(e.Msg.Content) + 4
		}
	}
	workingTokens := estimateTokens(task) + estimateTokens(notes) + estimateTokens(facts)
	longTokens := 0
	for _, it := range items {
		longTokens += estimateTokens(it.Text)
	}
	layers := map[string]bool{"short": s.layerShort, "working": s.layerWorking, "long": s.layerLong}
	s.mu.Unlock()
	writeJSON(w, map[string]any{"model": model, "items": items,
		"short":   map[string]any{"messages": messages, "tokens": tokens},
		"working": map[string]any{"task": task, "notes": notes, "facts": facts, "tokens": workingTokens},
		"long":    map[string]any{"tokens": longTokens},
		"tokens":  map[string]int{"short": tokens, "working": workingTokens, "long": longTokens},
		"layers":  layers})
}

// handleMemoryAdd serves POST /api/memory {"text","kind"}: an explicit user
// write starts durable (Confirmations=2), so the profile/fact is injected
// without waiting for compression to confirm it.
func (s *Server) handleMemoryAdd(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Text string `json:"text"`
		Kind string `json:"kind"`
	}
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, "Некорректный JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	text := strings.TrimSpace(in.Text)
	if text == "" {
		http.Error(w, "Текст записи пустой", http.StatusBadRequest)
		return
	}
	model := s.currentModel()
	it := userMemoryItem(text, in.Kind)
	it.Source = storage.SourceUser
	it.Confirmations = 2
	s.mu.Lock()
	saved := s.upsertMemoryLocked(model, it)
	s.mu.Unlock()
	writeJSON(w, map[string]any{"ok": true, "item": saved})
}

// handleMemoryLayers serves POST /api/memory/layers {"short","working",
// "long"}: the layer toggles of the memory panel. A switched-off layer is not
// sent to the model (as if it did not exist), its data stays intact.
func (s *Server) handleMemoryLayers(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Short   *bool `json:"short"`
		Working *bool `json:"working"`
		Long    *bool `json:"long"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&in); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	if in.Short != nil {
		s.layerShort = *in.Short
	}
	if in.Working != nil {
		s.layerWorking = *in.Working
	}
	if in.Long != nil {
		s.layerLong = *in.Long
	}
	layers := map[string]bool{"short": s.layerShort, "working": s.layerWorking, "long": s.layerLong}
	s.mu.Unlock()
	log.Printf("memory: слои short=%v working=%v long=%v", layers["short"], layers["working"], layers["long"])
	writeJSON(w, map[string]any{"ok": true, "layers": layers})
}

// handleMemoryDelete serves DELETE /api/memory?id=k3.
func (s *Server) handleMemoryDelete(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Не указан id записи", http.StatusBadRequest)
		return
	}
	model := s.currentModel()
	s.mu.Lock()
	ok := s.deleteMemoryLocked(model, id)
	s.mu.Unlock()
	if !ok {
		http.Error(w, "Запись не найдена", http.StatusNotFound)
		return
	}
	writeJSON(w, map[string]bool{"ok": true})
}

// memoryDebug returns the "Память" info for the details panel: the long-term
// block actually injected into the last request and the item count.
func (s *Server) memoryDebug() (block string, count int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	block = s.lastMemoryBlock
	if c, ok := s.memCache[s.model]; ok {
		count = len(c.items)
	}
	return block, count
}
