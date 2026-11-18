// Package server serves the browser chat interface.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	testassets "ai-chat-go"
	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/storage"
	"ai-chat-go/tools"
)

//go:embed static/index.html
var indexHTML []byte

// defaultNoReplyTimeout aborts the request when the model produces no token
// (visible content or reasoning trace) for this long. It can be overridden per
// request by the "timeout" field (5..30 s) from the UI slider.
const defaultNoReplyTimeout = 20 * time.Second

// modelRefreshTimeout caps how long /api/models spends querying the upstream
// /models endpoint before falling back to the last known model list.
const modelRefreshTimeout = 5 * time.Second

// Server keeps the conversation state for the browser chat.
type Server struct {
	client *api.Client
	cfg    config.Config

	genMu sync.Mutex // serializes chat generations
	mu    sync.Mutex // guards history and model selection

	// hist holds every conversation-tree node ever created for the active
	// model (append-only; cleared on reset/model switch/dialog load). The
	// dialog the user sees is the materialized path from the tree root to
	// activeLeaf — see activeMessagesLocked.
	hist []histEntry

	// byID indexes tree nodes by id (rebuilt on load); nextID numbers new
	// nodes ("m1", "m2", …). states holds per-leaf branch state — the
	// compression bookkeeping lives on the branch, not on tree nodes, which
	// stay immutable. switchable lists the leaves offered by the branch
	// switcher (in discovery order): a leaf becomes switchable when the
	// active path forks away from it. states may cover more leaves than
	// switchable (every reply leaf keeps its own token/facts state).
	byID       map[string]int
	nextID     int
	states     map[string]*branchState
	switchable []string
	activeLeaf string

	model          string        // currently selected model
	models         []string      // models offered in the contact sidebar
	noReplyTimeout time.Duration // abort when the model stays silent this long

	reserve      *api.Client // independent fallback provider (not via GPUStack)
	reserveModel string      // model served by the reserve provider

	lastDetail string // compact JSON of the last error/limit detail (Детали panel)

	// lastTrace keeps the last chat exchange's aggregated token accounting /
	// duration, served to the "Детали" panel. dialogSteps accumulates the
	// interaction steps across the whole dialog (not per exchange).
	lastTokens     int
	lastPrompt     int
	lastCompletion int
	lastTimeMS     int64
	dialogSteps    []tools.RunStep

	// On-the-fly compression settings sent by the client with each chat
	// request (and the scenario run): whether to compress, which strategy to
	// use, how many last messages form the kept window and after how many
	// messages the compression is triggered.
	compressEnabled   bool
	compressStrategy  string // summary | sliding | sticky
	compressKeep      int
	compressThreshold int

	// Layer toggles (RAM-only, like the compression settings): a switched-off
	// layer is not sent to the model at all, its data stays intact.
	layerShort   bool
	layerWorking bool
	layerLong    bool

	// inDialogRun is true while a scenario replay (/api/dialog/run) is in
	// progress: branch operations are refused (409) while it is held, so the
	// active path cannot move under the running replay.
	inDialogRun bool

	// inDialogRun is true while a scenario replay (/api/dialog/run) is in
	// progress: branch operations are refused (409) while it is held, so the
	// active path cannot move under the running replay.

	// compressSeq increments on every compression; the UI compares it to
	// re-render the chat when old messages were dropped. lastCompress holds
	// the info of the most recent compression for the "Сжатие" block.
	compressSeq  int
	lastCompress map[string]any

	// lastChatReq/lastChatResp freeze the request/response of the last chat
	// exchange (not the follow-up compression/validation calls) for the
	// "Запрос" block of the details panel.
	lastChatReq  string
	lastChatResp string

	historyFile string // legacy JSON context path (migration source)

	// Долговременная память (слой 3): SQLite-хранилище (nil = персистентность
	// выключена) и кэш записей по моделям. Рабочая память — поля Task/Notes
	// branchState (слой 2), краткосрочная — само дерево hist (слой 1).
	db              *storage.DB
	memCache        map[string]*memCache
	lastMemoryBlock string // долговременный блок последнего запроса («Детали»)
}

// histEntry is a conversation-tree node: the message plus the wall-clock time
// it was produced. The timestamp lets the UI group messages by day and show
// each message's time; it is also persisted so it survives a restart. ID
// ("m1", "m2", …) and Parent (the node this one replies to, empty for the
// tree root) wire the node into the conversation tree; the displayed dialog
// is the path from the root to the active leaf.
type histEntry struct {
	ID     string
	Parent string
	At     time.Time
	Msg    api.Message
}

// appendNodeLocked adds a message to the tree as a child of parent and
// returns the created node. Callers must hold s.mu.
func (s *Server) appendNodeLocked(parent string, msg api.Message, at time.Time) histEntry {
	s.nextID++
	e := histEntry{
		ID:     "m" + strconv.Itoa(s.nextID),
		Parent: parent,
		At:     at,
		Msg:    msg,
	}
	s.hist = append(s.hist, e)
	s.byID[e.ID] = len(s.hist) - 1
	// У узла появился потомок — он больше не лист: оставленная в
	// переключателе запись превратилась бы в «пустую» ветку-фантом, и новое
	// сообщение выглядело бы как ещё одна ветка вместо продолжения.
	if parent != "" {
		s.switchable = slices.DeleteFunc(s.switchable, func(id string) bool { return id == parent })
	}
	return e
}

// nodeLocked returns the tree node with the given id (ok=false when unknown).
// Callers must hold s.mu.
func (s *Server) nodeLocked(id string) (histEntry, bool) {
	if id == "" {
		return histEntry{}, false
	}
	idx, ok := s.byID[id]
	if !ok || idx < 0 || idx >= len(s.hist) {
		return histEntry{}, false
	}
	return s.hist[idx], true
}

// removeNodeLocked deletes a tree node by id (and any dangling state) and
// reports whether it existed. Callers must hold s.mu.
func (s *Server) removeNodeLocked(id string) bool {
	idx, ok := s.byID[id]
	if !ok || idx < 0 || idx >= len(s.hist) {
		return false
	}
	s.hist = append(s.hist[:idx], s.hist[idx+1:]...)
	delete(s.byID, id)
	// Reindex: byID maps ids to positions in the slice.
	for i := idx; i < len(s.hist); i++ {
		s.byID[s.hist[i].ID] = i
	}
	delete(s.states, id)
	return true
}

// isAncestorOrSelfLocked reports whether anc lies on the raw parent chain of
// id (or equals it). Callers must hold s.mu.
func (s *Server) isAncestorOrSelfLocked(anc, id string) bool {
	for cur := id; cur != ""; {
		if cur == anc {
			return true
		}
		e, ok := s.nodeLocked(cur)
		if !ok {
			return false
		}
		cur = e.Parent
	}
	return false
}

// pathToLeafLocked walks the raw parent chain from leaf to the tree root and
// returns the nodes root-first. Callers must hold s.mu.
func (s *Server) pathToLeafLocked(leaf string) []histEntry {
	var rev []histEntry
	for cur := leaf; cur != ""; {
		e, ok := s.nodeLocked(cur)
		if !ok {
			break
		}
		rev = append(rev, e)
		cur = e.Parent
	}
	out := make([]histEntry, 0, len(rev)+1)
	for i := len(rev) - 1; i >= 0; i-- {
		out = append(out, rev[i])
	}
	return out
}

// activeBranchLocked resolves the effective branch state of the active leaf:
// its own state, or the nearest ancestor's (inherited by continuation), or
// nil. Callers must hold s.mu.
func (s *Server) activeBranchLocked() *branchState {
	return s.branchStateForLocked(s.activeLeaf)
}

// branchStateForLocked walks the raw parent chain from leaf upwards and
// returns the first node's state — a leaf's own state shadows the inherited
// one (a fork inside a covered range stores an empty state exactly for that).
// The empty leaf ("") is the root of a just-wiped tree (no system prompt
// configured): its own state — e.g. the working memory kept across a chat
// reset — is resolvable and inherited by the first future message. Callers
// must hold s.mu.
func (s *Server) branchStateForLocked(leaf string) *branchState {
	for cur := leaf; ; {
		if b, ok := s.states[cur]; ok && b != nil {
			return b
		}
		if cur == "" {
			return nil
		}
		e, ok := s.nodeLocked(cur)
		if !ok {
			return nil
		}
		cur = e.Parent
	}
}

// ensureBranchStateLocked returns the active leaf's own state, creating it by
// inheriting the resolved ancestor state (or empty) when missing. Callers
// must hold s.mu.
func (s *Server) ensureBranchStateLocked() *branchState {
	if b := s.states[s.activeLeaf]; b != nil {
		return b
	}
	if inh := s.branchStateForLocked(s.activeLeaf); inh != nil {
		cp := *inh
		cp.Leaf = s.activeLeaf
		s.states[s.activeLeaf] = &cp
		return &cp
	}
	b := &branchState{Leaf: s.activeLeaf}
	s.states[s.activeLeaf] = b
	return b
}

// registerSwitchableLocked marks the active leaf as a switchable branch
// point (the switcher lists it once the active path moves elsewhere).
// Callers must hold s.mu.
func (s *Server) registerSwitchableLocked() {
	leaf := s.activeLeaf
	if leaf == "" {
		return
	}
	s.ensureBranchStateLocked()
	if !slices.Contains(s.switchable, leaf) {
		s.switchable = append(s.switchable, leaf)
	}
}

// branchTitleLocked picks a branch preview: the first user message on the
// path (falls back to the first non-empty content).
func branchTitleLocked(path []histEntry) string {
	fallback := ""
	for _, e := range path {
		switch e.Msg.Role {
		case "user":
			return strings.TrimSpace(e.Msg.Content)
		case "assistant":
			if fallback == "" && strings.TrimSpace(e.Msg.Content) != "" {
				fallback = strings.TrimSpace(e.Msg.Content)
			}
		}
	}
	return fallback
}

// inheritBranchStateLocked builds the branch state for a fork at id (an
// existing tree node): the fork records id as its fork point — its own
// compressions may only fold messages after it — and freezes the resolved
// active state's coverage as an inherited prefix stage when the fork point
// sits at/after the covered range (the Sticky Facts join the same way). A
// fork inside the covered range starts with clean state: the nodes between
// the fork point and the covered range become visible again.
// Callers must hold s.mu.
func (s *Server) inheritBranchStateLocked(id string) *branchState {
	b := &branchState{Leaf: id, Title: branchTitleLocked(s.pathToLeafLocked(id)), ForkUpTo: id}
	if cur := s.activeBranchLocked(); cur != nil {
		b.Frozen = append([]foldStage(nil), cur.Frozen...)
		if cur.CoveredUpTo != "" && s.isAncestorOrSelfLocked(cur.CoveredUpTo, id) {
			b.Frozen = append(b.Frozen, foldStage{
				CoveredUpTo: cur.CoveredUpTo,
				Summary:     cur.Summary,
				Plaque:      cur.Plaque,
			})
		}
		if cur.FactsUpTo == "" || s.isAncestorOrSelfLocked(cur.FactsUpTo, id) {
			b.Facts = cur.Facts
			b.FactsUpTo = cur.FactsUpTo
		}
		b.HistoryTokens = cur.HistoryTokens
	}
	return b
}

// activeMessagesLocked materializes the dialog the user sees and the model
// receives: the system prompt, then the branch's inherited prefix stages
// (each frozen stage folds its range and shows its plaque — nodes between
// stages stay visible), the shared tail up to and including the fork point
// (it anchors the branch and never folds), the branch's own compression
// plaque and the rest of the path. The second return value is the fork
// point's index in the materialized path (-1 when the whole dialog is the
// branch's own: the root branch, or the fork point already folded by an
// inherited stage — then the boundary is that stage's plaque). Callers must
// hold s.mu.
func (s *Server) activeMessagesLocked() ([]histEntry, int) {
	path := s.pathToLeafLocked(s.activeLeaf)
	b := s.activeBranchLocked()
	forkIdx := -1
	out := make([]histEntry, 0, len(path)+4)
	if len(path) > 0 && path[0].Msg.Role == "system" {
		out = append(out, path[0])
		path = path[1:]
	} else if s.cfg.System != "" {
		// The covered range can extend over the tree root: the system prompt
		// always governs, so re-seed it in materialized form.
		out = append(out, histEntry{At: time.Now(), Msg: api.SystemMessage(s.cfg.System)})
	}
	dropThrough := func(covered string) {
		for i, e := range path {
			if e.ID == covered {
				path = path[i+1:]
				return
			}
		}
	}
	addPlaque := func(text, role string) {
		out = append(out, histEntry{At: time.Now(), Msg: api.Message{Role: role, Content: text}})
	}
	if b != nil {
		// Frozen prefix stages inherited at fork time.
		for _, st := range b.Frozen {
			dropThrough(st.CoveredUpTo)
			if st.Summary != "" {
				addPlaque(st.Summary, "summary")
			} else if st.Plaque != "" {
				addPlaque(st.Plaque, "marker")
			}
		}
		// The shared tail ends at the fork point: keep it visible (it
		// anchors the branch's switcher pill) and mark the boundary.
		if b.ForkUpTo != "" {
			found := false
			for i, e := range path {
				if e.ID == b.ForkUpTo {
					out = append(out, path[:i+1]...)
					path = path[i+1:]
					forkIdx = len(out) - 1
					found = true
					break
				}
			}
			if !found {
				// The fork point itself is folded by the last inherited
				// stage: everything displayed after the stages is the
				// branch's own.
				forkIdx = len(out) - 1
			}
		}
		// The branch's own compression.
		dropThrough(b.CoveredUpTo)
		if b.Summary != "" {
			addPlaque(b.Summary, "summary")
		} else if b.Plaque != "" {
			addPlaque(b.Plaque, "marker")
		}
	}
	return append(out, path...), forkIdx
}

// forkPointOfLocked derives the divergence point of a leaf: the last node on
// its path that has more than one child (the fork the branch grew from).
// States saved before the fork point was recorded carry none; this
// derivation restores it so their own compression keeps the shared prefix
// visible. Callers must hold s.mu.
func (s *Server) forkPointOfLocked(leaf string) string {
	kids := make(map[string]int, len(s.hist))
	for _, e := range s.hist {
		if e.Parent != "" {
			kids[e.Parent]++
		}
	}
	fork := ""
	for _, e := range s.pathToLeafLocked(leaf) {
		if kids[e.ID] > 1 {
			fork = e.ID
		}
	}
	return fork
}

// branchLeaves lists the switchable leaves (discovery order) plus the active
// leaf last when it is not among them. Stale entries are dropped: a node
// that has gained children is no longer a branch end, and keeping it in the
// switcher would offer an empty phantom branch.
func (s *Server) branchLeaves() []string {
	hasChild := make(map[string]bool, len(s.hist))
	for _, e := range s.hist {
		if e.Parent != "" {
			hasChild[e.Parent] = true
		}
	}
	out := make([]string, 0, len(s.switchable)+1)
	for _, id := range s.switchable {
		// Узел с потомками обычно уже не лист (фантом пустой ветки), но
		// точка ветвления с собственным состоянием — это живая пустая
		// форк-ветка («Ветвление отсюда», от которой ещё не ушли новым
		// сообщением): она обязана оставаться в переключателе и после
		// ухода с неё, иначе вернуться к форку нечем («дерево стёрто»).
		// Продолженные ветки снимает appendNodeLocked в момент появления
		// потомка; проверка states страхует записи из старых файлов.
		if hasChild[id] && s.states[id] == nil {
			continue
		}
		if !slices.Contains(out, id) {
			out = append(out, id)
		}
	}
	if s.activeLeaf != "" && !slices.Contains(out, s.activeLeaf) {
		out = append(out, s.activeLeaf)
	}
	return out
}

// New creates a web chat server with the system prompt from cfg.
func New(client *api.Client, cfg config.Config) *Server {
	s := &Server{client: client, cfg: cfg, noReplyTimeout: defaultNoReplyTimeout}
	s.models = append([]string(nil), cfg.Models...)
	if len(s.models) == 0 {
		s.models = append([]string(nil), config.DefaultModels...)
	}
	s.model = cfg.Model
	if s.model == "" && len(s.models) > 0 {
		s.model = s.models[0]
	}
	// Independent reserve provider: not related to the main endpoint at all.
	if cfg.ReserveBaseURL != "" {
		s.reserve = api.New(cfg.ReserveBaseURL, cfg.ReserveAPIKey, cfg.Timeout)
		s.reserve.DebugLog = nil // панель «Детали» читает обмены из памяти клиента
		s.reserveModel = cfg.ReserveModel
	}
	s.byID = make(map[string]int)
	s.states = make(map[string]*branchState)
	if cfg.System != "" {
		e := s.appendNodeLocked("", api.SystemMessage(cfg.System), time.Now())
		s.activeLeaf = e.ID
	}
	s.historyFile = cfg.HistoryFile
	s.memCache = make(map[string]*memCache)
	// SQLite: долговременная память и персистентность контекста. При недоступной
	// базе работа продолжается без сохранения (как с пустым AI_DB_FILE).
	if cfg.DBFile != "" {
		db, err := storage.Open(cfg.DBFile)
		if err != nil {
			log.Printf("memory: SQLite недоступен (%v); персистентность отключена", err)
		} else {
			s.db = db
		}
	}
	s.compressStrategy = "summary"
	s.compressKeep = 5
	s.compressThreshold = 10
	s.layerShort, s.layerWorking, s.layerLong = true, true, true
	s.restore()
	return s
}

// restore reloads a persisted per-model conversation (and the active model) so
// a restart continues where the user left off. It is a no-op when persistence
// is disabled (empty historyFile). The context with the newest activity is
// chosen as the active model. The system message is re-seeded from the current
// configuration rather than stored.
func (s *Server) restore() {
	if s.db != nil {
		s.restoreSQLite()
		return
	}
	if s.historyFile == "" {
		return
	}
	s.migrateLegacy()

	// Pick the context file with the newest message as the active model.
	// On an exact tie (older builds seeded a model switch by copying the
	// whole context, producing twin files with identical timestamps) prefer
	// the file saved most recently — it reflects the latest session.
	bestFile, bestModel := "", ""
	var bestAt time.Time
	var bestMod time.Time
	for _, f := range s.contextFiles() {
		st, err := loadState(f)
		if err != nil || st == nil {
			continue
		}
		latest := latestAt(st.Messages)
		mod, _ := os.Stat(f)
		var modAt time.Time
		if mod != nil {
			modAt = mod.ModTime()
		}
		if latest.After(bestAt) || (latest.Equal(bestAt) && modAt.After(bestMod)) {
			bestAt = latest
			bestMod = modAt
			bestFile, bestModel = f, st.Model
		}
	}
	if bestFile == "" {
		return
	}
	// Restore the active model only if it is still a known choice (or reserve).
	if bestModel != "" {
		for _, m := range s.models {
			if m == bestModel {
				s.model = bestModel
				break
			}
		}
		if s.isReserve(bestModel) {
			s.model = bestModel
		}
	}
	st, err := loadState(bestFile)
	if err != nil || st == nil {
		return
	}
	s.loadTreeLocked(st)
}

// restoreSQLite continues the most recent SQLite-persisted conversation: the
// active model is the one with the newest message. Legacy JSON contexts are
// imported first, so the switch to SQLite loses nothing.
func (s *Server) restoreSQLite() {
	s.migrateLegacyToSQLite()
	model, err := s.db.LatestSessionModel()
	if err != nil {
		log.Printf("memory: не удалось определить последнюю сессию: %v", err)
	}
	if model != "" && (slices.Contains(s.models, model) || s.isReserve(model)) {
		s.model = model
	}
	st, err := s.db.LoadSession(s.model)
	if err != nil {
		log.Printf("memory: не удалось загрузить сессию модели %q: %v", s.model, err)
		return
	}
	if st == nil {
		return
	}
	s.loadTreeLocked(st)
}

// migrateLegacyToSQLite imports any legacy per-model JSON context files into
// the database (once: imported files are renamed to *.migrated). Models that
// already have a SQLite session are left untouched.
func (s *Server) migrateLegacyToSQLite() {
	if s.historyFile == "" {
		return
	}
	files := s.contextFiles()
	if len(files) == 0 {
		return
	}
	for _, f := range files {
		if strings.HasSuffix(f, ".migrated") {
			continue
		}
		st, err := loadState(f)
		if err != nil || st == nil || st.Model == "" {
			continue
		}
		if existing, err := s.db.LoadSession(st.Model); err != nil || existing != nil {
			continue // already migrated / newer data present
		}
		// Old flat files have no message ids; assign them before the insert,
		// otherwise the messages(model, id) unique index rejects the batch.
		st = normalizeLegacyState(st)
		if err := s.db.SaveSession(*st); err != nil {
			log.Printf("memory: миграция %s не удалась: %v", f, err)
			continue
		}
		if err := os.Rename(f, f+".migrated"); err != nil {
			log.Printf("memory: не удалось переименовать %s: %v", f, err)
			continue
		}
		log.Printf("memory: мигрирован контекст модели %q: %s -> SQLite", st.Model, f)
	}
}

// loadTreeLocked installs a persisted context into the tree: records with
// ids/parents restore as-is (branches included); a legacy flat file (no ids)
// migrates to one linear branch so nothing is lost across the upgrade.
// Callers must hold s.mu.
func (s *Server) loadTreeLocked(st *historyState) {
	s.hist = make([]histEntry, 0, len(st.Messages)+1)
	s.byID = make(map[string]int, len(st.Messages)+1)
	s.states = make(map[string]*branchState)
	s.switchable = nil
	s.activeLeaf = ""
	s.nextID = 0
	if s.cfg.System != "" {
		e := s.appendNodeLocked("", api.SystemMessage(s.cfg.System), time.Now())
		s.activeLeaf = e.ID
	}
	prev := s.activeLeaf
	maxID := 0
	for _, r := range st.Messages {
		id, parent := r.ID, r.Parent
		if id == "" {
			// Legacy record: number it and chain it linearly (skip ids already
			// taken — the re-seeded system node occupies "m1").
			parent = prev
			for {
				maxID++
				id = "m" + strconv.Itoa(maxID)
				if _, dup := s.byID[id]; !dup {
					break
				}
			}
		} else {
			if n, err := strconv.Atoi(strings.TrimPrefix(id, "m")); err == nil && n > maxID {
				maxID = n
			}
			if _, dup := s.byID[id]; dup {
				// Defensive: duplicate id in a corrupted file — renumber.
				maxID++
				id = "m" + strconv.Itoa(maxID)
				parent = prev
			}
		}
		e := histEntry{ID: id, Parent: parent, At: r.At, Msg: r.Message()}
		s.hist = append(s.hist, e)
		s.byID[id] = len(s.hist) - 1
		prev = id
	}
	s.nextID = maxID
	if st.NextID > s.nextID {
		s.nextID = st.NextID
	}
	for _, b := range st.Branches {
		if _, ok := s.byID[b.Leaf]; !ok {
			continue // state for an unknown leaf
		}
		if b.ForkUpTo == "" {
			// Legacy file: restore the fork point this branch grew from, so
			// its own compression folds only the branch's own messages and
			// the shared prefix (with the fork point) stays visible.
			b.ForkUpTo = s.forkPointOfLocked(b.Leaf)
		}
		if b.ForkUpTo != "" && b.CoveredUpTo != "" && s.isAncestorOrSelfLocked(b.CoveredUpTo, b.ForkUpTo) {
			// Legacy coverage lies entirely within the shared prefix: freeze
			// it as an inherited stage so the prefix keeps its plaque and
			// the branch starts with no own coverage.
			b.Frozen = []foldStage{{CoveredUpTo: b.CoveredUpTo, Summary: b.Summary, Plaque: b.Plaque}}
			b.CoveredUpTo, b.Summary, b.Plaque = "", "", ""
		}
		bb := b
		s.states[bb.Leaf] = &bb
	}
	for _, leaf := range st.Switchable {
		if s.states[leaf] == nil {
			continue
		}
		if !slices.Contains(s.switchable, leaf) {
			s.switchable = append(s.switchable, leaf)
		}
	}
	if len(st.Branches) == 0 && prev != "" && prev != s.activeLeaf {
		// Legacy migration: one linear branch ending at the last record,
		// carrying the legacy facts/history-tokens state.
		s.states[prev] = &branchState{
			Leaf:          prev,
			Title:         branchTitleLocked(s.pathToLeafLocked(prev)),
			Facts:         st.Facts,
			FactsUpTo:     prev,
			HistoryTokens: st.HistoryTokens,
		}
		if !slices.Contains(s.switchable, prev) {
			s.switchable = append(s.switchable, prev)
		}
	}
	// Restore the active branch; fall back to the last record.
	if st.Active != "" {
		if _, ok := s.byID[st.Active]; ok {
			s.activeLeaf = st.Active
		}
	} else if prev != "" {
		s.activeLeaf = prev
	}
}

// migrateLegacy moves any context files the older implementation left in the
// working directory (the single-file context and any per-model files) into the
// dedicated history folder, so no conversation is lost across this upgrade. It
// is a no-op if there is nothing to migrate or a per-model file already exists.
func (s *Server) migrateLegacy() {
	if s.historyFile == "" {
		return
	}
	name := filepath.Base(s.historyFile)
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	targetDir := filepath.Dir(s.historyFile)
	// historyFile itself may be a legacy single-file context (e.g. a custom
	// AI_HISTORY_FILE pointing at one file that predates per-model storage).
	s.migrateOne(s.historyFile, targetDir)
	// Per-model legacy files lived next to the process (the working directory).
	matches, _ := filepath.Glob(filepath.Join(".", stem+"*"+ext))
	for _, f := range matches {
		if filepath.Dir(f) == targetDir {
			continue // already under the dedicated folder
		}
		s.migrateOne(f, targetDir)
	}
}

// migrateOne migrates a single legacy context file (containing a Model) into
// the per-model file for that model, unless a per-model file already exists.
func (s *Server) migrateOne(f, _ string) {
	st, err := loadState(f)
	if err != nil || st == nil || st.Model == "" {
		return
	}
	dst := s.contextFileFor(st.Model)
	if dst == "" || dst == f {
		return
	}
	if _, err := os.Stat(dst); err == nil {
		return // per-model file already present
	}
	if err := saveState(dst, historyState{Model: st.Model, Messages: st.Messages}); err == nil {
		log.Printf("history: migrated legacy %s -> %s", f, dst)
		_ = os.Remove(f)
	}
}

// persist writes the current conversation (non-system turns) and the active
// model to its own per-model file. It is called after every history change and
// is a no-op when persistence is disabled.
func (s *Server) persist() {
	s.mu.Lock()
	var msgs []chatRecord
	for _, e := range s.hist {
		if e.Msg.Role != "system" {
			msgs = append(msgs, recordOf(e))
		}
	}
	// Top-level HistoryTokens/Facts mirror the active branch for backwards
	// compatibility; the full per-leaf state travels in Branches.
	historyTokens, facts := 0, ""
	if b := s.activeBranchLocked(); b != nil {
		historyTokens, facts = b.HistoryTokens, b.Facts
	}
	branchStates := make([]branchState, 0, len(s.states))
	for _, b := range s.states {
		if b != nil {
			branchStates = append(branchStates, *b)
		}
	}
	sort.Slice(branchStates, func(i, j int) bool { return branchStates[i].Leaf < branchStates[j].Leaf })
	switchable := append([]string(nil), s.switchable...)
	st := historyState{
		Model:         s.model,
		HistoryTokens: historyTokens,
		Facts:         facts,
		Messages:      msgs,
		Branches:      branchStates,
		Switchable:    switchable,
		Active:        s.activeLeaf,
		NextID:        s.nextID,
	}
	db := s.db
	path := s.contextFileFor(s.model)
	s.mu.Unlock()
	// SQLite — основной канал персистентности (контекст + рабочая память;
	// долговременная память пишется отдельными upsert-ами сразу при изменении).
	if db != nil {
		if err := db.SaveSession(st); err != nil {
			log.Printf("memory: не удалось сохранить сессию %q: %v", st.Model, err)
		}
		return
	}
	if path == "" {
		return
	}
	if err := saveState(path, st); err != nil {
		log.Printf("history: failed to save %s: %v", path, err)
	}
}

// contextFileFor returns the per-model context path for a given model.
func (s *Server) contextFileFor(model string) string {
	return modelHistoryPath(s.historyFile, model)
}

// contextFiles returns every per-model context file on disk.
func (s *Server) contextFiles() []string {
	if s.historyFile == "" {
		return nil
	}
	dir := filepath.Dir(s.historyFile)
	name := filepath.Base(s.historyFile)
	ext := filepath.Ext(name)
	stem := strings.TrimSuffix(name, ext)
	matches, _ := filepath.Glob(filepath.Join(dir, stem+"-*"+ext))
	return matches
}

// isReserve reports whether the model belongs to the reserve provider.
func (s *Server) isReserve(model string) bool {
	return s.reserveModel != "" && model == s.reserveModel
}

// clientFor returns the API client responsible for the given model: the
// reserve client for the reserve model, the main client otherwise.
func (s *Server) clientFor(model string) *api.Client {
	if s.isReserve(model) && s.reserve != nil {
		return s.reserve
	}
	return s.client
}

// refreshModels queries the upstream /models endpoint, drops obviously non-chat
// models and updates the sidebar list. On any error — or when nothing chat-like
// is left — the currently known list is kept untouched.
func (s *Server) refreshModels(ctx context.Context) {
	ids, err := s.client.ListModels(ctx)
	if err != nil {
		return
	}
	chat := filterChatModels(ids)
	if len(chat) == 0 {
		return
	}

	s.mu.Lock()
	s.models = chat
	// Keep a valid current selection after the list is replaced. The reserve
	// provider's models are never in the main list — an explicit user choice
	// of a reserve model must survive sidebar refreshes.
	found := false
	for _, m := range s.models {
		if m == s.model {
			found = true
			break
		}
	}
	if !found && !(s.isReserve(s.model) && s.reserve != nil) {
		s.model = s.models[0]
	}
	s.mu.Unlock()
}

// filterChatModels removes model IDs that are clearly not chat completions:
// embeddings, rerankers and speech/audio models, matched by name patterns.
func filterChatModels(ids []string) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		lower := strings.ToLower(id)
		if strings.Contains(lower, "embedding") ||
			strings.Contains(lower, "rerank") ||
			strings.Contains(lower, "whisper") ||
			strings.Contains(lower, "tts") ||
			strings.Contains(lower, "audio") ||
			strings.Contains(lower, "asr") {
			continue
		}
		out = append(out, id)
	}
	return out
}

// Handler returns the HTTP routes of the web chat.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleIndex)
	mux.HandleFunc("GET /api/info", s.handleInfo)
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("GET /api/history", s.handleHistory)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/model", s.handleModel)
	mux.HandleFunc("POST /api/reset", s.handleReset)
	mux.HandleFunc("POST /api/compress", s.handleCompressSettings)
	mux.HandleFunc("POST /api/branch", s.handleBranchFork)
	mux.HandleFunc("POST /api/branch/switch", s.handleBranchSwitch)
	mux.HandleFunc("GET /api/branches", s.handleBranches)
	mux.HandleFunc("DELETE /api/branch", s.handleBranchDelete)
	mux.HandleFunc("GET /api/debug", s.handleDebug)
	mux.HandleFunc("GET /api/memory", s.handleMemoryList)
	mux.HandleFunc("POST /api/memory/layers", s.handleMemoryLayers)
	mux.HandleFunc("POST /api/memory", s.handleMemoryAdd)
	mux.HandleFunc("DELETE /api/memory", s.handleMemoryDelete)
	mux.HandleFunc("GET /api/quiz", s.handleQuiz)
	mux.HandleFunc("POST /api/dialog", s.handleDialog)
	mux.HandleFunc("POST /api/dialog/run", s.handleDialogRun)
	mux.HandleFunc("POST /api/validate", s.handleValidate)
	return mux
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(indexHTML)
}

// preset combines a preset dialog (testdata/dialogs) with an optional quiz
// (testdata/quizzes). Only the dialog turns are loaded into the context; the
// quiz is served to the UI "+" menu (GET /api/quiz) so its questions can be
// sent as regular messages on demand.
type preset struct {
	dialog string
	quiz   string
}

// presetDialogs maps preset dialog names to their embedded texts. Each
// chat-scenario blob in the UI ("Диалог про шторы"…) loads one of these as
// the whole conversation context.
var presetDialogs = map[string]preset{
	"memory-check": {dialog: testassets.DialogMemoryCheck, quiz: testassets.QuizMemory},
}

// parseDialog turns a preset-dialog text into conversation turns. A line
// starting with "User:" opens a user turn, "Model:" — an assistant turn;
// any further non-empty line continues the turn opened most recently.
func parseDialog(text string) []api.Message {
	var out []api.Message
	var role, body string
	flush := func() {
		if body == "" {
			return
		}
		if role == "user" {
			out = append(out, api.UserMessage(body))
		} else {
			out = append(out, api.AssistantMessage(body))
		}
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "User:"):
			flush()
			role, body = "user", strings.TrimSpace(strings.TrimPrefix(line, "User:"))
		case strings.HasPrefix(line, "Model:"):
			flush()
			role, body = "assistant", strings.TrimSpace(strings.TrimPrefix(line, "Model:"))
		default:
			if strings.TrimSpace(line) == "" {
				continue
			}
			if body == "" {
				role, body = "assistant", strings.TrimSpace(line)
			} else {
				body += "\n" + strings.TrimSpace(line)
			}
		}
	}
	flush()
	return out
}

// parseQuiz turns a quiz text ("Вопрос:" / "Ответ:" pairs, one per line) into
// alternating messages: a question becomes a user turn, an answer — an
// assistant turn; any further non-empty line continues the turn opened most
// recently. The "Вопрос:"/"Ответ:" prefixes are stripped from the content.
func parseQuiz(text string) []api.Message {
	var out []api.Message
	var role, body string
	flush := func() {
		if body == "" {
			return
		}
		if role == "user" {
			out = append(out, api.UserMessage(body))
		} else {
			out = append(out, api.AssistantMessage(body))
		}
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "Вопрос:"):
			flush()
			role, body = "user", strings.TrimSpace(strings.TrimPrefix(line, "Вопрос:"))
		case strings.HasPrefix(line, "Ответ:"):
			flush()
			role, body = "assistant", strings.TrimSpace(strings.TrimPrefix(line, "Ответ:"))
		default:
			if strings.TrimSpace(line) == "" {
				continue
			}
			if body == "" {
				role, body = "user", strings.TrimSpace(line)
			} else {
				body += "\n" + strings.TrimSpace(line)
			}
		}
	}
	flush()
	return out
}

// quizPair is one question/answer pair of a preset quiz, served to the UI
// "+" menu by GET /api/quiz. The answer never goes to the model: it is shown
// in the chat as an expected-answer note next to the sent question.
type quizPair struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// quizPairs extracts the question/answer pairs from every preset quiz, in
// preset-name order for a deterministic listing.
func quizPairs() []quizPair {
	names := make([]string, 0, len(presetDialogs))
	for name := range presetDialogs {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []quizPair
	for _, name := range names {
		msgs := parseQuiz(presetDialogs[name].quiz)
		for i := 0; i+1 < len(msgs); i += 2 {
			if msgs[i].Role == "user" && msgs[i+1].Role == "assistant" {
				out = append(out, quizPair{Question: msgs[i].Content, Answer: msgs[i+1].Content})
			}
		}
	}
	return out
}

// handleQuiz serves the questions of the preset quizzes for the "+" menu.
func (s *Server) handleQuiz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{"questions": quizPairs()})
}

// handleDialog loads a preset dialog from testdata as the current conversation
// context: previous turns are replaced, the system prompt is kept, the result
// is persisted like a normal dialog and returned in the /api/history shape so
// the UI can render every message in the chat log.
func (s *Server) handleDialog(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	p, ok := presetDialogs[req.Name]
	if !ok {
		http.Error(w, "Неизвестный сценарий диалога: "+req.Name, http.StatusNotFound)
		return
	}
	msgs := parseDialog(p.dialog)
	// Викторина сценария в контекст не добавляется: её вопросы отправляются
	// только по выбору из меню «+» (GET /api/quiz для списка).
	if len(msgs) == 0 {
		http.Error(w, "Сценарий диалога пуст: "+req.Name, http.StatusInternalServerError)
		return
	}

	now := time.Now()
	s.mu.Lock()
	s.hist = nil
	s.byID = make(map[string]int, len(msgs)+1)
	s.states = make(map[string]*branchState)
	s.switchable = nil
	s.activeLeaf = ""
	s.nextID = 0
	if s.cfg.System != "" {
		e := s.appendNodeLocked("", api.SystemMessage(s.cfg.System), now)
		s.activeLeaf = e.ID
	}
	out := make([]historyMessage, 0, len(msgs))
	for _, m := range msgs {
		e := s.appendNodeLocked(s.activeLeaf, m, now)
		s.activeLeaf = e.ID
		out = append(out, historyMessage{Role: m.Role, Content: m.Content, At: now, ID: e.ID})
	}
	s.resetDialogStateLocked()
	s.mu.Unlock()
	s.persist()
	log.Printf("chat: preset dialog %q loaded (%d turns)", req.Name, len(msgs))
	writeJSON(w, map[string]any{"messages": out, "active": s.activeLeaf})
}

func (s *Server) handleInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"model":         s.currentModel(),
		"default_model": s.cfg.Model,
		"baseURL":       s.cfg.BaseURL,
		"max_tokens":    s.defaultMaxTokens(),
	})
}

// defaultMaxTokens is the max_tokens value offered to the client before a user
// changes it with the slider, clamped to the 1..15000 UI range.
func (s *Server) defaultMaxTokens() int {
	v := s.cfg.MaxTokens
	if v <= 0 {
		v = 1024
	}
	if v > 15000 {
		v = 15000
	}
	return v
}

// clampRounds keeps the tool-call round budget within the 1..10 range used by
// the UI slider; a non-positive value yields the default of 6 rounds.
func clampRounds(v int) int {
	switch {
	case v <= 0:
		return 6
	case v < 1:
		return 1
	case v > 10:
		return 10
	default:
		return v
	}
}

// normalizeFormat maps the browser's format selector to a value the API client
// understands: "json" requests a JSON object, "jsonschema" and
// "jsonschema_structured" request a JSON Schema, anything else plain text.
func normalizeFormat(f string) string {
	switch strings.ToLower(strings.TrimSpace(f)) {
	case "json":
		return "json"
	case "jsonschema", "json_schema":
		return "jsonschema"
	case "jsonschema_structured", "structured":
		return "jsonschema_structured"
	default:
		return "text"
	}
}

// currentModel returns the currently selected model.
func (s *Server) currentModel() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.model
}

// handleModels lists the models available in the contact sidebar, refreshing
// them from the upstream /models endpoint so the sidebar always shows current
// chat models. When the refresh fails the last known list is used (seeded from
// AI_MODELS / DefaultModels).
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), modelRefreshTimeout)
	s.refreshModels(ctx)
	cancel()

	s.mu.Lock()
	models := append([]string(nil), s.models...)
	current := s.model
	s.mu.Unlock()
	out := map[string]any{"models": models, "current": current}
	if s.reserveModel != "" {
		out["reserve"] = s.reserveModel
		out["reserve_ready"] = s.cfg.ReserveConfigured()
	}
	writeJSON(w, out)
}

// handleModel switches the active model. The outgoing model's context is saved
// to its own file and the newly-selected model's persisted context is loaded
// (a fresh session if that model has none yet).
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	req.Model = strings.TrimSpace(req.Model)
	if req.Model == "" {
		http.Error(w, "model is empty", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	valid := false
	for _, m := range s.models {
		if m == req.Model {
			valid = true
			break
		}
	}
	if !valid && s.isReserve(req.Model) {
		if !s.cfg.ReserveConfigured() {
			s.mu.Unlock()
			http.Error(w, "Резервный провайдер не настроен: задайте AI_RESERVE_API_KEY (и, при необходимости, AI_RESERVE_BASE_URL/AI_RESERVE_MODEL) в .env", http.StatusServiceUnavailable)
			return
		}
		valid = true
	}
	if !valid {
		s.mu.Unlock()
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}
	// Persist the outgoing model's context before switching, then load the
	// newly-selected model's own persisted context (or start fresh if none).
	oldModel := s.model
	oldPath := s.contextFileFor(oldModel)
	var oldState historyState
	oldState.Model = oldModel
	if b := s.activeBranchLocked(); b != nil {
		oldState.HistoryTokens = b.HistoryTokens
		oldState.Facts = b.Facts
	}
	for _, e := range s.hist {
		if e.Msg.Role != "system" {
			oldState.Messages = append(oldState.Messages, recordOf(e))
		}
	}
	oldState.Branches = make([]branchState, 0, len(s.states))
	for _, b := range s.states {
		if b != nil {
			oldState.Branches = append(oldState.Branches, *b)
		}
	}
	oldState.Switchable = append([]string(nil), s.switchable...)
	oldState.Active = s.activeLeaf
	oldState.NextID = s.nextID

	s.model = req.Model
	s.hist = nil
	s.resetDialogStateLocked()
	newPath := s.contextFileFor(req.Model)
	s.mu.Unlock()

	if oldPath != "" {
		if err := saveState(oldPath, oldState); err != nil {
			log.Printf("history: failed to save %s: %v", oldPath, err)
		}
	}
	if newPath != "" {
		if st, err := loadState(newPath); err == nil && st != nil {
			s.mu.Lock()
			s.loadTreeLocked(st)
			s.mu.Unlock()
		}
	}
	// No persisted context for the new model (or persistence disabled):
	// start a fresh dialog.
	s.mu.Lock()
	if len(s.hist) == 0 {
		s.loadTreeLocked(&historyState{})
	}
	s.mu.Unlock()

	log.Printf("chat: switched to model %s (context loaded)", req.Model)
	writeJSON(w, map[string]bool{"ok": true})
}

type historyMessage struct {
	Role    string    `json:"role"`
	Content string    `json:"content"`
	At      time.Time `json:"at"`
	// ID links the displayed message to its tree node (empty for synthesized
	// compression plaques); Alts lists the alternative branch leaves passing
	// through this message — more than one means the UI draws the ⟨1/2⟩
	// switcher.
	ID   string   `json:"id,omitempty"`
	Alts []string `json:"alts,omitempty"`
}

func (s *Server) handleHistory(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	path, _ := s.activeMessagesLocked()
	leaves := s.branchLeaves()
	// altsAt maps a node id to the ordered leaves whose path passes through
	// it (the alternative continuations offered by the switcher); leafChild
	// maps a node to its children whose subtree contains at least one leaf.
	altsAt := make(map[string][]string)
	leafChild := make(map[string]map[string]bool)
	leafSet := make(map[string]bool, len(leaves))
	for _, leaf := range leaves {
		leafSet[leaf] = true
		lpath := s.pathToLeafLocked(leaf)
		for i, e := range lpath {
			altsAt[e.ID] = append(altsAt[e.ID], leaf)
			if i+1 < len(lpath) && e.ID != "" && lpath[i+1].ID != "" {
				if leafChild[e.ID] == nil {
					leafChild[e.ID] = make(map[string]bool)
				}
				leafChild[e.ID][lpath[i+1].ID] = true
			}
		}
	}
	// Один переключатель на каждое ветвление. Ветвление в узле p — когда от
	// p ведёт больше одного поддерева с листьями, либо p сам — активный лист
	// с единственным продолжением (пустая ветка сразу после «Ветвление
	// отсюда»). Пилюля рисуется под самим p; единственное исключение —
	// «версии ответа» (переспросить: от p ведут ≥2 ответа модели),
	// тогда переключатель стоит под активным ответом, а не под вопросом.
	// Служебные записи (системный промпт, tool-обмен, summary/marker)
	// переключателей не получают.
	// Пилюли считаем по ПОЛНОМУ (сырому) пути активного листа: точка
	// ветвления может оказаться свёрнутой сжатием и в материализованный
	// путь не попасть, но переключатель веток терять нельзя.
	fullPath := s.pathToLeafLocked(s.activeLeaf)
	pills := make(map[string][]string)
	for i, e := range fullPath {
		switch e.Msg.Role {
		case "system", "tool", "marker", "summary":
			continue
		}
		if strings.TrimSpace(e.Msg.Content) == "" || e.ID == "" {
			continue
		}
		kids := leafChild[e.ID]
		// Ветвление — точка расхождения пути: ≥2 листа-потомка, либо сам
		// узел является листом при живом продолжении ниже (пустая
		// форк-ветка «Ветвление отсюда», от которой уже ушли). Простые
		// предки точки расхождения пилюля не получают.
		if len(kids) < 2 && !(leafSet[e.ID] && len(kids) >= 1) {
			continue
		}
		// «Версии ответа» (переспросить): от вопроса ведёт ≥2 ветки с
		// ответами модели — переключатель стоит под активным ответом.
		// Пустые ответы тоже считаются версиями: иначе регенерация
		// пустого ответа возвращает пилюлю под вопрос. Пустые
		// форк-ветки («Ветвление отсюда») версии не ломают: их листья
		// остаются в списке переключателя и доступны стрелками.
		// Промежуточные tool-обмены (assistant с tool_calls) версиями
		// ответов не считаются — это служебные узлы прогона инструментов.
		assistantKids := 0
		for c := range kids {
			if ce, ok := s.nodeLocked(c); ok && ce.Msg.Role == "assistant" && len(ce.Msg.ToolCalls) == 0 {
				assistantKids++
			}
		}
		answers := assistantKids >= 2
		if answers && i+1 < len(fullPath) && kids[fullPath[i+1].ID] {
			if nx := fullPath[i+1]; nx.Msg.Role == "assistant" && len(nx.Msg.ToolCalls) == 0 {
				pills[nx.ID] = altsAt[e.ID]
			} else {
				pills[e.ID] = altsAt[e.ID]
			}
		} else {
			pills[e.ID] = altsAt[e.ID]
		}
	}
	// Свёрнутые сжатием ветвления: их пилюли переносятся на первую
	// отображаемую запись материализованного пути — обычно на плашку
	// «Сжатый контекст» (псевдо-запись сводки, ID "") или на первый
	// видимый ход, иначе переключение между ветками пропадает
	// («дерево стёрто»). Видимое ветвление важнее переноса.
	displayed := make(map[string]bool, len(path))
	firstID := ""
	for _, e := range path {
		if e.ID == "" {
			if e.Msg.Role == "summary" || e.Msg.Role == "marker" {
				displayed[""] = true
			}
			continue
		}
		switch e.Msg.Role {
		case "system", "tool", "marker", "summary":
			continue
		}
		// Пустой ответ модели — отображаемая запись (фронт рисует пометку
		// «пустой ответ модели»), пилюля версий под ним не теряется.
		// Промежуточные tool-обмены (assistant с tool_calls, пустой
		// content) служебные: в историю чата не попадают.
		if strings.TrimSpace(e.Msg.Content) == "" && !(e.Msg.Role == "assistant" && len(e.Msg.ToolCalls) == 0) {
			continue
		}
		if firstID == "" {
			firstID = e.ID
		}
		displayed[e.ID] = true
	}
	// Цель переноса: плашка сводки в начале диалога, иначе первый видимый ход.
	target := ""
	if !displayed[""] {
		target = firstID
	}
	// ВАЖНО: не мутируем pills во время range — вставленный во время
	// итерации ключ может быть посещён тем же циклом и удалён повторным
	// проходом (пилюля «исчезала» в зависимости от порядка итерации).
	// Собираем результат в новый map.
	if len(pills) > 0 {
		rebuilt := make(map[string][]string, len(pills))
		targetTaken := false
		if _, has := pills[target]; has && target != "" {
			// У цели есть собственная видимая пилюля — она важнее.
			targetTaken = true
		}
		for id, alts := range pills {
			if id != "" && displayed[id] {
				rebuilt[id] = alts
				continue
			}
			if targetTaken {
				continue
			}
			if _, exists := rebuilt[target]; !exists {
				rebuilt[target] = alts
			}
		}
		pills = rebuilt
	}
	msgs := make([]historyMessage, 0, len(path))
	for _, e := range path {
		// Только отображаемые сообщения: без системного промпта и без
		// промежуточных tool-сообщений (ассистент с tool_calls и ответы
		// инструментов). Пользовательский ввод, итоговый ответ ассистента
		// (в том числе пустой — фронт показывает пометку «пустой ответ
		// модели», и под ним стоит пилюля версий) и служебные плашки
		// (summary/marker от сжатия).
		if e.Msg.Role == "system" || e.Msg.Role == "tool" {
			continue
		}
		// Пустой assistant без tool_calls — реальный пустой ответ модели
		// (фронт показывает пометку); assistant с tool_calls — служебный
		// промежуточный узел инструмента, в чате он не отображается.
		if strings.TrimSpace(e.Msg.Content) == "" && !(e.Msg.Role == "assistant" && len(e.Msg.ToolCalls) == 0) {
			continue
		}
		hm := historyMessage{Role: e.Msg.Role, Content: e.Msg.Content, At: e.At, ID: e.ID}
		// The switcher appears under every branching point of the active path.
		if alts, ok := pills[e.ID]; ok {
			hm.Alts = alts
		}
		msgs = append(msgs, hm)
	}
	active := s.activeLeaf
	s.mu.Unlock()
	writeJSON(w, map[string]any{"messages": msgs, "active": active})
}

// handleDebug returns the JSON of the last chat API request and response plus
// the dialog-wide step log and token accounting for the "Детали" side panel.
func (s *Server) handleDebug(w http.ResponseWriter, r *http.Request) {
	request, response := s.lastChatRequestResponse()
	s.mu.Lock()
	detail := s.lastDetail
	steps := append([]tools.RunStep(nil), s.dialogSteps...)
	tokens := s.lastTokens
	prompt := s.lastPrompt
	completion := s.lastCompletion
	timeMS := s.lastTimeMS
	historyTokens, facts := 0, ""
	if b := s.activeBranchLocked(); b != nil {
		historyTokens, facts = b.HistoryTokens, b.Facts
	}
	compressSeq := s.compressSeq
	var lastCompress map[string]any
	if s.lastCompress != nil {
		lastCompress = maps.Clone(s.lastCompress)
	}
	s.mu.Unlock()

	out := map[string]any{
		"request":           request,
		"response":          response,
		"steps":             steps,
		"tokens":            tokens,
		"prompt_tokens":     prompt,
		"completion_tokens": completion,
		"time_ms":           timeMS,
		"history_tokens":    historyTokens,
		"compress_seq":      compressSeq,
		"facts":             facts,
		"last_compress":     lastCompress,
	}
	// Слои памяти: фактически подставленный долговременный блок последнего
	// запроса и размер долговременного хранилища модели.
	memBlock, memCount := s.memoryDebug()
	out["memory_block"] = memBlock
	out["memory_count"] = memCount
	if detail != "" {
		out["detail"] = json.RawMessage(detail)
	}
	writeJSON(w, out)
}

// lastChatRequestResponse returns the frozen request/response of the last chat
// exchange: follow-up compression and validation calls do not overwrite the
// "Запрос" block of the details panel.
func (s *Server) lastChatRequestResponse() (request, response string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastChatReq, s.lastChatResp
}

// appendTrace stores the last exchange's token/duration accounting and appends
// its steps to the dialog-wide step log for the "Шаги" panel.
func (s *Server) appendTrace(steps []tools.RunStep, prompt, completion, tokens int, timeMS int64) {
	s.mu.Lock()
	s.dialogSteps = append(s.dialogSteps, steps...)
	s.lastPrompt = prompt
	s.lastCompletion = completion
	s.lastTokens = tokens
	s.lastTimeMS = timeMS
	s.mu.Unlock()
}

// clearTrace resets the dialog step log (used on dialog clear and model
// switch, where a new dialog begins).
func (s *Server) clearTrace() {
	s.mu.Lock()
	s.dialogSteps = nil
	s.lastTokens = 0
	s.lastPrompt = 0
	s.lastCompletion = 0
	s.lastTimeMS = 0
	s.mu.Unlock()
}

// setDetail stores the last error/limit detail as compact JSON for the
// "Детали" panel. Must be called without holding s.mu (it acquires it).
func (s *Server) setDetail(d map[string]any) {
	if d == nil {
		return
	}
	b, err := json.Marshal(d)
	if err != nil {
		return
	}
	s.mu.Lock()
	s.lastDetail = string(b)
	s.mu.Unlock()
}

// clearDetail resets the stored detail at the start of a chat request.
func (s *Server) clearDetail() {
	s.mu.Lock()
	s.lastDetail = ""
	s.mu.Unlock()
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	// Scope — какая часть данных стирается:
	//   "chat"    (по умолчанию) — сообщения чата и краткосрочный слой,
	//             рабочая память (задача, заметки, факты) сохраняется;
	//   "history" — краткосрочная И рабочая память (сообщения + задача/
	//             заметки/факты), долговременная остаётся;
	//   "working" — только рабочая память активной ветки, сообщения остаются;
	//   "long"    — только долговременная память текущей модели;
	//   "all"     — все уровни, включая долговременную (все модели).
	scope := "chat"
	var req struct {
		Scope string `json:"scope"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req); err == nil && req.Scope != "" {
		scope = req.Scope
	}
	switch scope {
	case "chat", "history", "working", "long", "all":
	default:
		http.Error(w, "Неизвестная область очистки: "+scope, http.StatusBadRequest)
		return
	}
	wipeMessages := scope == "chat" || scope == "history" || scope == "all"

	s.mu.Lock()
	// Рабочая память (задача, заметки, факты) переживает «Удалить чат»:
	// она описывает текущую задачу, а не конкретные сообщения.
	var keepTask, keepNotes, keepFacts string
	if scope == "chat" {
		if b := s.activeBranchLocked(); b != nil {
			keepTask, keepNotes, keepFacts = b.Task, b.Notes, b.Facts
		}
	}
	if wipeMessages {
		s.hist = make([]histEntry, 0, 16)
		s.byID = make(map[string]int)
		s.states = make(map[string]*branchState)
		s.switchable = nil
		s.activeLeaf = ""
		s.nextID = 0
		s.resetDialogStateLocked()
	} else if scope == "working" {
		// «Рабочая»: сообщения остаются на месте, чистится только рабочий
		// слой активной ветки.
		if b := s.activeBranchLocked(); b != nil {
			b.Task, b.Notes, b.Facts = "", "", ""
		}
	}
	// Очистка долговременного слоя сбрасывает его in-RAM зеркало, чтобы
	// панель «Память» перечитала пустую таблицу.
	if scope == "all" || scope == "long" {
		s.memCache = make(map[string]*memCache)
		s.lastMemoryBlock = ""
	}
	if wipeMessages && s.cfg.System != "" {
		e := s.appendNodeLocked("", api.SystemMessage(s.cfg.System), time.Now())
		s.activeLeaf = e.ID
	}
	if keepTask != "" || keepNotes != "" || keepFacts != "" {
		b := s.ensureBranchStateLocked()
		b.Task, b.Notes, b.Facts = keepTask, keepNotes, keepFacts
	}
	model := s.model
	db := s.db
	path := s.contextFileFor(model)
	s.mu.Unlock()
	// «Удалить чат»/«историю» чистят сессию текущей модели; «Удалить все»
	// стирает всё (все модели); «долговременную» — только таблицу памяти.
	// «Рабочую» в БД отдельно чистить не нужно: очищенное состояние ветки
	// сохраняется ниже через persist().
	if db != nil {
		var err error
		switch scope {
		case "all":
			err = db.DeleteAllData()
		case "long":
			err = db.DeleteMemories(model)
		case "chat", "history":
			err = db.DeleteSession(model)
		}
		if err != nil {
			log.Printf("memory: не удалось очистить хранилище (%s): %v", scope, err)
		}
	} else if path != "" && wipeMessages {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			log.Printf("history: failed to remove %s: %v", path, err)
		}
	}
	if scope == "working" {
		s.persist()
	}
	s.clearTrace()
	s.clearDetail()
	writeJSON(w, map[string]bool{"ok": true})
}

// handleCompressSettings applies the compression settings sent by the panel
// (POST /api/compress, same body as the compress field of /api/chat) and
// immediately runs the compression pass, so a new threshold/keep takes
// effect without waiting for the next message. Returns whether anything
// was compressed.
func (s *Server) handleCompressSettings(w http.ResponseWriter, r *http.Request) {
	var cs compressSettings
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&cs); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	// Сжатие вызывает модель — сериализуем с генерацией ответа и прогоном.
	if !s.genMu.TryLock() {
		http.Error(w, "Идёт генерация ответа — настройки применятся к следующему сообщению", http.StatusConflict)
		return
	}
	defer s.genMu.Unlock()
	s.setCompressSettings(cs)
	info := s.autoCompress(r.Context(), s.currentModel(), true, nil)
	writeJSON(w, map[string]any{"ok": true, "compressed": info != nil, "info": info})
}

// resetDialogStateLocked clears the per-dialog accounting: the sticky-facts
// block, the cumulative request-token total, the compression bookkeeping and
// the dialog step log. Callers must hold s.mu. Steps are cleared separately
// via clearTrace (which also resets the last-exchange metrics).
func (s *Server) resetDialogStateLocked() {
	s.compressSeq++
	s.lastCompress = nil
	s.lastChatReq = ""
	s.lastChatResp = ""
}

// guardBranchOp serializes branch operations with chat generations: a fork,
// switch or delete must not move the active path while the model is
// generating or a scenario replay is running. It answers 409 when busy.
func (s *Server) guardBranchOp(w http.ResponseWriter) bool {
	if !s.genMu.TryLock() {
		http.Error(w, "Идёт генерация ответа — подождите её завершения", http.StatusConflict)
		return false
	}
	defer s.genMu.Unlock()
	s.mu.Lock()
	inRun := s.inDialogRun
	s.mu.Unlock()
	if inRun {
		http.Error(w, "Идёт прогон сценария — операции ветвления недоступны", http.StatusConflict)
		return false
	}
	return true
}

// handleBranchFork starts an alternative continuation: POST /api/branch with
// {"after": "<id>"} makes the message id the active leaf (everything after it
// stays on the old branch, which remains switchable). Switching and forking
// never call the model.
func (s *Server) handleBranchFork(w http.ResponseWriter, r *http.Request) {
	var req struct {
		After string `json:"after"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.After = strings.TrimSpace(req.After)
	if !s.guardBranchOp(w) {
		return
	}
	s.mu.Lock()
	e, ok := s.nodeLocked(req.After)
	if !ok || e.Msg.Role == "system" {
		s.mu.Unlock()
		http.Error(w, "Сообщение не найдено: "+req.After, http.StatusNotFound)
		return
	}
	if e.ID == s.activeLeaf {
		// Already the active leaf: nothing to move.
		s.mu.Unlock()
		writeJSON(w, map[string]any{"id": e.ID, "ok": true})
		return
	}
	// The abandoned leaf keeps its state and becomes switchable; the fork
	// point inherits the branch state that applies at its position.
	st := s.inheritBranchStateLocked(e.ID)
	s.registerSwitchableLocked()
	s.states[e.ID] = st
	s.activeLeaf = e.ID
	s.mu.Unlock()
	s.persist()
	log.Printf("chat: branched at %s (alternative continuation)", e.ID)
	writeJSON(w, map[string]any{"id": e.ID, "ok": true})
}

// handleBranchSwitch activates another branch: POST /api/branch/switch with
// {"id": "<leaf>"} swaps the displayed dialog without calling the model.
func (s *Server) handleBranchSwitch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.ID = strings.TrimSpace(req.ID)
	if !s.guardBranchOp(w) {
		return
	}
	s.mu.Lock()
	if _, ok := s.nodeLocked(req.ID); !ok {
		s.mu.Unlock()
		http.Error(w, "Ветка не найдена: "+req.ID, http.StatusNotFound)
		return
	}
	if _, known := s.states[req.ID]; !known && req.ID != s.activeLeaf {
		s.mu.Unlock()
		http.Error(w, "Ветка не найдена: "+req.ID, http.StatusNotFound)
		return
	}
	if req.ID == s.activeLeaf {
		s.mu.Unlock()
		writeJSON(w, map[string]any{"id": req.ID, "ok": true})
		return
	}
	// The leaf we are leaving stays switchable (the user may want to come
	// back), just like the one a fork abandons.
	s.registerSwitchableLocked()
	s.activeLeaf = req.ID
	s.mu.Unlock()
	s.persist()
	log.Printf("chat: switched to branch %s", req.ID)
	writeJSON(w, map[string]any{"id": req.ID, "ok": true})
}

// handleBranches lists the switchable branches for the UI.
func (s *Server) handleBranches(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	type branchInfo struct {
		ID     string `json:"id"`
		Title  string `json:"title,omitempty"`
		Count  int    `json:"count"`
		Active bool   `json:"active"`
	}
	out := make([]branchInfo, 0, len(s.switchable)+1)
	for _, leaf := range s.branchLeaves() {
		bi := branchInfo{ID: leaf, Active: leaf == s.activeLeaf}
		path := s.pathToLeafLocked(leaf)
		bi.Count = len(path)
		bi.Title = branchTitleLocked(path)
		if b := s.states[leaf]; b != nil && b.Title != "" {
			bi.Title = b.Title
		}
		out = append(out, bi)
	}
	s.mu.Unlock()
	writeJSON(w, map[string]any{"branches": out})
}

// handleBranchDelete forgets a branch: DELETE /api/branch with {"id"} removes
// it from the switcher (its tree nodes stay — other branches may share them;
// forking at such a node later recreates the path). The active branch cannot
// be deleted.
func (s *Server) handleBranchDelete(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.ID = strings.TrimSpace(req.ID)
	if !s.guardBranchOp(w) {
		return
	}
	s.mu.Lock()
	if req.ID == s.activeLeaf {
		s.mu.Unlock()
		http.Error(w, "Нельзя удалить активную ветку — сначала переключитесь", http.StatusConflict)
		return
	}
	if _, ok := s.states[req.ID]; !ok {
		s.mu.Unlock()
		http.Error(w, "Ветка не найдена: "+req.ID, http.StatusNotFound)
		return
	}
	delete(s.states, req.ID)
	for i, l := range s.switchable {
		if l == req.ID {
			s.switchable = append(s.switchable[:i], s.switchable[i+1:]...)
			break
		}
	}
	s.mu.Unlock()
	s.persist()
	log.Printf("chat: branch %s deleted", req.ID)
	writeJSON(w, map[string]any{"ok": true})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	// Large pastes are legitimate: a 300k-token text JSON-escapes to several
	// megabytes, so allow up to 64 MiB and report a clear error beyond that.
	const maxBody = 64 << 20
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		s.setDetail(map[string]any{
			"type":    "bad_request",
			"summary": "Не удалось прочитать запрос",
			"detail":  "Тело POST /api/chat не прочитано: " + err.Error(),
		})
		http.Error(w, "Не удалось прочитать тело запроса: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(body) > maxBody {
		s.setDetail(map[string]any{
			"type":    "request_too_large",
			"summary": "Сообщение слишком большое",
			"detail":  fmt.Sprintf("Тело запроса %d МБ превышает лимит 64 МБ; сократите вставленный текст.", len(body)>>20),
		})
		http.Error(w, "Сообщение слишком большое: тело запроса превышает лимит 64 МБ", http.StatusRequestEntityTooLarge)
		return
	}
	var req struct {
		Message    string            `json:"message"`
		Format     string            `json:"format"`
		MaxTokens  int               `json:"maxTokens"`
		Timeout    int               `json:"timeout"` // допустимый таймаут ответа, секунды (5..30)
		MaxRounds  int               `json:"maxRounds"`
		Schema     string            `json:"schema"`
		Stop       string            `json:"stop"`
		Regenerate bool              `json:"regenerate"` // «Переспросить»: ответ без нового узла вопроса
		Compress   *compressSettings `json:"compress"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		s.setDetail(map[string]any{
			"type":    "bad_request",
			"summary": "Некорректное тело запроса",
			"detail":  "POST /api/chat не разобран как JSON: " + err.Error(),
		})
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Message = strings.TrimSpace(req.Message)
	if req.Message == "" && !req.Regenerate {
		http.Error(w, "Сообщение пустое", http.StatusBadRequest)
		return
	}

	format := normalizeFormat(req.Format)
	maxTokens := s.defaultMaxTokens()
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens < 1 {
		maxTokens = 1
	}
	if maxTokens > 15000 {
		maxTokens = 15000
	}

	// Максимальное количество циклов вызова инструментов: клиент задаёт 1..10
	// (ползунок), по умолчанию 6.
	maxRounds := clampRounds(req.MaxRounds)

	// Допустимый таймаут ответа: клиент задаёт 5..100 секунд (по умолчанию 90).
	timeout := s.noReplyTimeout
	if req.Timeout > 0 {
		secs := req.Timeout
		if secs < 5 {
			secs = 5
		}
		if secs > 30 {
			secs = 30
		}
		timeout = time.Duration(secs) * time.Second
	}

	// JSON Schema formats require a non-empty, valid JSON Schema.
	req.Schema = strings.TrimSpace(req.Schema)
	if format == "jsonschema" || format == "jsonschema_structured" {
		if req.Schema == "" {
			http.Error(w, "Для формата JSON Schema нужна схема", http.StatusBadRequest)
			return
		}
		if !json.Valid([]byte(req.Schema)) {
			http.Error(w, "Некорректный JSON Schema", http.StatusBadRequest)
			return
		}
	}

	s.clearDetail()

	// Настройки автосжатия приходят с каждым запросом: выбранная стратегия
	// применяется сразу после ответа модели, без действий со стороны
	// пользователя.
	if req.Compress != nil {
		s.setCompressSettings(*req.Compress)
	}

	model := s.currentModel()

	// Генерация держит genMu уже на этапе записи вопроса: операции ветвления
	// (POST /api/branch*) не должны переставлять активную ветку посреди
	// обмена. Они отвечают 409, пока genMu занят.
	s.genMu.Lock()
	defer s.genMu.Unlock()

	s.mu.Lock()
	var userNode histEntry
	if req.Regenerate {
		// «Переспросить»: вопрос уже в дереве — новый узел не создаётся,
		// новый ответ становится альтернативной веткой того же вопроса.
		path, _ := s.activeMessagesLocked()
		switch {
		case len(path) > 0 && path[len(path)-1].Msg.Role == "user":
			userNode = path[len(path)-1]
		case len(path) > 1 && path[len(path)-1].Msg.Role == "assistant" && path[len(path)-2].Msg.Role == "user":
			// Активный лист — ответ модели: ветвимся от вопроса, прежний
			// ответ остаётся переключаемой веткой.
			question := path[len(path)-2]
			st := s.inheritBranchStateLocked(question.ID)
			s.registerSwitchableLocked()
			s.states[question.ID] = st
			s.activeLeaf = question.ID
			userNode = question
		default:
			s.mu.Unlock()
			http.Error(w, "Переспросить можно только после сообщения пользователя", http.StatusConflict)
			return
		}
	} else {
		userNode = s.appendNodeLocked(s.activeLeaf, api.UserMessage(req.Message), time.Now())
		s.activeLeaf = userNode.ID
	}
	messages := s.buildModelMessagesLocked()
	messages = append(messages, api.SystemMessage(toolMemoryGuidance))
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	// Abort the request if the model produces no output within the allowed timeout.
	// Any generation counts as alive: visible content or the reasoning trace.
	first := make(chan struct{})
	var firstOnce sync.Once
	aborted := make(chan struct{})
	activity := func() { firstOnce.Do(func() { close(first) }) }
	go func() {
		select {
		case <-first:
			// First delta arrived; the model is alive.
		case <-time.After(timeout):
			close(aborted)
			log.Printf("chat: no response from model %q within %s; aborted", model, timeout)
			cancel()
		case <-ctx.Done():
		}
	}()

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	flusher, canFlush := w.(http.Flusher)

	// The status is sent lazily: if the API fails before the first token,
	// we can still answer with a real error code.
	var answer strings.Builder
	wrote := false
	writeDelta := func(delta string) {
		firstOnce.Do(func() { close(first) })
		answer.WriteString(delta)
		if !wrote {
			w.WriteHeader(http.StatusOK)
			wrote = true
		}
		_, _ = io.WriteString(w, delta)
		if canFlush {
			flusher.Flush()
		}
	}

	// Run the tool-calling loop: the model may call get_current_time / roll_dice
	// (and have the results fed back) before delivering a plain answer. The
	// final text is streamed to the browser through writeDelta; intermediate
	// tool exchanges stay internal. Each round's token usage is accumulated so
	// the "Детали" panel can show the total.
	start := time.Now()
	var used api.Usage
	// firstRound is the usage of the very first request (whole history plus the
	// new user message); lastRound that of the final one. The history size
	// after the exchange is firstRound.PromptTokens + lastRound.CompletionTokens:
	// intermediate tool exchanges are transient machinery, not part of history.
	var firstRound, lastRound api.Usage
	var roundUsage []api.Usage
	isFirstRound := true
	// runTokens is mutable: on an empty answer the retry below doubles the
	// budget (reasoning models can spend the whole limit on reasoning and
	// return no visible content).
	runTokens := maxTokens
	round := func(ctx context.Context, conv []api.Message) (string, []api.ToolCall, error) {
		content, calls, err := s.clientFor(model).StreamChatToolCalls(ctx, model, conv, runTokens, writeDelta, activity, api.Option{
			ResponseFormat: format,
			Schema:         req.Schema,
			Stop:           req.Stop,
			Tools:          s.toolDefinitions(),
		})
		u := s.clientFor(model).LastUsage()
		used.PromptTokens += u.PromptTokens
		used.CompletionTokens += u.CompletionTokens
		used.TotalTokens += u.TotalTokens
		if isFirstRound {
			firstRound = u
			isFirstRound = false
		}
		lastRound = u
		roundUsage = append(roundUsage, u)
		return content, calls, err
	}
	reply, steps, added, err := tools.RunLoop(ctx, round, messages, maxRounds, s.memoryToolExec(model))
	// Модель может вернуть пустой ответ без ошибки (reasoning-модели
	// израсходуют весь лимит на размышления) — повторяем запрос с удвоенным
	// лимитом до двух дополнительных попыток, как в прогоне сценария.
	for attempt := 0; err == nil && strings.TrimSpace(reply) == "" && attempt < 2; attempt++ {
		runTokens *= 2
		if runTokens > 15000 {
			runTokens = 15000
		}
		log.Printf("chat: empty model answer, retrying (%d/2) with max_tokens=%d", attempt+1, runTokens)
		reply, steps, added, err = tools.RunLoop(ctx, round, messages, maxRounds, s.memoryToolExec(model))
	}
	if err == nil && strings.TrimSpace(reply) == "" {
		err = fmt.Errorf("модель вернула пустой ответ (после 3 попыток)")
	}
	// Attach each round's usage to its "Раунд" step so the "Шаги" panel shows
	// per-call prompt/completion tokens; tool-execution steps stay tokenless.
	ri := 0
	for i := range steps {
		if steps[i].Tool == "" && steps[i].Text != "" {
			if ri < len(roundUsage) {
				steps[i].PromptTokens = roundUsage[ri].PromptTokens
				steps[i].CompletionTokens = roundUsage[ri].CompletionTokens
			}
			ri++
		}
	}
	s.appendTrace(steps, used.PromptTokens, used.CompletionTokens, used.TotalTokens, time.Since(start).Milliseconds())
	// «Запрос» панели «Детали» фиксирует именно обмен с моделью: последующие
	// вызовы сжатия и проверки его не перезатирают.
	if request, response := s.clientFor(model).LastExchange(); request != "" {
		s.setLastChatExchange(request, response)
	}

	isAborted := false
	select {
	case <-aborted:
		isAborted = true
	default:
	}

	s.mu.Lock()
	if err != nil {
		// Drop the failed user turn so it can be retried (regenerate keeps
		// the pre-existing question node).
		if !req.Regenerate {
			s.removeNodeLocked(userNode.ID)
			s.activeLeaf = userNode.Parent
		}
		s.mu.Unlock()
		s.persist()

		if isAborted {
			s.setDetail(map[string]any{
				"type":          "no_response",
				"finish_reason": "timeout",
				"max_tokens":    maxTokens,
				"summary":       "Модель не ответила за " + timeout.String(),
				"detail":        fmt.Sprintf("Модель не вернула ни одного токена за %s — запрос был отменён.", timeout),
			})
		} else {
			s.setDetail(map[string]any{
				"type":    "api_error",
				"summary": "Ошибка API",
				"detail":  err.Error(),
			})
		}

		if !wrote {
			if isAborted {
				http.Error(w, fmt.Sprintf("Модель не ответила за %s. Запрос отменён.", timeout), http.StatusGatewayTimeout)
				return
			}
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		fmt.Fprintf(w, "\n\n[ошибка: %v]", err)
		return
	}
	// If a stop sequence was requested, make sure the chat gets exactly the
	// text generated before it, even if a provider echoes the stop token.
	if req.Stop != "" {
		reply = strings.TrimSuffix(reply, req.Stop)
	}
	now := time.Now()
	if strings.TrimSpace(reply) != "" {
		// Persist the intermediate tool exchanges (assistant tool_calls + tool
		// results) and then the final answer, so a restart replays the same
		// conversation to the model. Nodes chain from the question node.
		parent := userNode.ID
		for _, am := range added {
			e := s.appendNodeLocked(parent, am, now)
			parent = e.ID
		}
		e := s.appendNodeLocked(parent, api.AssistantMessage(reply), now)
		s.activeLeaf = e.ID
		// History tokens: the first request carried the whole history plus the
		// new user message, and the final answer joins it. Intermediate tool
		// exchanges are transient and do not count toward the history.
		prompt := firstRound.PromptTokens
		if prompt == 0 {
			prompt = lastRound.PromptTokens
		}
		completion := lastRound.CompletionTokens
		if completion == 0 {
			completion = firstRound.CompletionTokens
		}
		b := s.ensureBranchStateLocked()
		b.HistoryTokens = prompt + completion
	} else if !req.Regenerate {
		// The model produced no visible content (for example a stop sequence
		// hit inside the reasoning trace). Drop the just-appended user message
		// so it does not linger as an orphan and bloat later requests.
		s.removeNodeLocked(userNode.ID)
		s.activeLeaf = userNode.Parent
	}
	s.mu.Unlock()
	s.persist()

	// Автосжатие на лету: выбранная стратегия применяется сразу после ответа,
	// если отображаемых сообщений стало больше «Размера контекста».
	s.autoCompress(ctx, model, false, nil)

	// Detect schema violations and token-limit truncation so the "Детали"
	// panel can surface an error block and the chat can log a short message.
	finish := s.clientFor(model).LastFinishReason()
	var detail map[string]any
	switch {
	case finish == "length":
		detail = map[string]any{
			"type":          "token_limit",
			"finish_reason": "length",
			"max_tokens":    maxTokens,
			"summary":       "Ответ превысил лимит токенов",
			"detail":        fmt.Sprintf("Модель остановилась по лимиту в %d токенов, не завершив ответ.", maxTokens),
		}
	case (format == "jsonschema" || format == "jsonschema_structured") && reply != "" && !json.Valid([]byte(reply)):
		detail = map[string]any{
			"type":    "schema_violation",
			"summary": "Ответ не соответствует JSON-схеме",
			"detail":  "Модель вернула не JSON несмотря на требование JSON Schema.",
			"content": reply,
		}
	}
	if detail != nil {
		s.setDetail(detail)
	}

	if !wrote {
		// Empty answer: still finish the stream properly.
		w.WriteHeader(http.StatusOK)
	}
}

// compressMaxTokens caps the summary generation during context compression.
const compressMaxTokens = 2000

// compressPrompt instructs the model how to summarize the outgoing part of
// the conversation during context compression.
const compressPrompt = "Ты помогаешь сжать историю диалога. Ниже идёт переписка, которую заменяют краткой сводкой. Составь сводку: сохрани ключевые факты, имена, цифры, договорённости и открытые вопросы. Пиши сжато, без приветствий и комментариев — только текст сводки. После сводки добавь отдельные строки «ФАКТ: ключ: значение» — важные факты переписки (имена, цифры, договорённости, сроки), по одному на строку. В самом конце добавь отдельную строку «ЗАДАЧА: …» — цель текущей задачи диалога одной строкой (или «ЗАДАЧА: (нет)», если задача не обсуждалась)."

// toolMemoryGuidance goes to the model with every chat and scenario-run
// request: without it (e.g. an empty system prompt) the model never learns
// about task_note and the working layer stays empty.
const toolMemoryGuidance = "Рабочая память: инструмент task_note фиксирует заметку текущей задачи. Как только пользователь называет задачу, срок, договорённость или ограничение — сразу вызывай task_note с короткой формулировкой (одна заметка — один факт). Если заметка формулирует саму цель задачи — начни её текст со слова «Задача:», тогда она перепишет поле задачи рабочей памяти, а не попадёт в список заметок."

// clampKeep keeps the compression window (how many last messages survive the
// compression) within the 3..30 range of the UI slider; non-positive yields
// the default of 5.
func clampKeep(v int) int {
	switch {
	case v <= 0:
		return 5
	case v < 3:
		return 3
	case v > 30:
		return 30
	default:
		return v
	}
}

// compressSettings is the client-side configuration of on-the-fly compression
// sent with every chat request and scenario run.
type compressSettings struct {
	Enabled   bool   `json:"enabled"`
	Strategy  string `json:"strategy"`
	Keep      int    `json:"keep"`
	Threshold int    `json:"threshold"`
}

// setCompressSettings stores the client-provided compression configuration;
// it applies from the current request onward.
func (s *Server) setCompressSettings(cs compressSettings) {
	s.mu.Lock()
	s.compressEnabled = cs.Enabled
	s.compressStrategy = normalizeStrategy(cs.Strategy)
	s.compressKeep = clampKeep(cs.Keep)
	s.compressThreshold = clampThreshold(cs.Threshold)
	s.mu.Unlock()
}

// clampThreshold bounds the compression trigger: the number of displayed
// messages after which compression starts (5..40, default 15).
func clampThreshold(v int) int {
	switch {
	case v <= 0:
		return 15
	case v < 5:
		return 5
	case v > 40:
		return 40
	default:
		return v
	}
}

// normalizeStrategy maps the strategy name from the UI to a known value.
func normalizeStrategy(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "sliding", "sliding_window", "slidingwindow":
		return "sliding"
	case "sticky", "sticky_facts", "stickyfacts":
		return "sticky"
	default:
		return "summary"
	}
}

// buildModelMessagesLocked converts the materialized dialog path into the
// conversation sent to the model. The memory layers join in a fixed order:
// the long-term block (profile always + confirmed facts matched to the last
// user message), the working block (branch task/notes) and the sticky-facts
// block, then the path itself (system prompt, stored summaries, window).
// Callers must hold s.mu.
func (s *Server) buildModelMessagesLocked() []api.Message {
	path, _ := s.activeMessagesLocked()
	messages := make([]api.Message, 0, len(path)+4)
	// Последнее сообщение пользователя — запрос для подбора релевантных
	// долговременных фактов.
	lastUser := ""
	for _, e := range path {
		if e.Msg.Role == "user" && e.Msg.Content != "" {
			lastUser = e.Msg.Content
		}
	}
	// Выключенный тумблером слой не отправляется модели вовсе — данные
	// остаются, но запрос строится так, будто слоя нет.
	if s.layerLong {
		if block, hit := s.longTermBlockLocked(s.model, lastUser); block != "" {
			messages = append(messages, api.SystemMessage(block))
			s.lastMemoryBlock = block
			s.touchMemoriesLocked(s.model, hit)
		} else {
			s.lastMemoryBlock = ""
		}
	} else {
		s.lastMemoryBlock = ""
	}
	if s.layerWorking {
		if block := s.workingBlockLocked(); block != "" {
			messages = append(messages, api.SystemMessage(block))
		}
		facts := ""
		if b := s.activeBranchLocked(); b != nil {
			facts = b.Facts
		}
		if s.compressEnabled && s.compressStrategy == "sticky" && facts != "" {
			messages = append(messages, api.SystemMessage(
				"Блок важных фактов диалога (ключ: значение), учитывай его при ответе:\n"+facts))
		}
	}
	if s.layerShort {
		for _, e := range path {
			msg := e.Msg
			switch msg.Role {
			case "marker":
				// Служебная плашка сжатия: видна в чате, модели не отправляется.
				continue
			case "summary":
				// Сводка сжатого контекста уходит модели как системное сообщение.
				msg = api.Message{Role: "system", Content: msg.Content}
			}
			messages = append(messages, msg)
		}
	} else {
		// Краткосрочный слой выключен: модель видит только текущее сообщение
		// пользователя (последнее в пути), без истории диалога.
		if len(path) > 0 && path[len(path)-1].Msg.Role == "user" {
			messages = append(messages, path[len(path)-1].Msg)
		}
	}
	return messages
}

// setLastChatExchange freezes the request/response of a chat exchange for the
// "Запрос" block of the details panel.
func (s *Server) setLastChatExchange(request, response string) {
	s.mu.Lock()
	s.lastChatReq = request
	s.lastChatResp = response
	s.mu.Unlock()
}

// setLastCompress stores the info of the most recent compression.
func (s *Server) setLastCompress(info map[string]any) {
	s.mu.Lock()
	s.lastCompress = info
	s.mu.Unlock()
}

// attachSubStep appends a nested step to the latest dialog step (the round
// that triggered the work): compression results and answer validation are
// shown inside the step they belong to.
func (s *Server) attachSubStep(step tools.RunStep) {
	s.mu.Lock()
	if n := len(s.dialogSteps); n > 0 {
		s.dialogSteps[n-1].Sub = append(s.dialogSteps[n-1].Sub, step)
	} else {
		s.dialogSteps = append(s.dialogSteps, step)
	}
	s.mu.Unlock()
}

// firstNonEmpty returns the first non-empty string.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// autoCompress applies the selected compression strategy when the displayed
// turn count outgrows the threshold. force compresses even below the
// threshold — as soon as the kept window («Количество сообщений») is smaller
// than the current turn count — so reducing the slider visibly folds the
// chat immediately. onStart, when set, fires right before the model-backed
// strategies run: the scenario run uses it to surface the compression wait
// in the UI.
// conversation outgrows the configured threshold ("Порог контекста"); the
// kept window is the "Количество сообщений" setting. It runs on the fly after
// each completed exchange — no user action needed. Returns the compression
// info map (nil when nothing was compressed).
func (s *Server) autoCompress(ctx context.Context, model string, force bool, onStart func()) map[string]any {
	s.mu.Lock()
	enabled, strategy, keep, threshold := s.compressEnabled, s.compressStrategy, s.compressKeep, s.compressThreshold
	s.mu.Unlock()
	if !enabled {
		return nil
	}

	// turns lists the displayed conversation turns (user turns and answered
	// assistant turns) of the active branch's materialized path; previous
	// summaries and markers do not count toward the kept window, so
	// compression cannot fold into itself.
	s.mu.Lock()
	path, forkIdx := s.activeMessagesLocked()
	var turns []int
	for i, e := range path {
		switch e.Msg.Role {
		case "user":
			turns = append(turns, i)
		case "assistant":
			if strings.TrimSpace(e.Msg.Content) != "" {
				turns = append(turns, i)
			}
		}
	}
	// Compression triggers past the threshold — or immediately under force
	// (settings applied by the user) when the kept window is smaller than
	// the current turn count. The kept window itself must be exceeded, so
	// compression cannot fold into itself.
	if len(turns) <= keep || (!force && len(turns) <= threshold) {
		s.mu.Unlock()
		return nil
	}
	cut := turns[len(turns)-keep] // index of the first kept turn
	// A branch folds only its own messages: everything at or before the fork
	// point (the shared prefix, the inherited stages and the fork point
	// itself) stays visible, so the fold range starts right after it. When
	// the kept window still reaches into the shared part, there is nothing
	// of the branch's own to fold yet.
	start := forkIdx + 1
	if cut <= start {
		s.mu.Unlock()
		return nil
	}
	compressed := 0
	var old []api.Message
	covered := ""
	for i, e := range path {
		// The leading system prompt, the shared prefix with the fork point
		// and the kept tail are never folded; a previous summary of this
		// branch is re-folded together with the turns it replaced.
		if i >= cut || i < start || e.Msg.Role == "system" {
			continue
		}
		if e.ID != "" {
			covered = e.ID // the last real tree node before the kept window
			compressed++
		}
		switch e.Msg.Role {
		case "user":
			old = append(old, e.Msg)
		case "assistant":
			if strings.TrimSpace(e.Msg.Content) != "" {
				old = append(old, e.Msg)
			}
		case "summary":
			old = append(old, api.SystemMessage(e.Msg.Content))
		}
	}
	s.mu.Unlock()
	if len(old) == 0 || covered == "" {
		return nil
	}
	if onStart != nil && strategy != "sliding" {
		// Долгая фаза: сжатие вызывает модель — сообщаем UI, чтобы
		// ожидание после ответа не выглядело зависанием. Sliding Window
		// модель не вызывает и происходит мгновенно.
		onStart()
	}

	switch strategy {
	case "sliding":
		return s.compressSliding(keep, compressed, covered)
	case "sticky":
		return s.compressSticky(ctx, model, keep, compressed, covered, old)
	default:
		return s.compressSummary(ctx, model, keep, compressed, covered, old)
	}
}

// compressSliding folds every displayed turn older than the kept window away
// on the active branch (a pure pointer move — the tree stays intact) without
// calling the model.
func (s *Server) compressSliding(keep, compressed int, covered string) map[string]any {
	s.mu.Lock()
	b := s.ensureBranchStateLocked()
	b.CoveredUpTo = covered
	b.Summary = ""
	b.Plaque = fmt.Sprintf("Сжатие контекста (Sliding Window): старые сообщения удалены, в контексте последние %d", keep)
	b.HistoryTokens = 0
	s.compressSeq++
	s.mu.Unlock()
	s.persist()

	info := map[string]any{"strategy": "sliding", "compressed": compressed, "kept": keep}
	s.setLastCompress(info)
	s.attachSubStep(tools.RunStep{
		Text: fmt.Sprintf("Сжатие контекста (Sliding Window): удалено %d сообщений, осталось %d", compressed, keep),
	})
	log.Printf("chat: context compressed on the fly: sliding window, %d turns dropped, %d kept", compressed, keep)
	return info
}

// compressSummary replaces the outgoing turns with a model-generated summary
// (the original compression behavior, now automatic) recorded on the active
// branch.
func (s *Server) compressSummary(ctx context.Context, model string, keep, compressed int, covered string, old []api.Message) map[string]any {
	sumReq := make([]api.Message, 0, len(old)+2)
	sumReq = append(sumReq, api.SystemMessage(compressPrompt))
	sumReq = append(sumReq, old...)
	sumReq = append(sumReq, api.UserMessage("Составь краткую сводку переписки выше."))

	cctx, cancel := context.WithTimeout(ctx, s.cfg.Timeout)
	defer cancel()
	start := time.Now()
	// Reasoning-модели могут израсходовать весь лимит токенов на размышления
	// и вернуть пустой текст (finish_reason=length) — повторяем запрос с
	// удвоенным бюджетом до трёх попыток.
	var summary strings.Builder
	budget := compressMaxTokens
	var err error
	for attempt := 0; ; attempt++ {
		summary.Reset()
		_, err = s.clientFor(model).StreamChatWithActivity(cctx, model, sumReq, budget,
			func(delta string) { summary.WriteString(delta) }, func() {})
		if err != nil || strings.TrimSpace(summary.String()) != "" || attempt >= 2 {
			break
		}
		budget *= 2
		if budget > 15000 {
			budget = 15000
		}
		log.Printf("chat: summary compression returned an empty summary, retrying with max_tokens=%d", budget)
	}
	if err != nil {
		log.Printf("chat: summary compression failed: %v", err)
		return nil
	}
	sumText := strings.TrimSpace(summary.String())
	if sumText == "" {
		log.Printf("chat: summary compression returned an empty summary")
		return nil
	}
	// Факты («ФАКТ: …» строки) уходят в блок фактов рабочего слоя, задача
	// («ЗАДАЧА: …») — в поле задачи, в тексте сводки остаётся только она.
	prevFacts := ""
	s.mu.Lock()
	if b := s.activeBranchLocked(); b != nil {
		prevFacts = b.Facts
	}
	s.mu.Unlock()
	factLines, sumBody := extractFactLines(sumText)
	newTask, sumText := splitTaskLine(sumBody)
	if sumText == "" && factLines == "" {
		log.Printf("chat: summary compression returned an empty summary")
		return nil
	}
	facts := mergeFacts(prevFacts, factLines)
	usage := s.clientFor(model).LastUsage()

	s.mu.Lock()
	b := s.ensureBranchStateLocked()
	b.CoveredUpTo = covered
	b.Summary = sumText
	b.Facts = facts
	if newTask != "" {
		b.Task = newTask
	}
	b.Plaque = ""
	b.HistoryTokens = 0
	s.compressSeq++
	s.mu.Unlock()
	s.persist()

	// Факты сводки тоже идут в долговременную память (как у Sticky Facts):
	// дедуп по ключу, повторные встречи поднимают подтверждения.
	promoted := s.promoteFactsToLongTerm(model, facts)

	info := map[string]any{
		"strategy":          "summary",
		"compressed":        compressed,
		"kept":              keep,
		"summary":           sumText,
		"task":              newTask,
		"facts":             facts,
		"promoted":          promoted,
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
		"time_ms":           time.Since(start).Milliseconds(),
	}
	s.setLastCompress(info)
	s.attachSubStep(tools.RunStep{
		Text:             fmt.Sprintf("Сжатие контекста (Summary): сжато %d сообщений в сводку", compressed),
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		Result:           sumText,
	})
	log.Printf("chat: context compressed on the fly: %d turns -> summary, %d kept", compressed, keep)
	return info
}

// stickyFactsPrompt instructs the model how to maintain the working memory
// during compression: rewrite the task goal and the "ключ: значение" facts
// block (with an optional kind tag per line). The response is parsed by
// splitTaskFacts/parseFactLine.
const stickyFactsPrompt = "Ты ведёшь рабочую память диалога. Ниже даны текущая задача, текущий блок фактов (формат «ключ: значение») и переписка, которая удаляется из контекста. Верни ровно два раздела:\n1) строку «ЗАДАЧА: …» — цель текущей задачи и её состояние одной строкой (или «(нет)», если задача не обсуждалась);\n2) строки фактов формата «ключ: значение», по одной на строку: сохрани ещё актуальные факты без изменений, добавь новые важные сведения (имена, цифры, договорённости, сроки), убери устаревшие. Помечай каждую строку тегом в начале: «[профиль]» (о пользователе), «[решение]» (принятые решения) или «[знание]» (прочее). Без нумерации и комментариев, кроме этих двух разделов."

// compressSticky folds the outgoing turns into the working memory: the model
// rewrites the task goal and the "ключ: значение" facts, the old messages are
// dropped, and every subsequent request receives the facts block plus the
// last kept messages. The extracted facts are also promoted into the
// long-term memory (deduplicated by key, confirmations counted) — facts
// compression is the primary source of the long-term layer.
func (s *Server) compressSticky(ctx context.Context, model string, keep, compressed int, covered string, old []api.Message) map[string]any {
	s.mu.Lock()
	facts, task := "", ""
	if b := s.activeBranchLocked(); b != nil {
		facts, task = b.Facts, b.Task
	}
	s.mu.Unlock()

	var transcript strings.Builder
	for _, m := range old {
		role := "Пользователь"
		switch m.Role {
		case "assistant":
			role = "Модель"
		case "system":
			role = "Сводка"
		}
		transcript.WriteString(role + ": " + m.Content + "\n")
	}

	curTask := firstNonEmpty(task, "(пока не сформулирована)")
	req := []api.Message{
		api.SystemMessage(stickyFactsPrompt),
		api.UserMessage("Текущая задача:\n" + curTask + "\n\nТекущий блок фактов:\n" + firstNonEmpty(facts, "(пока пуст)") + "\n\nПереписка, которая удаляется из контекста:\n" + transcript.String() + "\nВерни разделы ЗАДАЧА и факты."),
	}
	cctx, cancel := context.WithTimeout(ctx, s.cfg.Timeout)
	defer cancel()
	start := time.Now()
	// Тот же бюджет-ретрай, что и в compressSummary: reasoning-модели могут
	// вернуть пустой текст, израсходовав лимит на размышления.
	var out strings.Builder
	budget := compressMaxTokens
	var err error
	for attempt := 0; ; attempt++ {
		out.Reset()
		_, err = s.clientFor(model).StreamChatWithActivity(cctx, model, req, budget,
			func(delta string) { out.WriteString(delta) }, func() {})
		if err != nil || strings.TrimSpace(out.String()) != "" || attempt >= 2 {
			break
		}
		budget *= 2
		if budget > 15000 {
			budget = 15000
		}
		log.Printf("chat: sticky facts extraction returned an empty text, retrying with max_tokens=%d", budget)
	}
	if err != nil {
		log.Printf("chat: sticky facts extraction failed: %v", err)
		return nil
	}
	newTask, newFacts := splitTaskFacts(strings.TrimSpace(out.String()))
	if newFacts == "" {
		newFacts = facts // keep the previous block when the model returned no facts
	}
	if newTask == "" {
		newTask = task // keep the previous goal when the model skipped the section
	} else if strings.EqualFold(strings.TrimSpace(newTask), "(нет)") {
		newTask = ""
	}
	usage := s.clientFor(model).LastUsage()

	s.mu.Lock()
	b := s.ensureBranchStateLocked()
	b.Facts = newFacts
	b.FactsUpTo = covered
	b.CoveredUpTo = covered
	b.Summary = ""
	b.Task = newTask
	b.Plaque = fmt.Sprintf("Сжатие контекста (Sticky Facts): старые сообщения свёрнуты в блок фактов, в контексте последние %d", keep)
	b.HistoryTokens = 0
	s.compressSeq++
	s.mu.Unlock()
	s.persist()

	// Долговременная память: факты сжатия — основной канал её наполнения
	// (дедуп по ключу, повторные встречи поднимают счётчик подтверждений).
	promoted := s.promoteFactsToLongTerm(model, newFacts)

	info := map[string]any{
		"strategy":          "sticky",
		"compressed":        compressed,
		"kept":              keep,
		"facts":             newFacts,
		"task":              newTask,
		"promoted":          promoted,
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
		"time_ms":           time.Since(start).Milliseconds(),
	}
	s.setLastCompress(info)
	s.attachSubStep(tools.RunStep{
		Text:             fmt.Sprintf("Сжатие контекста (Sticky Facts): сжато %d сообщений в блок фактов, в долговременную память перенесено фактов: %d", compressed, promoted),
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		Result:           newFacts,
	})
	log.Printf("chat: context compressed on the fly: %d turns -> facts block, %d kept, %d facts promoted to long-term", compressed, keep, promoted)
	return info
}

// validateMaxTokens caps the answer-validation request.
const validateMaxTokens = 300

// validatePrompt instructs the model to judge a model answer against the
// expected answer of a quiz question.
const validatePrompt = "Ты — строгий проверяющий ответов. Сравни ответ модели с ожидаемым ответом на вопрос. Ответ верен, если он совпадает с ожидаемым по смыслу: те же факты, цифры и суть; формулировка может отличаться. Верни JSON-объект вида {\"correct\": true|false, \"comment\": \"краткое объяснение по-русски\"} и ничего, кроме него."

// handleValidate compares the model's answer with the expected answer of a
// quiz question by asking the model. The verdict colors the expected-answer
// note in the chat (зелёная — верно, красная — неверно); the call's prompt
// tokens join the dialog-wide total.
func (s *Server) handleValidate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Answer   string `json:"answer"`
		Expected string `json:"expected"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Answer = strings.TrimSpace(req.Answer)
	req.Expected = strings.TrimSpace(req.Expected)
	if req.Answer == "" || req.Expected == "" {
		http.Error(w, "Нужны непустые поля answer и expected", http.StatusBadRequest)
		return
	}

	// Serialize with chat generations: the validation request goes to the same
	// model and must not interleave with a running exchange or compression.
	s.genMu.Lock()
	defer s.genMu.Unlock()

	model := s.currentModel()
	messages := []api.Message{
		api.SystemMessage(validatePrompt),
		api.UserMessage("Ожидаемый ответ:\n" + req.Expected + "\n\nОтвет модели:\n" + req.Answer),
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()
	start := time.Now()
	answer, err := s.clientFor(model).Chat(ctx, model, messages, validateMaxTokens, api.Option{ResponseFormat: "json"})
	if err != nil {
		log.Printf("chat: validation failed: %v", err)
		http.Error(w, "Проверка не удалась: "+err.Error(), http.StatusBadGateway)
		return
	}
	usage := s.clientFor(model).LastUsage()
	// Токены проверки НЕ входят в «Токены запроса за сценарий»: счётчик накапливает
	// только чат-обмены и запросы сжатия контекста.

	correct, comment := parseVerdict(answer)
	verdict := "неверно"
	if correct {
		verdict = "верно"
	}
	s.attachSubStep(tools.RunStep{
		Text:             "Проверка ожидаемого ответа: " + verdict,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		Result:           comment,
	})
	log.Printf("chat: answer validated: %v (%s)", correct, comment)
	writeJSON(w, map[string]any{
		"correct":           correct,
		"comment":           comment,
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
		"total_tokens":      usage.TotalTokens,
		"time_ms":           time.Since(start).Milliseconds(),
	})
}

// parseVerdict extracts {correct, comment} from the model's JSON reply,
// falling back to an embedded JSON object and then to a keyword scan.
func parseVerdict(answer string) (bool, string) {
	var v struct {
		Correct *bool  `json:"correct"`
		Comment string `json:"comment"`
	}
	text := strings.TrimSpace(answer)
	if err := json.Unmarshal([]byte(text), &v); err == nil && v.Correct != nil {
		return *v.Correct, strings.TrimSpace(v.Comment)
	}
	if i := strings.Index(text, "{"); i >= 0 {
		if j := strings.LastIndex(text, "}"); j > i {
			if err := json.Unmarshal([]byte(text[i:j+1]), &v); err == nil && v.Correct != nil {
				return *v.Correct, strings.TrimSpace(v.Comment)
			}
		}
	}
	if strings.Contains(text, `"correct": true`) || strings.Contains(text, `"correct":true`) {
		return true, text
	}
	return false, text
}

// dialogUserTurns extracts the user lines of a parsed scenario. The run
// sends them to the model one by one and keeps the model's real answers —
// the assistant lines of the file are not used.
func dialogUserTurns(msgs []api.Message) []string {
	var out []string
	for _, m := range msgs {
		if m.Role == "user" && strings.TrimSpace(m.Content) != "" {
			out = append(out, m.Content)
		}
	}
	return out
}

// handleDialogRun replays a preset dialog through the model: every user turn
// of the scenario is sent as its own request, the model's real answer is
// kept as-is (nothing is mocked or replaced), and only then the next turn
// goes out. The response is a newline-delimited JSON event stream — user,
// delta, turn, compress, error, done — so the UI can show the replay live.
func (s *Server) handleDialogRun(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name      string            `json:"name"`
		Format    string            `json:"format"`
		MaxTokens int               `json:"maxTokens"`
		Timeout   int               `json:"timeout"`
		MaxRounds int               `json:"maxRounds"`
		Schema    string            `json:"schema"`
		Stop      string            `json:"stop"`
		Compress  *compressSettings `json:"compress"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "Некорректное тело запроса (ожидается JSON): "+err.Error(), http.StatusBadRequest)
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	p, ok := presetDialogs[req.Name]
	if !ok {
		http.Error(w, "Неизвестный сценарий диалога: "+req.Name, http.StatusNotFound)
		return
	}
	turns := dialogUserTurns(parseDialog(p.dialog))
	if len(turns) == 0 {
		http.Error(w, "Сценарий диалога пуст: "+req.Name, http.StatusInternalServerError)
		return
	}
	if req.Compress != nil {
		s.setCompressSettings(*req.Compress)
	}
	format := normalizeFormat(req.Format)
	maxTokens := s.defaultMaxTokens()
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens < 1 {
		maxTokens = 1
	}
	if maxTokens > 15000 {
		maxTokens = 15000
	}
	maxRounds := clampRounds(req.MaxRounds)
	timeout := s.noReplyTimeout
	if req.Timeout > 0 {
		secs := req.Timeout
		if secs < 5 {
			secs = 5
		}
		if secs > 30 {
			secs = 30
		}
		timeout = time.Duration(secs) * time.Second
	}
	req.Schema = strings.TrimSpace(req.Schema)
	if req.Schema != "" && !json.Valid([]byte(req.Schema)) {
		http.Error(w, "Некорректный JSON Schema", http.StatusBadRequest)
		return
	}
	if (format == "jsonschema" || format == "jsonschema_structured") && req.Schema == "" {
		http.Error(w, "Для формата JSON Schema нужна схема", http.StatusBadRequest)
		return
	}

	flusher, canFlush := w.(http.Flusher)
	w.Header().Set("Content-Type", "application/x-ndjson; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	emit := func(event map[string]any) {
		b, err := json.Marshal(event)
		if err != nil {
			return
		}
		_, _ = w.Write(append(b, '\n'))
		if canFlush {
			flusher.Flush()
		}
	}

	s.clearDetail()
	// Прогон продолжает АКТИВНУЮ ВЕТКУ: реплики сценария добавляются к
	// текущему диалогу, дерево веток и выполненные сжатия сохраняются —
	// сжатие по ходу прогона происходит в этой ветке и не трогает другие
	// ветки. Начать с чистого листа можно кнопкой «Очистить диалог»
	// (/api/reset) перед прогоном. genMu is taken first so branch
	// operations (which refuse 409 while it is held) cannot interleave
	// with the run.
	now := time.Now()
	runStart := time.Now()
	s.genMu.Lock()
	defer s.genMu.Unlock()
	s.mu.Lock()
	if s.activeLeaf == "" && s.cfg.System != "" {
		e := s.appendNodeLocked("", api.SystemMessage(s.cfg.System), now)
		s.activeLeaf = e.ID
	}
	s.resetDialogStateLocked()
	// Токены считаются только в рамках прогона сценария: реплики, сжатия,
	// tool-вызовы. Обычный чат и проверка ответов счётчик не пополняют.
	s.inDialogRun = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.inDialogRun = false
		s.mu.Unlock()
	}()
	s.clearTrace()
	s.persist()

	emit(map[string]any{"type": "start", "total": len(turns)})

	model := s.currentModel()
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	for i, userText := range turns {
		emit(map[string]any{"type": "user", "index": i, "content": userText})

		s.mu.Lock()
		parentOfQuestion := s.activeLeaf
		e := s.appendNodeLocked(s.activeLeaf, api.UserMessage(userText), time.Now())
		s.activeLeaf = e.ID
		questionID := e.ID
		messages := s.buildModelMessagesLocked()
		messages = append(messages, api.SystemMessage(toolMemoryGuidance))
		s.mu.Unlock()

		// Abort the turn when the model stays silent for the allowed timeout.
		var firstOnce sync.Once
		first := make(chan struct{})
		aborted := make(chan struct{})
		activity := func() { firstOnce.Do(func() { close(first) }) }
		go func() {
			select {
			case <-first:
				// The model is alive.
			case <-time.After(timeout):
				close(aborted)
				cancel()
			case <-ctx.Done():
			}
		}()

		start := time.Now()
		var used api.Usage
		var firstRound, lastRound api.Usage
		var roundUsage []api.Usage
		isFirstRound := true
		// runTokens is mutable: on an empty answer the retry below doubles
		// the budget (reasoning models can spend the whole limit on
		// reasoning and return no visible content).
		runTokens := maxTokens
		round := func(ctx context.Context, conv []api.Message) (string, []api.ToolCall, error) {
			content, calls, err := s.clientFor(model).StreamChatToolCalls(ctx, model, conv, runTokens,
				func(delta string) {
					firstOnce.Do(func() { close(first) })
					emit(map[string]any{"type": "delta", "index": i, "text": delta})
				}, activity, api.Option{
					ResponseFormat: format,
					Schema:         req.Schema,
					Stop:           req.Stop,
					Tools:          s.toolDefinitions(),
				})
			u := s.clientFor(model).LastUsage()
			used.PromptTokens += u.PromptTokens
			used.CompletionTokens += u.CompletionTokens
			used.TotalTokens += u.TotalTokens
			if isFirstRound {
				firstRound = u
				isFirstRound = false
			}
			lastRound = u
			roundUsage = append(roundUsage, u)
			return content, calls, err
		}
		attemptRun := func() (string, error) {
			content, steps, _, err := tools.RunLoop(ctx, round, messages, maxRounds, s.memoryToolExec(model))
			ri := 0
			for k := range steps {
				if steps[k].Tool == "" && steps[k].Text != "" {
					if ri < len(roundUsage) {
						steps[k].PromptTokens = roundUsage[ri].PromptTokens
						steps[k].CompletionTokens = roundUsage[ri].CompletionTokens
					}
					ri++
				}
			}
			s.appendTrace(steps, used.PromptTokens, used.CompletionTokens, used.TotalTokens, time.Since(start).Milliseconds())
			if request, response := s.clientFor(model).LastExchange(); request != "" {
				s.setLastChatExchange(request, response)
			}
			return content, err
		}
		content, err := attemptRun()
		// Модель может вернуть пустой ответ без ошибки — реплика остаётся
		// без ответа, а прогон молча уходит дальше. Повторяем запрос (до
		// двух дополнительных попыток); при повторных пустых ответах
		// останавливаем прогон с ошибкой, а не продолжаем молча.
		for attempt := 0; err == nil && strings.TrimSpace(content) == "" && attempt < 2; attempt++ {
			// Reasoning-модели (например deepseek-flash) могут израсходовать
			// весь лимит токенов на размышления: content приходит пустым
			// (finish_reason=length). Повторяем запрос с удвоенным лимитом,
			// чтобы ответу хватило бюджета.
			runTokens *= 2
			if runTokens > 15000 {
				runTokens = 15000
			}
			log.Printf("chat: dialog run turn %d: empty model answer, retrying (%d/2) with max_tokens=%d", i+1, attempt+1, runTokens)
			content, err = attemptRun()
		}
		if err == nil && strings.TrimSpace(content) == "" {
			err = fmt.Errorf("модель вернула пустой ответ на реплику %d (после 3 попыток)", i+1)
		}

		isAborted := false
		select {
		case <-aborted:
			isAborted = true
		default:
		}
		if err != nil {
			// Drop the failed user turn and stop the replay.
			s.mu.Lock()
			s.removeNodeLocked(questionID)
			s.activeLeaf = parentOfQuestion
			s.mu.Unlock()
			s.persist()
			// Детали ошибки — в панель «Детали» (блок «Ошибка»), как при
			// обычном чат-обмене. Остановку пользователем (закрытие соединения)
			// ошибкой не считаем.
			if r.Context().Err() != nil {
				// Прогон остановлен кнопкой «Остановить».
			} else if isAborted {
				s.setDetail(map[string]any{
					"type":          "no_response",
					"finish_reason": "timeout",
					"max_tokens":    maxTokens,
					"summary":       "Модель не ответила за " + timeout.String(),
					"detail":        fmt.Sprintf("Модель не вернула ни одного токена за %s — прогон остановлен.", timeout),
				})
			} else {
				s.setDetail(map[string]any{
					"type":    "api_error",
					"summary": "Ошибка API при прогоне сценария",
					"detail":  err.Error(),
				})
			}
			if isAborted {
				emit(map[string]any{"type": "error", "error": fmt.Sprintf("Модель не ответила за %s — прогон остановлен.", timeout)})
			} else {
				emit(map[string]any{"type": "error", "error": err.Error()})
			}
			return
		}

		// Ответ модели — реальный, сохраняется как есть: никакой замены
		// каноническим текстом из файла.
		s.mu.Lock()
		ans := s.appendNodeLocked(questionID, api.AssistantMessage(content), time.Now())
		s.activeLeaf = ans.ID
		prompt := firstRound.PromptTokens
		if prompt == 0 {
			prompt = lastRound.PromptTokens
		}
		completion := lastRound.CompletionTokens
		if completion == 0 {
			completion = firstRound.CompletionTokens
		}
		b := s.ensureBranchStateLocked()
		b.HistoryTokens = prompt + completion
		s.mu.Unlock()
		s.persist()

		emit(map[string]any{"type": "turn", "index": i, "content": content})

		if info := s.autoCompress(ctx, model, false, func() {
			emit(map[string]any{"type": "compressing"})
		}); info != nil {
			emit(map[string]any{"type": "compress", "info": info})
		}
	}

	// Итог прогона: число реплик и время — события забирает UI прогона.
	emit(map[string]any{
		"type":    "done",
		"turns":   len(turns),
		"time_ms": time.Since(runStart).Milliseconds(),
	})
	log.Printf("chat: preset dialog %q replayed through the model (%d turns)", req.Name, len(turns))
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
