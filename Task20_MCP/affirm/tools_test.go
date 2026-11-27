package affirm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-chat-go/affirmations"
)

// newFakes поднимает фейковые affirmations.dev и переводчик и возвращает
// тулкит на них (офлайн-тесты, без внешних сервисов).
func newFakes(t *testing.T, affirmBody string, translateStatus int) *Toolkit {
	t.Helper()
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(affirmBody))
	}))
	t.Cleanup(src.Close)

	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		if translateStatus != http.StatusOK {
			w.WriteHeader(translateStatus)
			return
		}
		_, _ = w.Write([]byte(`[[["Ты способен на удивительные вещи","You are capable of amazing things",null,null,10]],null,"en"]`))
	}))
	t.Cleanup(tr.Close)

	return NewToolkit(affirmations.New(src.URL, tr.URL+"/translate_a/single", time.Second))
}

// TestDefs — реестр инструментов: имена и порядок стабильны.
func TestDefs(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"You are capable of amazing things"}`, http.StatusOK)
	want := []string{"daily_affirmation", "translate_text", "affirmation_status"}
	defs := tk.Defs()
	if len(defs) != len(want) {
		t.Fatalf("ожидалось %d инструментов, получено %d", len(want), len(defs))
	}
	for i, name := range want {
		if defs[i].Name != name {
			t.Errorf("инструмент %d: ожидалось %q, получено %q", i, name, defs[i].Name)
		}
		if defs[i].Description == "" {
			t.Errorf("инструмент %q: пустое описание", name)
		}
	}
}

// TestInvokeDailyAffirmation — путь режима -probe: аффирмация переводится.
func TestInvokeDailyAffirmation(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"You are capable of amazing things"}`, http.StatusOK)
	raw, err := tk.Invoke(context.Background(), "daily_affirmation", nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	res, ok := raw.(affirmations.Result)
	if !ok {
		t.Fatalf("неожиданный тип результата: %T", raw)
	}
	if res.Affirmation != "Ты способен на удивительные вещи" {
		t.Errorf("ожидался перевод, получено: %q", res.Affirmation)
	}
	if !inEmojiSet(res.Emoji) {
		t.Errorf("смайлик вне набора: %q", res.Emoji)
	}
}

// TestInvokeTranslateText — произвольный перевод с направлением по умолчанию.
func TestInvokeTranslateText(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"Keep going"}`, http.StatusOK)
	raw, err := tk.Invoke(context.Background(), "translate_text", json.RawMessage(`{"text":"You are capable of amazing things"}`))
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	res, ok := raw.(map[string]any)
	if !ok {
		t.Fatalf("неожиданный тип результата: %T", raw)
	}
	if res["translated"] != "Ты способен на удивительные вещи" {
		t.Errorf("неожиданный перевод: %v", res["translated"])
	}
	if res["source"] != "en" || res["target"] != "ru" {
		t.Errorf("ожидалось en→ru, получено %v→%v", res["source"], res["target"])
	}
}

// TestTranslateTextEmpty — пустой текст отклоняется с ошибкой.
func TestTranslateTextEmpty(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"Keep going"}`, http.StatusOK)
	if _, err := tk.TranslateText(context.Background(), TranslateArgs{Text: "   "}); err == nil {
		t.Error("ожидалась ошибка на пустой текст")
	}
}

// TestInvokeUnknown — неизвестное имя инструмента даёт понятную ошибку.
func TestInvokeUnknown(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"Keep going"}`, http.StatusOK)
	if _, err := tk.Invoke(context.Background(), "vibe_today", nil); err == nil {
		t.Error("ожидалась ошибка на неизвестный инструмент")
	}
}

// TestStatus — диагностика провайдеров через Invoke.
func TestStatus(t *testing.T) {
	tk := newFakes(t, `{"affirmation":"Keep going"}`, http.StatusOK)
	raw, err := tk.Invoke(context.Background(), "affirmation_status", nil)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	res := raw.(map[string]any)
	if res["server"] != "affirm-mcp" {
		t.Errorf("неожиданное имя сервера: %v", res["server"])
	}
	prov := res["providers"].(map[string]any)
	aff := prov["affirmations"].(map[string]any)
	if aff["status"] != "ok" {
		t.Errorf("аффирмации: ожидался ok, получено %v", aff["status"])
	}
	tst := prov["translate"].(map[string]any)
	if tst["status"] != "ok" {
		t.Errorf("переводчик: ожидался ok, получено %v", tst["status"])
	}
}

// emojiSet — копия волшебного набора из affirmations (пакет его не экспортирует).
var emojiSet = []string{"✨", "🌟", "⭐", "💫", "🔮", "🌙", "🌠", "🪄", "🌌", "🎆"}

// inEmojiSet — проверка, что смайлик из волшебного набора.
func inEmojiSet(s string) bool {
	for _, e := range emojiSet {
		if e == s {
			return true
		}
	}
	return false
}
