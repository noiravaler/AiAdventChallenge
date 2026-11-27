package affirmations

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// inEmojiSet — проверка, что смайлик из волшебного набора.
func inEmojiSet(s string) bool {
	for _, e := range magicEmoji {
		if e == s {
			return true
		}
	}
	return false
}

// TestRussian — полный путь: аффирмация + перевод через фейковые серверы.
func TestRussian(t *testing.T) {
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "application/json" {
			t.Errorf("ожидается Accept: application/json, получено %q", r.Header.Get("Accept"))
		}
		_, _ = w.Write([]byte(`{"affirmation":"I know you'll sort it out"}`))
	}))
	defer src.Close()

	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm: %v", err)
		}
		if got := r.Form.Get("q"); got != "I know you'll sort it out" {
			t.Errorf("переводчику ушёл неверный текст: %q", got)
		}
		if r.Form.Get("sl") != "en" || r.Form.Get("tl") != "ru" {
			t.Errorf("неверная пара языков: %q→%q", r.Form.Get("sl"), r.Form.Get("tl"))
		}
		_, _ = w.Write([]byte(`[[["Я знаю, что ты всё получится","I know you'll sort it out",null,null,10]],null,"en"]`))
	}))
	defer tr.Close()

	c := New(src.URL, tr.URL+"/translate_a/single", 5*time.Second)
	res, err := c.Russian(context.Background())
	if err != nil {
		t.Fatalf("Russian: %v", err)
	}
	if res.Affirmation != "Я знаю, что ты всё получится" {
		t.Errorf("ожидался перевод, получено: %q", res.Affirmation)
	}
	if !inEmojiSet(res.Emoji) {
		t.Errorf("смайлик вне набора: %q", res.Emoji)
	}
}

// TestRussianTranslateDown — переводчик упал: инструмент не падает,
// возвращается оригинал.
func TestRussianTranslateDown(t *testing.T) {
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"affirmation":"You are enough"}`))
	}))
	defer src.Close()

	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer tr.Close()

	c := New(src.URL, tr.URL, 5*time.Second)
	res, err := c.Russian(context.Background())
	if err != nil {
		t.Fatalf("Russian: %v", err)
	}
	if res.Affirmation != "You are enough" {
		t.Errorf("без перевода ожидается оригинал, получено: %q", res.Affirmation)
	}
	if !inEmojiSet(res.Emoji) {
		t.Errorf("смайлик вне набора: %q", res.Emoji)
	}
}

// TestRussianNoTranslator — переводчик не настроен: только оригинал.
func TestRussianNoTranslator(t *testing.T) {
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"affirmation":"Keep going"}`))
	}))
	defer src.Close()

	c := New(src.URL, "", 5*time.Second)
	res, err := c.Russian(context.Background())
	if err != nil {
		t.Fatalf("Russian: %v", err)
	}
	if res.Affirmation != "Keep going" {
		t.Errorf("неожиданный результат: %+v", res)
	}
	if !inEmojiSet(res.Emoji) {
		t.Errorf("смайлик вне набора: %q", res.Emoji)
	}
}

// TestFetchErrors — некорректные ответы источника дают ошибку.
func TestFetchErrors(t *testing.T) {
	for name, body := range map[string]string{
		"не-JSON":      `<html>oops</html>`,
		"пустая фраза": `{"affirmation":"   "}`,
	} {
		src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		c := New(src.URL, "", 5*time.Second)
		if _, err := c.Russian(context.Background()); err == nil {
			t.Errorf("%s: ожидалась ошибка", name)
		}
		src.Close()
	}
}

// TestTranslate — произвольный текст и произвольное направление перевода.
func TestTranslate(t *testing.T) {
	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm: %v", err)
		}
		// Пустые языки подставляются на клиенте: en→ru по умолчанию.
		if r.Form.Get("sl") != "en" || r.Form.Get("tl") != "ru" {
			t.Errorf("неверная пара языков: %q→%q", r.Form.Get("sl"), r.Form.Get("tl"))
		}
		if r.Form.Get("q") != "Everything will work out" {
			t.Errorf("неверный текст: %q", r.Form.Get("q"))
		}
		_, _ = w.Write([]byte(`[[["Всё получится","Everything will work out",null,null,10]],null,"en"]`))
	}))
	defer tr.Close()

	c := New("https://www.affirmations.dev", tr.URL, 5*time.Second)
	out, err := c.Translate(context.Background(), "Everything will work out", "", "")
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if out != "Всё получится" {
		t.Errorf("неожиданный перевод: %q", out)
	}
}

// TestTranslateCustomDirection — нестандартная пара языков доходит до gtx.
func TestTranslateCustomDirection(t *testing.T) {
	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		if r.Form.Get("sl") != "ru" || r.Form.Get("tl") != "de" {
			t.Errorf("неверная пара языков: %q→%q", r.Form.Get("sl"), r.Form.Get("tl"))
		}
		_, _ = w.Write([]byte(`[[["Alles gut","всё хорошо",null,null,10]],null,"ru"]`))
	}))
	defer tr.Close()

	c := New("https://www.affirmations.dev", tr.URL, 5*time.Second)
	out, err := c.Translate(context.Background(), "всё хорошо", "ru", "de")
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if out != "Alles gut" {
		t.Errorf("неожиданный перевод: %q", out)
	}
}

// TestStatus — диагностика обоих провайдеров: доступные и упавшие.
func TestStatus(t *testing.T) {
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"affirmation":"Stay strong"}`))
	}))
	defer src.Close()
	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[[["привет","hello",null,null,10]],null,"en"]`))
	}))
	defer tr.Close()

	c := New(src.URL, tr.URL, 5*time.Second)
	st := c.Status(context.Background())
	aff := st["affirmations"].(map[string]any)
	if aff["status"] != "ok" {
		t.Errorf("аффирмации: ожидался ok, получено %v", aff["status"])
	}
	tst := st["translate"].(map[string]any)
	if tst["status"] != "ok" {
		t.Errorf("переводчик: ожидался ok, получено %v", tst["status"])
	}
}

// TestStatusTranslateDown — упавший переводчик виден в статусе,
// а аффирмации остаются ok.
func TestStatusTranslateDown(t *testing.T) {
	src := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"affirmation":"Stay strong"}`))
	}))
	defer src.Close()
	tr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer tr.Close()

	c := New(src.URL, tr.URL, 5*time.Second)
	st := c.Status(context.Background())
	aff := st["affirmations"].(map[string]any)
	if aff["status"] != "ok" {
		t.Errorf("аффирмации: ожидался ok, получено %v", aff["status"])
	}
	tst := st["translate"].(map[string]any)
	if tst["status"] != "error" {
		t.Errorf("переводчик: ожидался error, получено %v", tst["status"])
	}
}

// TestParseGTX — разбор сегментов ответа gtx, включая склейку.
func TestParseGTX(t *testing.T) {
	out, err := parseGTX([]byte(`[[["У тебя ","всё получится","",null,10],["всё отлично","",null,10]],null,"en"]`))
	if err != nil {
		t.Fatalf("parseGTX: %v", err)
	}
	if out != "У тебя всё отлично" {
		t.Errorf("склейка сегментов: %q", out)
	}
	if _, err := parseGTX([]byte(`{"error":true}`)); err == nil {
		t.Error("не-JSON принят без ошибки")
	}
}
