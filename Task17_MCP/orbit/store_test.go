package orbit

import (
	"path/filepath"
	"testing"
)

// fixtureChart — карта для тестов хранилища.
func fixtureChart(name string) *Chart {
	ch := &Chart{}
	ch.Name = name
	ch.Subject.Name = name
	ch.Subject.Year, ch.Subject.Month, ch.Subject.Day = 1995, 7, 21
	ch.Subject.Hour, ch.Subject.Minute = 14, 30
	ch.Subject.City = "Kazan"
	ch.Subject.Timezone = "Europe/Moscow"
	ch.Location = testLocation()
	ch.Location.Name = "Kazan"
	ch.Location.CountryCode = "RU"
	return ch
}

// TestStoreSaveGetPersist — сохранение, выбор «последней» и перезагрузка с диска.
func TestStoreSaveGetPersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "charts.json")
	st, err := NewStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Save(fixtureChart("Anna")); err != nil {
		t.Fatal(err)
	}
	if err := st.Save(fixtureChart("Bob")); err != nil {
		t.Fatal(err)
	}

	last, err := st.Get("")
	if err != nil || last.Name != "Bob" {
		t.Fatalf("последняя карта должна быть Bob, получено %v (%v)", last, err)
	}
	anna, err := st.Get("ANNA ") // регистр и пробелы нормализуются
	if err != nil || anna.Name != "Anna" {
		t.Fatalf("поиск без учёта регистра сломан: %v (%v)", anna, err)
	}
	if _, err := st.Get("Unknown"); err == nil {
		t.Fatal("неизвестное имя должно давать ошибку")
	}

	// Перезагрузка с того же файла: данные и указатель «последней» живы.
	st2, err := NewStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if last2, err := st2.Get(""); err != nil || last2.Name != "Bob" {
		t.Fatalf("персистентность сломана: %v (%v)", last2, err)
	}
	if n := len(st2.List()); n != 2 {
		t.Fatalf("должно быть 2 карты, получено %d", n)
	}
}

// TestStoreEmpty — поведение пустого хранилища.
func TestStoreEmpty(t *testing.T) {
	st, err := NewStore(filepath.Join(t.TempDir(), "charts.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Get(""); err == nil {
		t.Fatal("пустое хранилище должно отказывать в Get без имени")
	}
	if st.Last() != nil {
		t.Fatal("Last пустого хранилища должен быть nil")
	}
}
