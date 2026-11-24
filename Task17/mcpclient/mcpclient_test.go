package mcpclient

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestContentTextIgnoresImages — модель получает только текстовую часть
// результата: base64 изображений и пути в контекст не утекают.
func TestContentTextIgnoresImages(t *testing.T) {
	res := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: `{"type":"natal","chart":"Anna","bytes":110131}`},
			&mcp.ImageContent{Data: []byte("<svg>ВЕСЬ SVG И ВЕСЬ ПУТЬ</svg>"), MIMEType: "image/svg+xml"},
		},
	}
	text := contentText(res)
	if strings.Contains(text, "<svg>") || strings.Contains(text, "image/svg") {
		t.Fatalf("содержимое изображения утекло в текст для модели: %s", text)
	}
	if !strings.Contains(text, `"chart":"Anna"`) {
		t.Fatalf("текстовая часть потерялась: %s", text)
	}
}

// TestSaveImage — изображение пишется в настраиваемый каталог с расширением
// по MIME-типу.
func TestSaveImage(t *testing.T) {
	dir := t.TempDir()
	old := imagesDir
	imagesDir = func() string { return dir }
	defer func() { imagesDir = old }()

	ic := &mcp.ImageContent{Data: []byte("<svg>ok</svg>"), MIMEType: "image/svg+xml"}
	path, err := saveImage(ic, 1)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Ext(path) != ".svg" {
		t.Fatalf("неожиданное расширение: %s", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "<svg>ok</svg>" {
		t.Fatalf("содержимое не сохранилось: %s", raw)
	}
}

// TestCollectImages — изображения из результата превращаются в веб-пути
// /files/<имя>, а файлы реально пишутся в каталог кэша.
func TestCollectImages(t *testing.T) {
	dir := t.TempDir()
	old := imagesDir
	imagesDir = func() string { return dir }
	defer func() { imagesDir = old }()

	res := &mcp.CallToolResult{Content: []mcp.Content{
		&mcp.TextContent{Text: "сводка для модели"},
		&mcp.ImageContent{Data: []byte("<svg>карта</svg>"), MIMEType: "image/svg+xml"},
		&mcp.ImageContent{Data: nil, MIMEType: "image/png"}, // пустое — пропускается
	}}
	urls := collectImages(res)
	if len(urls) != 1 {
		t.Fatalf("urls = %v, want 1", urls)
	}
	if len(urls[0]) < 8 || urls[0][:7] != "/files/" {
		t.Fatalf("путь не начинается с /files/: %q", urls[0])
	}
	raw, err := os.ReadFile(filepath.Join(dir, filepath.Base(urls[0])))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "<svg>карта</svg>" {
		t.Fatalf("содержимое файла не совпало: %s", raw)
	}
}

// TestImageExt — таблица расширений.
func TestImageExt(t *testing.T) {
	cases := map[string]string{
		"image/svg+xml": ".svg",
		"image/png":     ".png",
		"image/jpeg":    ".jpg",
		"image/gif":     ".gif",
		"image/webp":    ".webp",
		"unknown":       ".img",
	}
	for mime, want := range cases {
		if got := imageExt(mime); got != want {
			t.Fatalf("imageExt(%q) = %q, ожидалось %q", mime, got, want)
		}
	}
}
