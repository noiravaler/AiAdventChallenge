package mcpclient

import (
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// imagesDir — каталог для изображений из результатов инструментов;
// переменная для тестов.
var imagesDir = defaultImagesDir

func defaultImagesDir() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "ai-chat-go", "charts")
}

// collectImages сохраняет изображения из результата MCP-инструмента и
// возвращает их веб-пути ("/files/<имя>") для показа в чате. В модель эти
// пути не попадают: модель получает только текстовую часть результата
// (см. contentText).
func collectImages(res *mcp.CallToolResult) []string {
	var urls []string
	n := 0
	for _, item := range res.Content {
		ic, ok := item.(*mcp.ImageContent)
		if !ok || len(ic.Data) == 0 {
			continue
		}
		n++
		path, err := saveImage(ic, n)
		if err != nil {
			continue
		}
		urls = append(urls, "/files/"+filepath.Base(path))
	}
	return urls
}

// saveImage пишет изображение в кэш-каталог с именем по времени.
func saveImage(ic *mcp.ImageContent, n int) (string, error) {
	dir := imagesDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	name := "chart-" + time.Now().Format("20060102-150405.000") + "-" + strconv.Itoa(n) + imageExt(ic.MIMEType)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, ic.Data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// imageExt подбирает расширение по MIME-типу.
func imageExt(mime string) string {
	switch mime {
	case "image/svg+xml":
		return ".svg"
	case "image/png":
		return ".png"
	case "image/jpeg", "image/jpg":
		return ".jpg"
	case "image/gif":
		return ".gif"
	case "image/webp":
		return ".webp"
	default:
		return ".img"
	}
}

// ImagesDir возвращает каталог, куда складываются изображения из результатов
// MCP-инструментов; он раздаётся сервером чата по маршруту /files/.
func ImagesDir() string { return imagesDir() }
