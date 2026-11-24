package orbit

import (
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestImageResult — chart_render в MCP-транспорте: текст для модели без
// путей и image-контент для клиента.
func TestImageResult(t *testing.T) {
	rr := &RenderResult{
		Model:    map[string]any{"type": "natal", "chart": "Anna", "bytes": 5},
		SVG:      []byte("<svg/>"),
		FileName: "anna-natal.svg",
	}
	res := imageResult(rr)
	if len(res.Content) != 2 {
		t.Fatalf("ожидались текст + изображение, получено %d контентов", len(res.Content))
	}
	tc, ok := res.Content[0].(*mcp.TextContent)
	if !ok {
		t.Fatalf("первый контент не текст: %T", res.Content[0])
	}
	for _, banned := range []string{"anna-natal.svg", "<svg", "/charts/"} {
		if strings.Contains(tc.Text, banned) {
			t.Fatalf("в тексте для модели найдено %q: %s", banned, tc.Text)
		}
	}
	ic, ok := res.Content[1].(*mcp.ImageContent)
	if !ok {
		t.Fatalf("второй контент не изображение: %T", res.Content[1])
	}
	if ic.MIMEType != "image/svg+xml" || string(ic.Data) != "<svg/>" {
		t.Fatalf("неожиданное изображение: mime=%s data=%q", ic.MIMEType, ic.Data)
	}
}
