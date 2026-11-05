package main

import (
	"fmt"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/app/web"
	"ai-chat-go/config"
)

// runWeb starts the local web chat and opens it in the browser.
func runWeb(client *api.Client, cfg config.Config, port int, open bool) error {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		// Port is taken: fall back to a random free port.
		ln, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return err
		}
	}
	url := fmt.Sprintf("http://%s", ln.Addr())

	server := &http.Server{
		Handler:           web.New(client, cfg).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	fmt.Printf("AI chat | model: %s | endpoint: %s\n", cfg.Model, cfg.BaseURL)
	fmt.Printf("Web chat: %s  (Ctrl+C to stop)\n", url)
	if open {
		if err := openInBrowser(url); err != nil {
			fmt.Println("Не удалось открыть браузер автоматически:", err)
		}
	}
	return server.Serve(ln)
}

func openInBrowser(url string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", url).Start()
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	default:
		return exec.Command("xdg-open", url).Start()
	}
}
