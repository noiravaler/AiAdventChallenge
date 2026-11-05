package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"ai-chat-go/api"
	"ai-chat-go/config"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	fs := flag.NewFlagSet("ai-chat", flag.ExitOnError)
	baseURL := fs.String("base-url", cfg.BaseURL, "API base URL (OpenAI-compatible), e.g. https://api.openai.com/v1")
	apiKey := fs.String("api-key", "", "API key; prefer the AI_API_KEY environment variable")
	model := fs.String("model", cfg.Model, "model name")
	system := fs.String("system", cfg.System, "system prompt")
	maxTokens := fs.Int("max-tokens", cfg.MaxTokens, "maximum reply tokens (0 = provider default)")
	timeout := fs.Duration("timeout", cfg.Timeout, "timeout for one API request")
	noStream := fs.Bool("no-stream", false, "wait for the whole answer instead of streaming it")
	useWeb := fs.Bool("web", true, "run the web interface and open it in the browser")
	cliMode := fs.Bool("cli", false, "use the interactive terminal chat instead of the web page")
	webPort := fs.Int("port", 5190, "port for the web interface")
	noOpen := fs.Bool("no-open", false, "do not open the browser automatically")

	fs.Usage = func() {
		fmt.Fprint(fs.Output(), `Usage:
  ai-chat [flags]            open the chat as a local web page in the browser
  ai-chat [flags] -cli       chat in the terminal
  ai-chat [flags] <prompt>   send one prompt and exit

Flags:
`)
		fs.PrintDefaults()
		fmt.Fprintf(fs.Output(), `
Environment:
  AI_API_KEY     API key (required unless the server is local)
  AI_BASE_URL    API base URL (default %s)
  AI_MODEL       model name (default %s)
  AI_SYSTEM      system prompt
  AI_TIMEOUT     request timeout, e.g. 90s (default %s)
  AI_MAX_TOKENS  maximum reply tokens
`, config.DefaultBaseURL, config.DefaultModel, config.DefaultTimeout)
	}

	if err := fs.Parse(os.Args[1:]); err != nil {
		return err
	}

	if key := strings.TrimSpace(*apiKey); key != "" {
		cfg.APIKey = key
	}
	cfg.BaseURL = strings.TrimRight(strings.TrimSpace(*baseURL), "/")
	cfg.Model = strings.TrimSpace(*model)
	cfg.System = strings.TrimSpace(*system)
	cfg.MaxTokens = *maxTokens
	cfg.Timeout = *timeout
	if err := cfg.Validate(); err != nil {
		return err
	}

	client := api.New(cfg.BaseURL, cfg.APIKey, cfg.Timeout)
	// Log every API request/response to the app console as JSON lines.
	client.DebugLog = os.Stdout

	history := make([]api.Message, 0, 16)
	resetHistory := func() {
		history = history[:0]
		if cfg.System != "" {
			history = append(history, api.SystemMessage(cfg.System))
		}
	}
	resetHistory()

	request := func(ctx context.Context, messages []api.Message, onDelta func(string)) (string, error) {
		if *noStream {
			answer, err := client.Chat(ctx, cfg.Model, messages, cfg.MaxTokens)
			if err != nil {
				return "", err
			}
			if onDelta != nil {
				onDelta(answer)
			}
			return answer, nil
		}
		return client.StreamChat(ctx, cfg.Model, messages, cfg.MaxTokens, onDelta)
	}

	if fs.NArg() > 0 {
		prompt := strings.Join(fs.Args(), " ")
		history = append(history, api.UserMessage(prompt))

		ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
		defer cancel()

		if _, err := request(ctx, history, func(s string) { fmt.Print(s) }); err != nil {
			return err
		}
		fmt.Println()
		return nil
	}

	if *useWeb && !*cliMode {
		return runWeb(client, cfg, *webPort, !*noOpen)
	}

	fmt.Printf("AI chat | model: %s | endpoint: %s\n", cfg.Model, cfg.BaseURL)
	if cfg.System != "" {
		fmt.Printf("system: %s\n", cfg.System)
	}
	fmt.Println("Type /help for commands, /exit to quit.")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("you> ")
		line, readErr := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return readErr
		}

		switch {
		case line == "":
			// Nothing to send; EOF is handled at the end of the loop.

		case line == "/exit" || line == "/quit":
			fmt.Println("Bye.")
			return nil

		case line == "/help":
			printHelp()

		case line == "/reset":
			resetHistory()
			fmt.Println("History cleared.")

		case line == "/model":
			fmt.Printf("Current model: %s\n", cfg.Model)

		case strings.HasPrefix(line, "/model "):
			name := strings.TrimSpace(strings.TrimPrefix(line, "/model "))
			if name == "" {
				fmt.Println("Usage: /model <name>")
			} else {
				cfg.Model = name
				fmt.Printf("Model switched to %s\n", cfg.Model)
			}

		case line == "/system":
			if cfg.System == "" {
				fmt.Println("System prompt is not set.")
			} else {
				fmt.Printf("System prompt: %s\n", cfg.System)
			}

		case strings.HasPrefix(line, "/system "):
			cfg.System = strings.TrimSpace(strings.TrimPrefix(line, "/system "))
			resetHistory()
			fmt.Println("System prompt updated; history cleared.")

		case strings.HasPrefix(line, "/"):
			fmt.Printf("Unknown command %q. Type /help to list commands.\n", line)

		default:
			history = append(history, api.UserMessage(line))

			ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
			started := false
			printDelta := func(s string) {
				if !started {
					fmt.Print("ai> ")
					started = true
				}
				fmt.Print(s)
			}

			answer, err := request(ctx, history, printDelta)
			cancel()

			if err != nil {
				if started {
					fmt.Println()
				}
				history = history[:len(history)-1] // drop the failed turn
				fmt.Fprintln(os.Stderr, "error:", err)
			} else {
				if !started {
					fmt.Print("ai> ")
				}
				fmt.Println()
				if strings.TrimSpace(answer) != "" {
					history = append(history, api.AssistantMessage(answer))
				}
			}
		}

		if errors.Is(readErr, io.EOF) {
			fmt.Println()
			return nil
		}
	}
}

func printHelp() {
	fmt.Print(`Commands:
  /help             show this help
  /reset            clear the conversation history
  /model            show the current model
  /model <name>     switch model
  /system           show the system prompt
  /system <text>    set the system prompt (clears history)
  /exit, /quit      exit
`)
}
