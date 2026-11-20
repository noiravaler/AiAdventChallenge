package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"

	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/server"
	"ai-chat-go/storage"
	"ai-chat-go/tools"
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
	client.DebugLog = nil // консольное JSON-логирование отключено; панель «Детали» читает их из памяти

	history := make([]api.Message, 0, 16)
	resetHistory := func() {
		history = history[:0]
		if cfg.System != "" {
			history = append(history, api.SystemMessage(cfg.System))
		}
	}
	resetHistory()

	toolOpts := []api.Option{{Tools: tools.Definitions()}}
	request := func(ctx context.Context, messages []api.Message, onDelta func(string)) (string, error) {
		round := func(ctx context.Context, conv []api.Message) (string, []api.ToolCall, error) {
			if *noStream {
				content, calls, err := client.ChatToolCalls(ctx, cfg.Model, conv, cfg.MaxTokens, toolOpts...)
				if err == nil && content != "" && onDelta != nil {
					onDelta(content)
				}
				return content, calls, err
			}
			return client.StreamChatToolCalls(ctx, cfg.Model, conv, cfg.MaxTokens, onDelta, nil, toolOpts...)
		}
		content, _, _, err := tools.RunLoop(ctx, round, messages, 0)
		return content, err
	}

	// Инварианты: общая таблица SQLite с веб-версией; в CLI меняются только
	// слэш-командами /invariant (явная команда пользователя). Судья — та же
	// схема, что и в веб-сервере: AI_INVARIANTS_JUDGE_MODEL или первая модель
	// списка, отличная от отвечающей.
	var invDB *storage.DB
	invGate := func() []storage.Invariant {
		if invDB == nil {
			if cfg.DBFile == "" {
				return nil
			}
			db, err := storage.Open(cfg.DBFile)
			if err != nil {
				fmt.Fprintln(os.Stderr, "invariants:", err)
				return nil
			}
			invDB = db
		}
		items, err := invDB.ListInvariants()
		if err != nil {
			fmt.Fprintln(os.Stderr, "invariants:", err)
			return nil
		}
		return server.EnabledOnly(items)
	}
	judgeModel := func() string {
		if j := strings.TrimSpace(cfg.InvariantsJudge); j != "" {
			return j
		}
		// Дефолт — маленькая не-reasoning модель (см. config.DefaultInvariantsJudge).
		if config.DefaultInvariantsJudge != cfg.Model && slices.Contains(cfg.Models, config.DefaultInvariantsJudge) {
			return config.DefaultInvariantsJudge
		}
		for _, m := range cfg.Models {
			if m != cfg.Model {
				return m
			}
		}
		return cfg.Model
	}

	if fs.NArg() > 0 {
		prompt := strings.Join(fs.Args(), " ")
		history = append(history, api.UserMessage(prompt))

		ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
		defer cancel()

		// Инвариантный гейт: буферизация + проверка до вывода.
		gate := invGate()
		stream := func(s string) { fmt.Print(s) }
		if len(gate) > 0 {
			stream = func(string) {}
		}
		answer, err := request(ctx, history, stream)
		if err == nil && len(gate) > 0 && strings.TrimSpace(answer) != "" {
			if chk := server.CheckInvariants(ctx, client, judgeModel(), gate, answer, cfg.Timeout); chk != nil && chk.Violated {
				answer = server.RefusalText(chk)
			}
		}
		if err != nil {
			return err
		}
		if len(gate) > 0 {
			fmt.Print(answer)
		}
		fmt.Println()
		return nil
	}

	if *useWeb && !*cliMode {
		return server.Run(client, cfg, *webPort, !*noOpen)
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

		case line == "/invariant" || strings.HasPrefix(line, "/invariant "):
			runInvariantCommand(invDB, strings.TrimSpace(strings.TrimPrefix(line, "/invariant")))

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

			// Инвариантный гейт: при включённых инвариантах ответ буферизуется
			// и выдаётся только после проверки (regex + LLM-судья) — в терминал
			// не печатается ничего до вердикта.
			gate := invGate()
			stream := printDelta
			if len(gate) > 0 {
				stream = func(string) {}
			}

			answer, err := request(ctx, history, stream)
			if err == nil && len(gate) > 0 && strings.TrimSpace(answer) != "" {
				if chk := server.CheckInvariants(ctx, client, judgeModel(), gate, answer, cfg.Timeout); chk != nil && chk.Violated {
					answer = server.RefusalText(chk)
				}
			}
			cancel()

			if err != nil {
				if started {
					fmt.Println()
				}
				history = history[:len(history)-1] // drop the failed turn
				fmt.Fprintln(os.Stderr, "error:", err)
			} else {
				if len(gate) > 0 {
					if strings.TrimSpace(answer) != "" {
						fmt.Print("ai> " + answer)
					}
				} else if !started {
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
  /invariant        list invariants (same table as the web panel)
  /invariant add <text>            add an invariant
  /invariant edit <id> <text>      change an invariant's wording
  /invariant rm <id>               delete an invariant
  /invariant on <id> / off <id>    enable / disable in the gate
  /exit, /quit      exit
`)
}

// runInvariantCommand handles the explicit /invariant commands of the CLI:
// the only way to change the invariant list from the terminal (the model has
// no path to it — the gate reads the table, nothing writes it from a dialog).
func runInvariantCommand(db *storage.DB, rest string) {
	if db == nil {
		fmt.Println("Персистентность отключена (AI_DB_FILE пуст) — инварианты недоступны.")
		return
	}
	cmd, arg, _ := strings.Cut(strings.TrimSpace(rest), " ")
	arg = strings.TrimSpace(arg)
	switch cmd {
	case "":
		items, err := db.ListInvariants()
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		if len(items) == 0 {
			fmt.Println("Инвариантов нет: проверка ответов выключена. Добавьте: /invariant add <текст>")
			return
		}
		for _, it := range items {
			state := "вкл"
			if !it.Enabled {
				state = "выкл"
			}
			fmt.Printf("#%d [%s] %s\n", it.ID, state, it.Text)
			if it.Patterns != "" {
				fmt.Printf("      шаблоны: %s\n", it.Patterns)
			}
		}
		journal, err := db.ListInvariantLog(10)
		if err != nil {
			return
		}
		if len(journal) > 0 {
			fmt.Println("Последние изменения:")
			for _, l := range journal {
				fmt.Printf("  %s — %s (старое: %q, новое: %q)\n",
					l.At.Format("02.01.2006 15:04:05"), l.Action, l.OldValue, l.NewValue)
			}
		}
	case "add":
		if arg == "" {
			fmt.Println("Usage: /invariant add <текст>")
			return
		}
		it, err := db.AddInvariant(arg, "", storage.InvSourceCLI)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		fmt.Printf("Инвариант #%d добавлен: %s\n", it.ID, it.Text)
	case "edit":
		id, text, ok := strings.Cut(arg, " ")
		if !ok || strings.TrimSpace(text) == "" {
			fmt.Println("Usage: /invariant edit <id> <новый текст>")
			return
		}
		n, err := strconv.ParseInt(strings.TrimSpace(id), 10, 64)
		if err != nil {
			fmt.Println("Usage: /invariant edit <id> <новый текст>")
			return
		}
		it, err := db.UpdateInvariant(n, strings.TrimSpace(text), "", storage.InvSourceCLI)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		fmt.Printf("Инвариант #%d изменён (старая формулировка в журнале): %s\n", it.ID, it.Text)
	case "rm":
		n, err := strconv.ParseInt(strings.TrimSpace(arg), 10, 64)
		if err != nil {
			fmt.Println("Usage: /invariant rm <id>")
			return
		}
		if err := db.DeleteInvariant(n, storage.InvSourceCLI); err != nil {
			fmt.Println("error:", err)
			return
		}
		fmt.Printf("Инвариант #%d удалён (формулировка осталась в журнале).\n", n)
	case "on", "off":
		n, err := strconv.ParseInt(strings.TrimSpace(arg), 10, 64)
		if err != nil {
			fmt.Printf("Usage: /invariant %s <id>\n", cmd)
			return
		}
		if err := db.SetInvariantEnabled(n, cmd == "on", storage.InvSourceCLI); err != nil {
			fmt.Println("error:", err)
			return
		}
		fmt.Printf("Инвариант #%d %s.\n", n, map[string]string{"on": "включён", "off": "выключен"}[cmd])
	default:
		fmt.Println("Usage: /invariant [list | add <текст> | edit <id> <текст> | rm <id> | on <id> | off <id>]")
	}
}
