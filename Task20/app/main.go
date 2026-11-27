package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"ai-chat-go/api"
	"ai-chat-go/config"
	"ai-chat-go/mcpboot"
	"ai-chat-go/mcpclient"
	"ai-chat-go/scheduler"
	"ai-chat-go/server"
	"ai-chat-go/tools"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	// Гарантированно закрываем MCP-сессию при выходе: по сигналу (Ctrl+C /
	// kill) и по обычному возврату из run() через defer. Без этого соединение
	// к серверу оставалось бы висеть до истечения таймаута на его стороне.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sig)
	defer mcpclient.CloseDefault()
	go func() {
		s := <-sig
		mcpclient.CloseDefault()
		mcpboot.StopAll() // останавливаем MCP-серверы, поднятые при старте
		fmt.Fprintln(os.Stderr, "\nполучен сигнал", s, "— MCP-сессия закрыта, выходим")
		os.Exit(0)
	}()

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
	mcpURL := fs.String("mcp-urls", mcpURLsString(cfg.MCPServers), "MCP servers as comma-separated slug=URL pairs; empty uses the default list")
	mcpTimeout := fs.Duration("mcp-timeout", cfg.MCPTimeout, "timeout for one MCP tool call and the session handshake")
	schedOn := fs.Bool("scheduler", cfg.Scheduler, "run the background scheduler (scheduled agent jobs)")
	schedDB := fs.String("scheduler-db", cfg.SchedulerDB, "scheduler SQLite database (jobs and run journal)")

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
  AI_MCP_URLS    MCP servers: comma-separated slug=URL pairs ("off" disables)
  AI_MCP_TIMEOUT timeout for one MCP tool call (default 30s)
  AI_MCP_AUTOSTART boot local MCP servers when down (default on; set 0 to disable)
  AI_MCP_DIR     directory of the orbit MCP server (auto-detected: ../Task17_MCP)
  AI_MCP_BIN     prebuilt orbit binary; default is "go run ./app"
  AI_MCP_LOG     orbit server log file (default bin/orbit-mcp.log inside AI_MCP_DIR)
  AI_MCP_AFFIRM_DIR  directory of the affirm MCP server (auto-detected: ../Task20_MCP)
  AI_MCP_AFFIRM_BIN  prebuilt affirm binary; default is "go run ./app"
  AI_MCP_AFFIRM_LOG  affirm server log file (default bin/affirm-mcp.log inside AI_MCP_AFFIRM_DIR)
  AI_SCHEDULER   background scheduler on/off (default on; set 0 to disable)
  AI_SCHEDULER_DB   scheduler SQLite database (default scheduler.db)
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
	if v := strings.TrimSpace(*mcpURL); v != "" {
		if v == "none" || v == "off" {
			cfg.MCPServers = nil
		} else if parsed := parseMCPFlags(v); len(parsed) > 0 {
			cfg.MCPServers = parsed
		}
	}
	if *mcpTimeout > 0 {
		cfg.MCPTimeout = *mcpTimeout
	}
	if err := cfg.Validate(); err != nil {
		return err
	}

	// Автозапуск локальных MCP-серверов (AI_MCP_AUTOSTART): перед подключением
	// проверяем endpoint'ы и поднимаем ещё не запущенные. Поддержаны slugs
	// "orbit" (Task17_MCP, сам поднимает движок Astrologer-API) и "affirm"
	// (Task20_MCP) — получается цепочка: чат → MCP-серверы → движок.
	if cfg.MCPAutostart && len(cfg.MCPServers) > 0 {
		for _, msg := range mcpboot.Ensure(context.Background(), cfg.MCPServers, mcpboot.Options{
			Dir:       cfg.MCPDir,
			Bin:       cfg.MCPBin,
			Timeout:   cfg.MCPStartTimeout,
			LogFile:   cfg.MCPLog,
			AffirmDir: cfg.MCPAffirmDir,
			AffirmBin: cfg.MCPAffirmBin,
			AffirmLog: cfg.MCPAffirmLog,
		}) {
			log.Printf("mcpboot: %s", msg)
		}
		defer mcpboot.StopAll()
	}

	// Connect to the remote MCP servers (if configured): their tools join the
	// built-in function-calling tools, and unknown tool names are proxied to
	// the owning server. A failure is not fatal: the chat works without MCP,
	// the hub keeps retrying in the background and swaps the tool list in on
	// success. The handshake is bounded by a short timeout, so a down server
	// only costs a brief warning, not a long startup hang.
	if len(cfg.MCPServers) > 0 {
		defs := make([]mcpclient.ServerDef, 0, len(cfg.MCPServers))
		for _, s := range cfg.MCPServers {
			defs = append(defs, mcpclient.ServerDef{Slug: s.Slug, Name: s.Name, URL: s.URL})
		}
		hub := mcpclient.NewHub(defs, cfg.MCPTimeout)
		hub.SetOnToolsChange(tools.SetExtra)
		tools.SetExternal(hub.Execute)
		mcpclient.Install(hub)
		errs := hub.DialAll(context.Background())
		total := hub.InfoCount()
		if len(errs) > 0 {
			for _, e := range errs {
				fmt.Fprintf(os.Stderr, "warning: MCP[%s] недоступен (%v)\n", e.Slug, e.Err)
			}
			fmt.Fprintf(os.Stderr, "warning: продолжаю без части MCP; автопереподключение каждые %s\n", mcpclient.RetryInterval)
		}
		if total > 0 {
			log.Printf("MCP: подключено серверов: %d/%d, инструментов суммарно: %d, таймаут вызова: %s",
				total, len(defs), len(tools.All()), cfg.MCPTimeout)
		}
		hub.StartAutoReconnect(0)
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

	toolOpts := []api.Option{{Tools: tools.All()}}
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
		content, _, err := tools.RunLoop(ctx, round, messages, 0)
		return content, err
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

	// Планировщик фоновых задач: горутина-тикер запускает задачи по
	// расписанию. Промптовые задачи идут через тот же конвейер (модель +
	// инструменты + MCP); задачи с полем tool зовут MCP-инструмент напрямую,
	// без модели (0 токенов). Задачи выполняются, только пока приложение
	// запущено; пропущенные (пока чат был закрыт) не догоняются — при старте
	// пересчитывается ближайший будущий момент. Задачи и журнал запусков
	// хранятся в SQLite (-scheduler-db).
	var sched *scheduler.Scheduler
	if *schedOn {
		runner := func(ctx context.Context, job scheduler.Job) (string, error) {
			// Прямой вызов MCP-инструмента: результат инструмента без
			// обработки попадает в журнал запусков.
			if name := strings.TrimSpace(job.Tool); name != "" {
				h := mcpclient.Default()
				if h == nil {
					return "", fmt.Errorf("MCP отключён (AI_MCP_URLS=off) — прямой вызов инструмента %s недоступен", name)
				}
				text, _, _, _, err := h.Execute(ctx, name, nil)
				return text, err
			}
			msgs := make([]api.Message, 0, 4)
			sys := strings.TrimSpace(job.System)
			if sys == "" {
				sys = cfg.System
			}
			if sys != "" {
				msgs = append(msgs, api.SystemMessage(sys))
			}
			msgs = append(msgs, api.UserMessage(job.Prompt))
			model := strings.TrimSpace(job.Model)
			if model == "" {
				model = cfg.Model
			}
			// Нестриминговый прогон: вывод уходит в журнал результатов, а не
			// в чат. Инструменты берём свежим срезом на каждый раунд — набор
			// меняется при переподключении MCP.
			round := func(ctx context.Context, conv []api.Message) (string, []api.ToolCall, error) {
				return client.ChatToolCalls(ctx, model, conv, cfg.MaxTokens, api.Option{Tools: tools.All()})
			}
			answer, _, err := tools.RunLoop(ctx, round, msgs, 0)
			return answer, err
		}
		sched = scheduler.New(runner, scheduler.Options{
			DB:         *schedDB,
			RunTimeout: cfg.Timeout,
		})
		if err := sched.Load(); err != nil {
			log.Printf("scheduler: %v (продолжаю с пустым списком)", err)
		}
		scheduler.Install(sched)
		defer scheduler.Install(nil)
	}

	if *useWeb && !*cliMode {
		if sched != nil {
			sched.Start()
		}
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

		case line == "/tools":
			list := tools.All()
			fmt.Printf("Доступно инструментов: %d\n", len(list))
			for _, t := range list {
				fmt.Printf("  • %-28s %s\n", t.Function.Name, firstLine(t.Function.Description))
			}

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

		case line == "/sched":
			printJobs(sched)

		case strings.HasPrefix(line, "/sched add "):
			// /sched add <имя> | <расписание> | <промпт>
			rest := strings.TrimSpace(strings.TrimPrefix(line, "/sched add "))
			parts := strings.Split(rest, "|")
			if sched == nil {
				fmt.Println("Scheduler is disabled (AI_SCHEDULER=0).")
			} else if len(parts) != 3 {
				fmt.Println("Usage: /sched add <имя> | <расписание> | <промпт>")
				fmt.Println("  пример: /sched add Брифинг | @every 1h | Собери брифинг дня")
				fmt.Println("  расписание: @every 30m / 09:00 / 09:00,18:30 / [пн,ср,пт] 09:00")
			} else {
				job, err := sched.Add(scheduler.Job{
					Name:     strings.TrimSpace(parts[0]),
					Schedule: strings.TrimSpace(parts[1]),
					Prompt:   strings.TrimSpace(parts[2]),
					Enabled:  true,
				})
				if err != nil {
					fmt.Fprintln(os.Stderr, "error:", err)
				} else {
					fmt.Printf("Задача %s добавлена; следующий запуск: %s\n", shortID(job.ID), fmtTime(job.NextRun))
				}
			}

		case strings.HasPrefix(line, "/sched run "), strings.HasPrefix(line, "/sched del "),
			strings.HasPrefix(line, "/sched on "), strings.HasPrefix(line, "/sched off "):
			fields := strings.Fields(line)
			prefix := fields[2]
			if sched == nil {
				fmt.Println("Scheduler is disabled (AI_SCHEDULER=0).")
				break
			}
			job, ok := jobByPrefix(sched.List(), prefix)
			if !ok {
				fmt.Printf("Задача %q не найдена; /sched — список.\n", prefix)
				break
			}
			var err error
			switch {
			case strings.HasPrefix(line, "/sched run "):
				err = sched.RunNow(job.ID)
				if err == nil {
					fmt.Printf("Задача %q запущена в фоне.\n", job.Name)
				}
			case strings.HasPrefix(line, "/sched del "):
				err = sched.Remove(job.ID)
				if err == nil {
					fmt.Printf("Задача %q удалена.\n", job.Name)
				}
			default:
				on := strings.HasPrefix(line, "/sched on ")
				_, err = sched.Toggle(job.ID, on)
				if err == nil {
					state := "выключена"
					if on {
						state = "включена, следующий запуск: " + fmtTime(job.NextRun)
					}
					fmt.Printf("Задача %q %s.\n", job.Name, state)
				}
			}
			if err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
			}

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

// printHelp prints the CLI command reference.
func printHelp() {
	fmt.Print(`Commands:
  /help             show this help
  /reset            clear the conversation history
  /model            show the current model
  /model <name>     switch model
  /tools            list available tools (local + MCP)
  /sched            list scheduled jobs
  /sched add <имя> | <расписание> | <промпт>
                    add a scheduled job, e.g. /sched add Брифинг | @every 1h | Собери брифинг
  /sched run <id>   run a job right now (does not shift the schedule)
  /sched on|off <id>  enable/disable a job
  /sched del <id>   delete a job
  /system           show the system prompt
  /system <text>    set the system prompt (clears history)
  /exit, /quit      exit
`)
}

// printJobs renders the scheduler job list in the terminal.
func printJobs(sched *scheduler.Scheduler) {
	if sched == nil {
		fmt.Println("Планировщик выключен (AI_SCHEDULER=0).")
		return
	}
	jobs := sched.List()
	if len(jobs) == 0 {
		fmt.Println("Задач нет. Добавить: /sched add <имя> | <расписание> | <промпт>")
		return
	}
	fmt.Printf("Задач: %d\n", len(jobs))
	for _, j := range jobs {
		state := "вкл"
		if !j.Enabled {
			state = "выкл"
		}
		fmt.Printf("  • %s  %-24s %s\n", shortID(j.ID), j.Schedule, j.Name)
		fmt.Printf("      %s; след. запуск: %s", state, fmtTime(j.NextRun))
		if j.Running {
			fmt.Print(" (выполняется)")
		}
		fmt.Println()
		if j.LastRun.Compare(time.Time{}) != 0 {
			if j.LastOK {
				fmt.Printf("      прошлый запуск: %s, ок за %d мс\n", fmtTime(j.LastRun), j.LastDurMS)
			} else {
				fmt.Printf("      прошлый запуск: %s, ошибка: %s\n", fmtTime(j.LastRun), j.LastError)
			}
		}
	}
}

// jobByPrefix finds a job by full id or its unambiguous prefix.
func jobByPrefix(jobs []scheduler.Job, prefix string) (scheduler.Job, bool) {
	if prefix == "" {
		return scheduler.Job{}, false
	}
	var found []scheduler.Job
	for _, j := range jobs {
		if j.ID == prefix || strings.HasPrefix(j.ID, prefix) {
			found = append(found, j)
		}
	}
	if len(found) == 1 {
		return found[0], true
	}
	return scheduler.Job{}, false
}

// shortID returns the first 8 characters of a job id.
func shortID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}

// fmtTime renders a time as "02.01 15:04" (or "—" for the zero value).
func fmtTime(t time.Time) string {
	if t.Compare(time.Time{}) == 0 {
		return "—"
	}
	return t.Format("02.01 15:04")
}

// firstLine returns the first non-empty line of a multi-line string.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

// mcpURLsString renders the server list back into the slug=URL pair format.
func mcpURLsString(servers []config.MCPServer) string {
	if len(servers) == 0 {
		return ""
	}
	parts := make([]string, 0, len(servers))
	for _, s := range servers {
		parts = append(parts, s.Slug+"="+s.URL)
	}
	return strings.Join(parts, ",")
}

// parseMCPFlags parses the -mcp-urls flag value (slug=URL pairs); display
// names are taken from the configured list when the slug matches.
func parseMCPFlags(v string) []config.MCPServer {
	var out []config.MCPServer
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		slug, url, _ := strings.Cut(part, "=")
		slug, url = strings.TrimSpace(slug), strings.TrimRight(strings.TrimSpace(url), "/")
		if slug == "" || url == "" {
			continue
		}
		name := ""
		for _, s := range config.DefaultMCPServers {
			if strings.EqualFold(s.Slug, slug) {
				name = s.Name
				break
			}
		}
		if name == "" {
			name = strings.ToUpper(slug[:1]) + slug[1:]
		}
		out = append(out, config.MCPServer{Slug: slug, Name: name, URL: url})
	}
	return out
}
