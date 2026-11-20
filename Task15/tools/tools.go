// Package tools implements the function-calling tools exposed to the chat
// model: the current time in a region and a six-sided die roll. It also drives
// the tool-calling loop so callers (web server and CLI) share one
// implementation.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	"ai-chat-go/api"
)

// maxToolRounds bounds how many times the model may call tools in one turn;
// it prevents infinite tool loops.
const maxToolRounds = 6

// RoundFunc executes one chat round (a single API call) for the given
// conversation and returns the accumulated text and any tool calls the model
// requested.
type RoundFunc func(ctx context.Context, conversation []api.Message) (content string, toolCalls []api.ToolCall, err error)

// ExecFunc replaces the default Execute inside RunLoop so a caller can
// intercept tool calls with stateful tools of its own (the web server's
// memory tools) and delegate the rest to Execute.
type ExecFunc func(name string, args json.RawMessage) (string, error)

// MemoryDefinitions returns the OpenAI function definitions of the memory
// tools. They are executed by the caller (the web server, via ExecFunc),
// not by this package: they need access to the conversation state.
func MemoryDefinitions() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "remember",
				Description: "Сохраняет факт в долговременную память, чтобы он был доступен и в будущих диалогах (имя пользователя, его предпочтения, принятые решения, важные знания). text — формулировка вида «ключ: значение» либо законченная фраза; kind — профиль (о пользователе), решение (о проекте) или знание (по умолчанию). По умолчанию записи вида profile/knowledge хранят одно актуальное значение (новая формулировка ЗАМЕНЯЕТ старую), decision — накапливаются списком с датами. policy и ttl позволяют переопределить это поведение.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"text": map[string]any{
							"type":        "string",
							"description": "Факт одной фразой, лучше в виде «ключ: значение», например «имя: Аня».",
						},
						"kind": map[string]any{
							"type":        "string",
							"enum":        []string{"profile", "decision", "knowledge"},
							"description": "Тип записи: profile — сведения о пользователе, decision — принятые решения, knowledge — прочие знания.",
						},
						"policy": map[string]any{
							"type":        "string",
							"enum":        []string{"latest", "append"},
							"description": "Необязательное правило жизни факта: latest — хранить одно актуальное значение (новое заменяет старое, срок/статус/версия), append — накапливать записи списком с датами (решения, история изменений).",
						},
						"ttl": map[string]any{
							"type":        "string",
							"enum":        []string{"task", "permanent"},
							"description": "Необязательный срок жизни. task — факт удаляется автоматически, когда задача вкладки завершена (done) или сброшена (reset). permanent — факт остаётся навсегда, даже если записан во время задачи. По умолчанию знания (kind=knowledge), записанные во время активной задачи, считаются её частью (план, шаги, даты, статусы) и удаляются вместе с ней; профиль и решения остаются.",
						},
					},
					"required": []string{"text"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "recall",
				Description: "Ищет записи в долговременной памяти по ключевым словам. Используй, когда нужен факт, который мог быть сохранён ранее (в этом или прошлых диалогах).",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "Поисковый запрос ключевыми словами.",
						},
						"kind": map[string]any{
							"type":        "string",
							"enum":        []string{"profile", "decision", "knowledge"},
							"description": "Необязательный фильтр по типу записей.",
						},
					},
					"required": []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "forget",
				Description: "Удаляет запись из долговременной памяти по её id (id возвращает инструмент recall).",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{
							"type":        "string",
							"description": "Идентификатор записи, например k3.",
						},
					},
					"required": []string{"id"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "task_note",
				Description: "Добавляет заметку в рабочую память текущей задачи (артефакты, ограничения, договорённости), чтобы они не потерялись при сжатии контекста. Вызывай сразу, как только в диалоге появляется задача, срок или важная договорённость.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"text": map[string]any{
							"type":        "string",
							"description": "Заметка одной строкой.",
						},
					},
					"required": []string{"text"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "task_state",
				Description: "Управляет состоянием текущей задачи как конечным автоматом: этапы planning → execution → validation → done (из validation можно вернуться в execution, если найдены проблемы; из done — начать новую задачу через planning или убрать завершённую запись через reset). Порядок жёсткий: сначала set planning, затем plan (утверждённый план — без него transition в execution отклоняется), затем execution, затем validation, и только после успешной валидации — done (финал без валидации невозможен). Вызывай, когда задача сформулирована (action=set, stage=planning), когда этап меняется (action=advance или set), когда сформулирован план шагов (action=plan), и когда нужно узнать текущее состояние (action=get). Переходы строгие: server проверяет их кодом до применения и отклоняет недопустимые с объяснением причины. Этап отражает фактическую фазу работы: утвердили план — сразу advance в execution, не оставляй задачу в planning с готовым планом. За один ответ — не более одного перехода этапа; финальный done требует полностью выполненного плана. В рабочей памяти завершённая задача (done) отображается краткой записью-историей без инструкций переходов; она убирается action=reset или перезаписывается новой задачей.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"action": map[string]any{
							"type":        "string",
							"enum":        []string{"get", "advance", "set", "plan", "reset"},
							"description": "get — показать состояние; advance — перейти к следующему этапу; set — установить этап (нужен stage); plan — задать список шагов плана (plan) и сколько уже выполнено (completed); reset — отменить задачу и очистить состояние (из planning) или убрать завершённую запись (из done).",
						},
						"stage": map[string]any{
							"type":        "string",
							"enum":        []string{"planning", "execution", "validation", "done"},
							"description": "Целевой этап (только для action=set).",
						},
						"step": map[string]any{
							"type":        "string",
							"description": "Текущий шаг одной строкой (опционально, для advance/set).",
						},
						"expect": map[string]any{
							"type":        "string",
							"description": "Ожидаемое действие одной строкой (опционально, для advance/set).",
						},
						"plan": map[string]any{
							"type":        "array",
							"items":       map[string]any{"type": "string"},
							"description": "Названия шагов плана по порядку (только для action=plan); заменяет весь план.",
						},
						"completed": map[string]any{
							"type":        "integer",
							"description": "Сколько шагов плана уже выполнено (0..len(plan)); следующий шаг становится активным (для action=plan).",
						},
					},
					"required": []string{"action"},
				},
			},
		},
	}
}

// Definitions returns the OpenAI function tool definitions exposed to the model.
func Definitions() []api.Tool {
	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_current_time",
				Description: "Возвращает текущие дату и время в заданном регионе (IANA-таймзона, например Europe/Moscow).",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"region": map[string]any{
							"type":        "string",
							"description": "Название IANA-таймзоны, например Europe/Moscow, Asia/Tokyo или America/New_York.",
						},
					},
					"required": []string{"region"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "roll_dice",
				Description: "Бросает шестигранный игральный кубик (d6): случайное целое от 1 до 6, каждое равновероятно. Используй ТОЛЬКО когда нужен именно игральный кубик. Для подбрасывания монеты/жребия инструмент НЕ использовать — просто ответь случайно.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	}
}

// Execute runs a single tool by name with the given JSON arguments and returns
// the textual result to feed back to the model.
func Execute(name string, args json.RawMessage) (string, error) {
	fn, ok := executors[name]
	if !ok {
		return "", fmt.Errorf("неизвестный инструмент: %s", name)
	}
	return fn(args)
}

var executors = map[string]func(json.RawMessage) (string, error){
	"get_current_time": currentTime,
	"roll_dice":        rollDice,
}

// RunStep is one logged entry of the tool-calling loop, shown as a "Шаг" in
// the "Детали" panel. A model round is described by Text; a tool execution is
// described by Tool, Args (the raw JSON arguments) and Result. Round steps may
// also carry the API call's token usage (filled in by the caller, which owns
// the usage accounting). Sub holds nested steps: tool executions requested in
// that round and auxiliary work (compression, validation) attached by the
// caller to the round they belong to.
type RunStep struct {
	Text             string    `json:"text"`
	Tool             string    `json:"tool,omitempty"`
	Args             string    `json:"args,omitempty"`
	Result           string    `json:"result,omitempty"`
	PromptTokens     int       `json:"prompt_tokens,omitempty"`
	CompletionTokens int       `json:"completion_tokens,omitempty"`
	Sub              []RunStep `json:"sub,omitempty"`
}

// RunLoop drives a tool-calling conversation. Each round asks the model; if it
// requests tools they are executed and their results fed back, until the model
// returns a plain answer or the round budget (maxRounds, <=0 for the default)
// is exhausted. The final text, the ordered list of interaction steps and the
// intermediate tool messages (assistant tool_calls + tool results, in order) are
// returned; the latter are what must be persisted so a restart can replay the
// same conversation. On error `added` is nil.
//
// An optional ExecFunc (exactly one is used) overrides the default Execute for
// every tool of the loop; the web server passes one that first handles its
// stateful memory tools and delegates the rest to Execute.
//
// Tool executions are logged as steps nested inside the round that requested
// them (RunStep.Sub), so the "Шаги" panel shows one top-level entry per round.
func RunLoop(ctx context.Context, round RoundFunc, initial []api.Message, maxRounds int, exec ...ExecFunc) (content string, steps []RunStep, added []api.Message, err error) {
	runTool := Execute
	if len(exec) > 0 && exec[0] != nil {
		runTool = exec[0]
	}
	if maxRounds <= 0 {
		maxRounds = maxToolRounds
	}
	conversation := append([]api.Message(nil), initial...)
	for i := 0; ; i++ {
		content, calls, err := round(ctx, conversation)
		if err != nil {
			steps = append(steps, RunStep{Text: fmt.Sprintf("Раунд %d: ошибка обращения к модели", i+1)})
			return content, steps, nil, err
		}
		rs := RunStep{}
		if len(calls) == 0 {
			rs.Text = fmt.Sprintf("Раунд %d: ответ модели получен", i+1)
			steps = append(steps, rs)
			return content, steps, added, nil
		}
		rs.Text = fmt.Sprintf("Раунд %d: модель запросила инструменты", i+1)
		if i >= maxRounds {
			steps = append(steps, rs)
			return content, steps, added, nil
		}
		am := api.AssistantToolCallMessage(calls)
		conversation = append(conversation, am)
		added = append(added, am)
		for _, call := range calls {
			result, rerr := runTool(call.Function.Name, json.RawMessage(call.Function.Arguments))
			if rerr != nil {
				result = "ошибка: " + rerr.Error()
			}
			rs.Sub = append(rs.Sub, RunStep{
				Tool:   call.Function.Name,
				Args:   call.Function.Arguments,
				Result: result,
			})
			tr := api.ToolResultMessage(call.ID, result)
			conversation = append(conversation, tr)
			added = append(added, tr)
		}
		steps = append(steps, rs)
	}
}

// currentTime returns the current date and time in the requested IANA timezone.
func currentTime(args json.RawMessage) (string, error) {
	var in struct {
		Region string `json:"region"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", fmt.Errorf("некорректные аргументы: %w", err)
	}
	region := strings.TrimSpace(in.Region)
	if region == "" {
		return "", fmt.Errorf("не указан регион")
	}
	loc, err := time.LoadLocation(region)
	if err != nil {
		return "", fmt.Errorf("неизвестный регион/таймзона: %s", region)
	}
	return fmt.Sprintf("%s (%s)", time.Now().In(loc).Format(time.RFC3339), loc), nil
}

// rollDice returns a uniformly random integer from 1 to 6.
func rollDice(_ json.RawMessage) (string, error) {
	return fmt.Sprintf("%d", rand.IntN(6)+1), nil
}
