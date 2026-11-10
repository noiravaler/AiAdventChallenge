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
// described by Tool, Args (the raw JSON arguments) and Result.
type RunStep struct {
	Text   string `json:"text"`
	Tool   string `json:"tool,omitempty"`
	Args   string `json:"args,omitempty"`
	Result string `json:"result,omitempty"`
}

// RunLoop drives a tool-calling conversation. Each round asks the model; if it
// requests tools they are executed and their results fed back, until the model
// returns a plain answer or the round budget (maxRounds, <=0 for the default)
// is exhausted. The final text, the ordered list of interaction steps and the
// intermediate tool messages (assistant tool_calls + tool results, in order) are
// returned; the latter are what must be persisted so a restart can replay the
// same conversation. On error `added` is nil.
func RunLoop(ctx context.Context, round RoundFunc, initial []api.Message, maxRounds int) (content string, steps []RunStep, added []api.Message, err error) {
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
		if len(calls) == 0 {
			steps = append(steps, RunStep{Text: fmt.Sprintf("Раунд %d: ответ модели получен", i+1)})
			return content, steps, added, nil
		}
		steps = append(steps, RunStep{Text: fmt.Sprintf("Раунд %d: модель запросила инструменты", i+1)})
		if i >= maxRounds {
			return content, steps, added, nil
		}
		am := api.AssistantToolCallMessage(calls)
		conversation = append(conversation, am)
		added = append(added, am)
		for _, call := range calls {
			result, rerr := Execute(call.Function.Name, json.RawMessage(call.Function.Arguments))
			if rerr != nil {
				result = "ошибка: " + rerr.Error()
			}
			steps = append(steps, RunStep{
				Tool:   call.Function.Name,
				Args:   call.Function.Arguments,
				Result: result,
			})
			tr := api.ToolResultMessage(call.ID, result)
			conversation = append(conversation, tr)
			added = append(added, tr)
		}
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
