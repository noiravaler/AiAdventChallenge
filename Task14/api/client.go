// Package api implements a small client for OpenAI-compatible chat completion APIs.
package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Message is a single chat message. For assistant tool-call turns the Content is
// empty and ToolCalls carries the requested functions; a tool result is a
// message with Role "tool", ToolCallID matching the call and the result in
// Content.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// SystemMessage creates a system message.
func SystemMessage(text string) Message { return Message{Role: "system", Content: text} }

// UserMessage creates a user message.
func UserMessage(text string) Message { return Message{Role: "user", Content: text} }

// AssistantMessage creates an assistant message.
func AssistantMessage(text string) Message { return Message{Role: "assistant", Content: text} }

// Tool is an OpenAI-compatible function tool definition sent in the request.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a single callable function exposed to the model.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolCall is a function invocation requested by the model. Arguments is the
// raw JSON argument object as produced by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction is the name and JSON-encoded arguments of a tool call.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// AssistantToolCallMessage builds the assistant turn that returns the tool calls
// the model requested (no visible content). It is sent together with the
// matching ToolResultMessage turns to continue the conversation.
func AssistantToolCallMessage(toolCalls []ToolCall) Message {
	return Message{Role: "assistant", ToolCalls: toolCalls}
}

// ToolResultMessage builds the tool turn holding the result of one tool call.
func ToolResultMessage(toolCallID, result string) Message {
	return Message{Role: "tool", ToolCallID: toolCallID, Content: result}
}

// Usage carries the token accounting returned by an OpenAI-compatible
// completion response (present in non-streaming bodies and, for streaming,
// in the final chunk when stream_options.include_usage is set).
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// streamOptions asks a streaming endpoint to include the usage block in the
// final chunk. OpenAI-compatible providers that do not support it simply
// ignore the field.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Client talks to an OpenAI-compatible /chat/completions endpoint.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client

	// DebugLog, when not nil, receives one JSON object per line describing
	// every API request and the corresponding response.
	DebugLog io.Writer
	logMu    sync.Mutex

	lastRequest string // compact JSON of the last request body
	lastResp    string // compact JSON of the last response
	lastFinish  string // last finish_reason from the most recent stream
	lastUsage   Usage  // last usage reported by the endpoint
}

// New creates a client. A zero timeout means no client-level timeout.
func New(baseURL, apiKey string, timeout time.Duration) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// Option customizes a single chat request, e.g. by asking for a JSON response.
type Option struct {
	// ResponseFormat is one of "json", "jsonschema", "jsonschema_structured",
	// or anything else (plain text).
	ResponseFormat string
	// Schema is the JSON Schema used when ResponseFormat is jsonschema or
	// jsonschema_structured. It must itself be valid JSON.
	Schema string
	// Stop is an optional stop sequence: generation stops when the model
	// emits this text. Empty means no stop sequence is sent.
	Stop string
	// Tools are the OpenAI function tools exposed to the model. Empty (nil)
	// disables function calling.
	Tools []Tool
}

// responseFormat builds the response_format body for an Option.
func (o Option) responseFormat() json.RawMessage {
	switch o.ResponseFormat {
	case "json":
		return json.RawMessage(`{"type":"json_object"}`)
	case "jsonschema", "jsonschema_structured":
		if strings.TrimSpace(o.Schema) == "" {
			return nil
		}
		strict := o.ResponseFormat == "jsonschema_structured"
		obj := map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"strict": strict,
				"schema": json.RawMessage(o.Schema),
			},
		}
		b, err := json.Marshal(obj)
		if err != nil {
			return nil
		}
		return json.RawMessage(b)
	default:
		return nil
	}
}

// stopSequence returns the stop sequence as the array form accepted by the
// OpenAI-compatible API, or nil when it is empty.
func (o Option) stopSequence() []string {
	s := strings.TrimSpace(o.Stop)
	if s == "" {
		return nil
	}
	return []string{s}
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Stream         bool            `json:"stream,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	StreamOptions  *streamOptions  `json:"stream_options,omitempty"`
}

type apiErrorBody struct {
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			// tool_calls deltas stream one fragment per item; see the
			// accumulator used in streamChat.
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			// Reasoning models stream their thinking trace before the answer;
			// providers use either field name for it.
			ReasoningRC string `json:"reasoning_content"`
			ReasoningR  string `json:"reasoning"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (c *Client) logRequest(payload chatRequest) {
	c.record("request", map[string]any{
		"time":    time.Now().UTC().Format(time.RFC3339),
		"type":    "request",
		"url":     c.baseURL + "/chat/completions",
		"payload": payload,
	})
}

func (c *Client) logResponse(content string, err error) {
	event := map[string]any{
		"time": time.Now().UTC().Format(time.RFC3339),
		"type": "response",
	}
	if err != nil {
		event["error"] = err.Error()
	} else {
		event["content"] = content
	}
	c.record("response", event)
}

// record stores the last request/response JSON (for the "Детали" panel) and,
// when DebugLog is configured, also writes one line to it.
func (c *Client) record(kind string, event map[string]any) {
	b, err := json.Marshal(event)
	if err != nil {
		return
	}
	c.logMu.Lock()
	defer c.logMu.Unlock()
	if c.DebugLog != nil {
		_, _ = c.DebugLog.Write(append(b, '\n'))
	}
	if kind == "request" {
		c.lastRequest = string(b)
	} else {
		c.lastResp = string(b)
	}
}

// LastExchange returns the compact JSON of the last request and response body,
// used by the "Детали" side panel.
func (c *Client) LastExchange() (request, response string) {
	c.logMu.Lock()
	defer c.logMu.Unlock()
	return c.lastRequest, c.lastResp
}

// LastFinishReason returns the finish_reason observed in the most recent
// streaming response, or "" if none was received (e.g. the stream aborted).
func (c *Client) LastFinishReason() string {
	c.logMu.Lock()
	defer c.logMu.Unlock()
	return c.lastFinish
}

// LastUsage returns the token usage reported by the most recent completion
// response (streaming or not), or the zero value when the endpoint did not
// report it.
func (c *Client) LastUsage() Usage {
	c.logMu.Lock()
	defer c.logMu.Unlock()
	return c.lastUsage
}

// ListModels returns the IDs of the models available on the server's
// OpenAI-compatible /models endpoint.
func (c *Client) ListModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, responseError(resp)
	}

	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode models: %w", err)
	}
	ids := make([]string, 0, len(body.Data))
	for _, m := range body.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	return ids, nil
}

// Chat sends a non-streaming request and returns the full answer.
func (c *Client) Chat(ctx context.Context, model string, messages []Message, maxTokens int, opts ...Option) (answer string, err error) {
	content, _, err := c.ChatToolCalls(ctx, model, messages, maxTokens, opts...)
	return content, err
}

// ChatToolCalls sends a non-streaming request and returns both the answer text
// and any tool calls the model requested.
func (c *Client) ChatToolCalls(ctx context.Context, model string, messages []Message, maxTokens int, opts ...Option) (answer string, calls []ToolCall, err error) {
	defer func() { c.logResponse(answer, err) }()

	req := chatRequest{Model: model, Messages: messages, MaxTokens: maxTokens}
	if len(opts) > 0 {
		req.ResponseFormat = opts[0].responseFormat()
		req.Stop = opts[0].stopSequence()
		req.Tools = opts[0].Tools
	}
	resp, err := c.post(ctx, req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	var body chatResponse
	if err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&body); err != nil {
		return "", nil, fmt.Errorf("decode response: %w", err)
	}
	c.logMu.Lock()
	c.lastUsage = body.Usage
	c.logMu.Unlock()
	if body.Error != nil {
		return "", nil, errors.New(body.Error.Message)
	}
	if len(body.Choices) == 0 {
		return "", nil, errors.New("API returned no choices")
	}
	return body.Choices[0].Message.Content, body.Choices[0].Message.ToolCalls, nil
}

// StreamChat sends a streaming request, calls onDelta for every token chunk
// and returns the complete answer. onDelta may be nil.
func (c *Client) StreamChat(ctx context.Context, model string, messages []Message, maxTokens int, onDelta func(string), opts ...Option) (answer string, err error) {
	answer, _, err = c.streamChat(ctx, model, messages, maxTokens, onDelta, nil, opts...)
	return
}

// StreamChatWithActivity behaves like StreamChat but additionally calls
// onActivity for every received model token (content or reasoning trace).
// Callers use it to detect that the model is alive while visible content has
// not started yet. onActivity may be nil.
func (c *Client) StreamChatWithActivity(ctx context.Context, model string, messages []Message, maxTokens int, onDelta func(string), onActivity func(), opts ...Option) (answer string, err error) {
	answer, _, err = c.streamChat(ctx, model, messages, maxTokens, onDelta, onActivity, opts...)
	return
}

// StreamChatToolCalls behaves like StreamChatWithActivity but additionally
// returns the tool calls the model requested during the stream. Callers use it
// to drive function-calling loops.
func (c *Client) StreamChatToolCalls(ctx context.Context, model string, messages []Message, maxTokens int, onDelta func(string), onActivity func(), opts ...Option) (answer string, calls []ToolCall, err error) {
	return c.streamChat(ctx, model, messages, maxTokens, onDelta, onActivity, opts...)
}

// accumToolCall assembles one streamed tool call across delta fragments.
type accumToolCall struct {
	index int
	id    string
	name  string
	args  strings.Builder
}

func (c *Client) streamChat(ctx context.Context, model string, messages []Message, maxTokens int, onDelta func(string), onActivity func(), opts ...Option) (answer string, calls []ToolCall, err error) {
	defer func() { c.logResponse(answer, err) }()
	c.logMu.Lock()
	c.lastFinish = ""
	c.lastUsage = Usage{}
	c.logMu.Unlock()

	req := chatRequest{
		Model:     model,
		Messages:  messages,
		Stream:    true,
		MaxTokens: maxTokens,
		StreamOptions: &streamOptions{
			IncludeUsage: true,
		},
	}
	if len(opts) > 0 {
		req.ResponseFormat = opts[0].responseFormat()
		req.Stop = opts[0].stopSequence()
		req.Tools = opts[0].Tools
	}
	resp, err := c.post(ctx, req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	var sb strings.Builder
	accum := make([]*accumToolCall, 0)
	indexToPos := make(map[int]int)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4<<20)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return sb.String(), toolCallList(accum, indexToPos), nil
		}

		var chunk streamChunk
		if err = json.Unmarshal([]byte(data), &chunk); err != nil {
			return sb.String(), nil, fmt.Errorf("decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return sb.String(), nil, errors.New(chunk.Error.Message)
		}
		// With stream_options.include_usage the final chunk carries the token
		// accounting (usually with an empty choices array).
		if chunk.Usage.TotalTokens > 0 || chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			c.logMu.Lock()
			c.lastUsage = chunk.Usage
			c.logMu.Unlock()
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				c.logMu.Lock()
				c.lastFinish = *choice.FinishReason
				c.logMu.Unlock()
			}
			if choice.Delta.Content != "" {
				sb.WriteString(choice.Delta.Content)
				if onDelta != nil {
					onDelta(choice.Delta.Content)
				}
				continue
			}
			// No visible content in this fragment: it is either a reasoning
			// trace or a fragment of a tool call. Both prove the model is
			// generating, so surface them through onActivity.
			if len(choice.Delta.ToolCalls) > 0 {
				for _, tc := range choice.Delta.ToolCalls {
					pos, ok := indexToPos[tc.Index]
					if !ok {
						pos = len(accum)
						indexToPos[tc.Index] = pos
						accum = append(accum, &accumToolCall{index: tc.Index})
					}
					a := accum[pos]
					if tc.ID != "" {
						a.id = tc.ID
					}
					if tc.Function.Name != "" {
						a.name = tc.Function.Name
					}
					a.args.WriteString(tc.Function.Arguments)
				}
				if onActivity != nil {
					onActivity()
				}
				continue
			}
			if onActivity != nil && (choice.Delta.ReasoningRC != "" || choice.Delta.ReasoningR != "") {
				onActivity()
			}
		}
	}
	if err = scanner.Err(); err != nil {
		return sb.String(), nil, fmt.Errorf("read stream: %w", err)
	}
	return sb.String(), toolCallList(accum, indexToPos), nil
}

// toolCallList orders the accumulated tool calls by their stream index and
// converts them to the ToolCall type sent back in the conversation.
func toolCallList(accum []*accumToolCall, indexToPos map[int]int) []ToolCall {
	if len(accum) == 0 {
		return nil
	}
	sort.Slice(accum, func(i, j int) bool { return accum[i].index < accum[j].index })
	calls := make([]ToolCall, len(accum))
	for i, a := range accum {
		calls[i] = ToolCall{
			ID:   a.id,
			Type: "function",
			Function: ToolCallFunction{
				Name:      a.name,
				Arguments: a.args.String(),
			},
		}
	}
	return calls
}

func (c *Client) post(ctx context.Context, payload chatRequest) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	c.logRequest(payload)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream, application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		return nil, responseError(resp)
	}
	return resp, nil
}

func responseError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))

	var body apiErrorBody
	if json.Unmarshal(raw, &body) == nil && body.Error != nil && body.Error.Message != "" {
		return fmt.Errorf("API error %d: %s", resp.StatusCode, body.Error.Message)
	}
	text := strings.TrimSpace(string(raw))
	if text == "" {
		return fmt.Errorf("API error %d", resp.StatusCode)
	}
	return fmt.Errorf("API error %d: %s", resp.StatusCode, text)
}
