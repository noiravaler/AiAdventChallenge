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
	"strings"
	"sync"
	"time"
)

// Message is a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// SystemMessage creates a system message.
func SystemMessage(text string) Message { return Message{Role: "system", Content: text} }

// UserMessage creates a user message.
func UserMessage(text string) Message { return Message{Role: "user", Content: text} }

// AssistantMessage creates an assistant message.
func AssistantMessage(text string) Message { return Message{Role: "assistant", Content: text} }

// Client talks to an OpenAI-compatible /chat/completions endpoint.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client

	// DebugLog, when not nil, receives one JSON object per line describing
	// every API request and the corresponding response.
	DebugLog io.Writer
	logMu    sync.Mutex
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

type chatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream,omitempty"`
	MaxTokens int       `json:"max_tokens,omitempty"`
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
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// logLine writes one compact JSON object followed by a newline.
func (c *Client) logLine(v any) {
	if c.DebugLog == nil {
		return
	}
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	c.logMu.Lock()
	defer c.logMu.Unlock()
	_, _ = c.DebugLog.Write(append(b, '\n'))
}

func (c *Client) logRequest(payload chatRequest) {
	if c.DebugLog == nil {
		return
	}
	c.logLine(map[string]any{
		"time":    time.Now().UTC().Format(time.RFC3339),
		"type":    "request",
		"url":     c.baseURL + "/chat/completions",
		"payload": payload,
	})
}

func (c *Client) logResponse(content string, err error) {
	if c.DebugLog == nil {
		return
	}
	event := map[string]any{
		"time": time.Now().UTC().Format(time.RFC3339),
		"type": "response",
	}
	if err != nil {
		event["error"] = err.Error()
	} else {
		event["content"] = content
	}
	c.logLine(event)
}

// Chat sends a non-streaming request and returns the full answer.
func (c *Client) Chat(ctx context.Context, model string, messages []Message, maxTokens int) (answer string, err error) {
	defer func() { c.logResponse(answer, err) }()

	resp, err := c.post(ctx, chatRequest{
		Model:     model,
		Messages:  messages,
		MaxTokens: maxTokens,
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var body chatResponse
	if err = json.NewDecoder(io.LimitReader(resp.Body, 16<<20)).Decode(&body); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if body.Error != nil {
		return "", errors.New(body.Error.Message)
	}
	if len(body.Choices) == 0 {
		return "", errors.New("API returned no choices")
	}
	return body.Choices[0].Message.Content, nil
}

// StreamChat sends a streaming request, calls onDelta for every token chunk
// and returns the complete answer. onDelta may be nil.
func (c *Client) StreamChat(ctx context.Context, model string, messages []Message, maxTokens int, onDelta func(string)) (answer string, err error) {
	defer func() { c.logResponse(answer, err) }()

	resp, err := c.post(ctx, chatRequest{
		Model:     model,
		Messages:  messages,
		Stream:    true,
		MaxTokens: maxTokens,
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var sb strings.Builder
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
			return sb.String(), nil
		}

		var chunk streamChunk
		if err = json.Unmarshal([]byte(data), &chunk); err != nil {
			return sb.String(), fmt.Errorf("decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return sb.String(), errors.New(chunk.Error.Message)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == "" {
				continue
			}
			sb.WriteString(choice.Delta.Content)
			if onDelta != nil {
				onDelta(choice.Delta.Content)
			}
		}
	}
	if err = scanner.Err(); err != nil {
		return sb.String(), fmt.Errorf("read stream: %w", err)
	}
	return sb.String(), nil
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
