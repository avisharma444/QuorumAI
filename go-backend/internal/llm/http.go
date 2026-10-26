package llm

import (
	"errors"
	"fmt"
	"io"
	"net/http"
)

const (
	// maxResponseBytes caps how much of a provider response is read.
	maxResponseBytes = 8 << 20
	// maxErrorBodyBytes caps how much of an error body is copied into errors
	// and logs.
	maxErrorBodyBytes = 512
)

// APIError is a non-200 HTTP response from a provider.
type APIError struct {
	Provider   string
	StatusCode int
	Body       string // truncated response body
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s API error (status %d): %s", e.Provider, e.StatusCode, e.Body)
}

// postJSON sends req and returns the body of a 200 response. Transport
// errors wrap the underlying error, so errors.Is(err, context.Canceled) and
// net.Error timeouts still work for callers. Request URLs must never contain
// credentials, because transport errors include the URL.
func postJSON(client *http.Client, req *http.Request, provider string) ([]byte, error) {
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s request failed: %w", provider, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, fmt.Errorf("%s: failed to read response: %w", provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, &APIError{Provider: provider, StatusCode: resp.StatusCode, Body: truncate(string(body), maxErrorBodyBytes)}
	}
	return body, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

var errNoMessages = errors.New("at least one message is required")
