package llm

import (
	"context"
	"errors"
	"time"

	"go.opentelemetry.io/otel/attribute"

	"example.com/councilai-starter/internal/metrics"
	"example.com/councilai-starter/internal/telemetry"
)

// Call sends one model request and records it: a span "llm.<stage>" (stage,
// provider, outcome, token counts; never the prompt or the answer), the
// request-duration histogram, the failure counter, and provider-reported
// token counts. Every model request in the backend goes through here, so
// each one is counted exactly once.
func Call(ctx context.Context, stage string, client LLMClient, opts GenerateOptions) (*Response, error) {
	provider := metrics.Provider(client.ModelName())
	ctx, span := telemetry.Start(ctx, "llm."+stage,
		attribute.String("llm.stage", stage), attribute.String("llm.provider", provider))
	defer span.End()
	start := time.Now()
	resp, err := client.GenerateChat(ctx, opts)
	elapsed := time.Since(start).Seconds()
	// A timeout (stage or request deadline) is a model failure; a caller
	// cancellation (client went away) is not.
	result := "success"
	switch {
	case err != nil && errors.Is(ctx.Err(), context.DeadlineExceeded):
		result = "timeout"
	case err != nil && ctx.Err() != nil:
		result = "cancelled"
	case err != nil || resp == nil:
		result = "error"
	}
	metrics.LLMRequestDuration.WithLabelValues(stage, provider, result).Observe(elapsed)
	span.SetAttributes(attribute.String("llm.result", result))
	if result != "success" {
		if result != "cancelled" {
			metrics.LLMFailures.WithLabelValues(stage, provider).Inc()
		}
		telemetry.Fail(span, "llm_"+result)
		return resp, err
	}
	if resp.Usage.Reported() {
		metrics.LLMTokens.WithLabelValues(stage, provider, "prompt").Add(float64(resp.Usage.PromptTokens))
		metrics.LLMTokens.WithLabelValues(stage, provider, "completion").Add(float64(resp.Usage.CompletionTokens))
		span.SetAttributes(attribute.Int("llm.usage.prompt_tokens", resp.Usage.PromptTokens),
			attribute.Int("llm.usage.completion_tokens", resp.Usage.CompletionTokens))
	}
	return resp, nil
}
