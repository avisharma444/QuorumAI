"""OpenTelemetry tracing for the RAG service.

An intentional extension: the reference Python service had no tracing.
The Go backend injects a W3C ``traceparent`` into every call; FastAPI
instrumentation extracts it, so Python spans join the Go request's trace.

Configured by environment (OpenTelemetry names):

* ``OTEL_TRACES_EXPORTER``: ``none`` (default), ``console`` or ``otlp``.
* ``OTEL_EXPORTER_OTLP_ENDPOINT``: OTLP/HTTP base URL, e.g. ``http://jaeger:4318``.
* ``OTEL_TRACES_SAMPLER_ARG``: ratio for new root traces (sampled parents are followed).

With ``none`` no SDK provider is installed and no span is recorded.
Export runs in a background batch processor: a missing collector logs
export errors but never fails a request. Spans carry sizes, counts and
timings only, never document text, questions or embeddings, and outgoing
web-search requests are not instrumented (no trace headers reach
third-party sites).
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)
SERVICE_NAME = "councilai-python-rag"
_tracer = trace.get_tracer(SERVICE_NAME)


def setup_tracing(app, *, exporter: str | None = None, span_exporter=None):
    """Install the provider and instrument the FastAPI app. Returns the
    provider (or None when tracing is off). ``span_exporter`` lets tests
    pass an in-memory exporter (exported synchronously)."""
    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()]))
    kind = (exporter or os.getenv("OTEL_TRACES_EXPORTER", "none")).strip().lower()
    if span_exporter is None and kind in ("", "none", "noop"):
        logger.info("Tracing off (OTEL_TRACES_EXPORTER=%s)", kind or "none")
        return None

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    try:
        ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    except ValueError:
        ratio = 1.0
    ratio = ratio if 0 < ratio <= 1 else 1.0
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME, "service.version": "1.0.0"}),
        sampler=ParentBased(TraceIdRatioBased(ratio)),
    )
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif kind == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif kind == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
        if not base:
            logger.warning("OTEL_TRACES_EXPORTER=otlp without OTEL_EXPORTER_OTLP_ENDPOINT; tracing off")
            return None
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces", timeout=5)))
    else:
        logger.warning("Unknown OTEL_TRACES_EXPORTER=%r; tracing off", kind)
        return None
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="health,ready", exclude_spans=["receive", "send"]
    )
    logger.info("Tracing on: exporter=%s, sampling %.2f of new traces", kind if span_exporter is None else "test", ratio)
    return provider


@contextmanager
def span(name: str, **attributes):
    """A child span of the current request (no-op when tracing is off).
    Only pass counts, sizes, timings and fixed labels as attributes."""
    with _tracer.start_as_current_span(name, attributes=attributes) as s:
        yield s
