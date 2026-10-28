# Everything runs in Docker; no local Go or Python toolchain is needed.
# Integration tests that need the running stack are documented in docs/development.md.

GO_IMAGE  := golang:1.23
RAG_TEST  := councilai-rag-test

.PHONY: help up run down test test-go test-rag test-eval rag-test-image smoke benchmark

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

up: ## Build and start the stack (mock models by default)
	docker compose up --build

run: up

down: ## Stop the stack (keeps volumes)
	docker compose down

test: test-go test-rag test-eval ## Run all unit test suites

test-go: ## Go: gofmt, vet and race tests (integration tests skip without services)
	docker run --rm -v councilai-gomod:/go/pkg/mod -v "$(CURDIR)":/src -w /src/go-backend $(GO_IMAGE) \
	  sh -c 'test -z "$$(gofmt -l .)" && go vet ./... && go test -race -count=1 ./...'

rag-test-image:
	docker build -q --target test -t $(RAG_TEST) python-rag

test-rag: rag-test-image ## Python retrieval service tests (real OCR and models)
	docker run --rm -v "$(CURDIR)/python-rag":/srv -v councilai-hf:/models $(RAG_TEST)

test-eval: rag-test-image ## Evaluation harness tests
	docker run --rm -v "$(CURDIR)":/repo -w /repo/evaluation $(RAG_TEST) \
	  python -m pytest -q -p no:cacheprovider tests

smoke: ## End-to-end smoke test against the running stack
	python3 tests/e2e_smoke.py

benchmark: ## HTTP benchmark against the running stack
	python3 benchmarks/benchmark_http.py
