.PHONY: test run benchmark

test:
	cd go-backend && go test -race ./...
	cd python-rag && python -m pytest -q tests

run:
	docker compose up --build

benchmark:
	python benchmarks/benchmark_http.py
