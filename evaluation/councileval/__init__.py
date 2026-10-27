"""CouncilAI evaluation harness: datasets, metrics, runners and reports.

Retrieval and cache experiments use the real components (BGE embeddings,
Chroma, cross-encoder, Redis Stack via the Go cache code). Answer-quality
runs go through the real HTTP API. Mock-model (offline) runs validate the
workflow only and never produce accuracy claims.
"""
DATASET_VERSION = "councileval-v1"
