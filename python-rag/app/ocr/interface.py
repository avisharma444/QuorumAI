"""Pluggable OCR interface — every extraction backend implements this (reference)."""
from abc import ABC, abstractmethod

from app.models import OCRResult


class OCRBackend(ABC):
    @abstractmethod
    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extract structured blocks from a document."""

    @abstractmethod
    def name(self) -> str:
        """Backend identifier recorded as ocr_method."""


class ExtractionError(RuntimeError):
    """A backend could not run (missing engine, failed conversion)."""
