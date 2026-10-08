"""Package Translation cho Backend."""

from backend.translation.base import BaseTranslator
from backend.translation.registry import TranslationModelRegistry
from backend.translation.prompts import PromptStrategy, get_prompt_strategy
from backend.translation.context import ContextManager
from backend.translation.dedup import TranslationDedupState
from backend.translation.glossary import TranslationGlossary, get_glossary
from backend.translation.pronoun_guard import enforce_no_minh
from backend.translation.romaji import romanize as romanize_name
from backend.translation.engine import (
    GGUFTranslator,
    get_translator,
    reset_translator,
    GGUFTranslationEngine,
    get_translation_engine,
    reset_translation_engine,
)

from backend.translation import lifecycle
from backend.translation import lifecycle as hotswap  # Alias tương thích ngược

# Alias tương thích ngược
TranslationContextTracker = ContextManager
TranslationDeduplicator = TranslationDedupState

__all__ = [
    "lifecycle",
    "hotswap",
    "BaseTranslator",
    "TranslationModelRegistry",
    "PromptStrategy",
    "get_prompt_strategy",
    "ContextManager",
    "TranslationContextTracker",
    "TranslationDedupState",
    "TranslationDeduplicator",
    "TranslationGlossary",
    "get_glossary",
    "enforce_no_minh",
    "romanize_name",
    "GGUFTranslator",
    "GGUFTranslationEngine",
    "get_translator",
    "get_translation_engine",
    "reset_translator",
    "reset_translation_engine",
]

