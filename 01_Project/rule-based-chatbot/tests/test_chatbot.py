"""Unit tests for the rule-based chatbot."""

import pytest

from src.chatbot import RuleBasedChatbot
from src.utils import sanitize


# ---------- Sanitization ----------
def test_sanitize_lowercases():
    assert sanitize("HeLLo") == "hello"


def test_sanitize_strips_edges():
    assert sanitize("   hello   ") == "hello"


def test_sanitize_removes_punctuation():
    assert sanitize("Hello!!!") == "hello"


def test_sanitize_collapses_spaces():
    assert sanitize("how   are   you") == "how are you"


# ---------- Response logic ----------
def test_known_intent_returns_real_response():
    bot = RuleBasedChatbot()
    assert bot.get_response("hello") != bot.fallback


def test_unknown_intent_returns_fallback():
    bot = RuleBasedChatbot()
    assert bot.get_response("xyzzy nonsense") == bot.fallback


# ---------- Exit logic ----------
@pytest.mark.parametrize("cmd", ["exit", "quit", "bye", "goodbye", "stop"])
def test_exit_commands_trigger_exit(cmd):
    bot = RuleBasedChatbot()
    assert bot.should_exit(cmd) is True


def test_non_exit_command_does_not_exit():
    bot = RuleBasedChatbot()
    assert bot.should_exit("hello") is False


# ---------- Spec compliance ----------
def test_knowledge_base_has_at_least_five_intents():
    from src.knowledge_base import RESPONSES
    assert len(RESPONSES) >= 5