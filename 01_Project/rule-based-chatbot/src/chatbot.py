"""Core rule-based chatbot engine (IPO model).

    INPUT   -> sanitize()
    PROCESS -> get_response()   (O(1) hash-map lookup)
    OUTPUT  -> print()
"""

from src.knowledge_base import (
    RESPONSES,
    EXIT_COMMANDS,
    EXIT_MESSAGE,
    FALLBACK_RESPONSE,
)
from src.utils import sanitize


class RuleBasedChatbot:
    """A deterministic, white-box chatbot."""

    def __init__(self, name: str = "DecodeBot") -> None:  #its a constructor method that initializes the chatbot with a name and sets up its responses, exit commands, fallback response, and running state. its also called special methods or dunder methods because they have double underscores before and after the method name.
        self.name = name
        self.responses: dict[str, str] = RESPONSES
        self.exit_commands: set[str] = EXIT_COMMANDS
        self.fallback: str = FALLBACK_RESPONSE
        self.is_running: bool = True

    # ---------- Phase 1: INPUT ----------
    def sanitize_input(self, raw_input: str) -> str:
        """Normalize raw text before any lookup."""
        return sanitize(raw_input)

    # ---------- Phase 2: PROCESS ----------
    def get_response(self, clean_input: str) -> str:
        """O(1) lookup with atomic fallback."""
        return self.responses.get(clean_input, self.fallback)

    def should_exit(self, clean_input: str) -> bool:
        """Kill-command check (separate from conversational logic)."""
        return clean_input in self.exit_commands

    # ---------- Phase 3: OUTPUT (loop) ----------
    def run(self) -> None:
        """The infinite heartbeat loop until the kill command."""
        print(
            f"{self.name}: Hello! I'm a rule-based AI. "
            f"Type 'help' for options, or 'exit' to quit.\n"
        )

        while self.is_running:
            # --- input guard: never let Ctrl+C crash the bot ---
            try:
                raw_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{self.name}: Session interrupted. Goodbye!")
                break

            clean_input = self.sanitize_input(raw_input)

            # --- ignore empty / whitespace-only input ---
            if not clean_input:
                continue

            # --- exit strategy ---
            if self.should_exit(clean_input):
                print(f"{self.name}: {EXIT_MESSAGE}")
                self.is_running = False
                break

            # --- normal response ---
            reply = self.get_response(clean_input)
            print(f"{self.name}: {reply}")