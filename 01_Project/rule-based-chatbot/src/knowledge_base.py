"""Knowledge base for the rule-based chatbot.

Keys   = sanitized user intents (lowercase, no punctuation, no outer spaces)
Values = deterministic bot responses

Exit commands are kept SEPARATE from the response dictionary because they
trigger control flow (break), not a conversational reply. Mixing them would
violate the single-responsibility principle.
"""

RESPONSES: dict[str, str] = {
    # --- Greetings ---
    "hello":        "Hi there! I'm DecodeBot. How can I help you today?",
    "hi":           "Hello! What can I do for you?",
    "hey":          "Hey! Great to see you.",
    "good morning": "Good morning! Ready to build something?",
    "good evening": "Good evening! How was your day?",

    # --- Identity / status ---
    "how are you":     "I'm a logic engine, so I'm always running at 100%. How about you?",
    "what is your name": "I'm DecodeBot, a rule-based AI built at DecodeLabs.",
    "who made you":    "I was built by an AI Engineering intern at DecodeLabs.",

    # --- Help ---
    "help": "You can greet me, ask how I am, ask my name, or type 'exit' to quit.",

    # --- Gratitude ---
    "thanks":   "You're welcome!",
    "thank you": "Anytime. That's what deterministic logic is for.",
}

# Triggers the clean break (kill command) — PDF p.11
EXIT_COMMANDS: set[str] = {"exit", "quit", "bye", "goodbye", "stop"}

EXIT_MESSAGE: str = "Goodbye! Session terminated cleanly."

# Default response for unknown intents (handled atomically by dict.get)
FALLBACK_RESPONSE: str = "I do not understand. Type 'help' to see what I can do."