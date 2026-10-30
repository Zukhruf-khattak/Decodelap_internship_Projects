"""Entry point for the rule-based chatbot."""

from src.chatbot import RuleBasedChatbot


if __name__ == "__main__":
    bot = RuleBasedChatbot(name="DecodeBot")
    bot.run()