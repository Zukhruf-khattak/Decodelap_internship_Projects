# Rule-Based AI Chatbot — DecodeLabs Project 1

A deterministic, white-box chatbot built with pure Python control flow.
No ML libraries. No API keys. Just logic.

## Project status
- [x] Continuous `while` loop
- [x] Input sanitization (case, whitespace, punctuation)
- [x] Dictionary knowledge base (10+ intents)
- [x] O(1) lookup with `.get()` and atomic fallback
- [x] Clean exit via kill command
- [x] Unit tests

## Folder structure
rule-based-chatbot/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── knowledge_base.py
│   └── utils.py
├── tests/
│   └── test_chatbot.py
└── docs/
    └── PROJECT_REPORT.md

## Run
```bash
python main.py