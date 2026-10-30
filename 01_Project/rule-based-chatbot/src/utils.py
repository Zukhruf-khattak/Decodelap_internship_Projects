"""Input sanitization helpers.

Implements Phase 1 of the IPO model (PDF p.10):
raw input -> normalization -> cleaned input.
"""

import string


def sanitize(text: str) -> str:
    """Normalize user input.

    Steps:
        1. Lowercase          -> 'HeLLo'    becomes 'hello'
        2. Strip edges        -> '  hello ' becomes 'hello'
        3. Remove punctuation -> 'hello!!!' becomes 'hello'
        4. Collapse spaces    -> 'how  are you' becomes 'how are you'
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())