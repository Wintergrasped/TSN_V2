"""
Test callsign extraction patterns
"""
import re

# Current pattern from code
CURRENT_PATTERN = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z]{1,4})\b",
    re.IGNORECASE,
)

# Test cases from real transcripts
test_cases = [
    "This is K7ABC listening",  # Normal case
    "This is K 7 ABC listening",  # Spaces in callsign
    "This is K 7ABC listening",  # Space after prefix
    "Hello, K7ABC here",  # After comma
    "KJ7OX on",  # From actual transcript
    "Got it. KJ70X on. Not you listening.",  # From actual transcript
    "Hello K7 ABC",  # Space before suffix
    "Copy, WB7UZU",  # After comma and space
]

print("Testing CURRENT pattern:")
print("=" * 60)
for text in test_cases:
    matches = CURRENT_PATTERN.findall(text)
    print(f"Text: {text}")
    print(f"  Matches: {matches}")
    print()

# Better pattern that handles spaces/commas
# This pattern allows optional spaces and punctuation around callsigns
IMPROVED_PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*[A-Z]{1,4})(?:[^A-Z0-9]|$)",
    re.IGNORECASE,
)

print("\nTesting IMPROVED pattern (allows spaces within callsign):")
print("=" * 60)
for text in test_cases:
    matches = IMPROVED_PATTERN.findall(text)
    # Normalize by removing spaces
    normalized = [m.replace(' ', '') for m in matches]
    print(f"Text: {text}")
    print(f"  Matches: {normalized}")
    print()
