"""
Debug the regex pattern
"""
import re

# Current pattern
PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*[A-Z]{1,4})(?:[^A-Z0-9]|$)",
    re.IGNORECASE,
)

test_cases = [
    "KJ70X",  # Has 0 not O - is this valid?
    "W 1 A B C D",  # Multiple spaces
    "W1ABCD",  # Normal
]

for text in test_cases:
    matches = PATTERN.findall(text)
    print(f"'{text}' -> {matches}")
    
# The issue with W 1 A B C D is that [A-Z]{1,4} expects 1-4 consecutive letters
# but "A B C D" has spaces, so it only matches "A"

# Need to allow spaces within the suffix too
IMPROVED_PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z]\s*){1,4})(?:[^A-Z0-9]|$)",
    re.IGNORECASE,
)

print("\nWith improved pattern:")
for text in test_cases:
    matches = IMPROVED_PATTERN.findall(text)
    # Clean up trailing spaces
    cleaned = [m.rstrip() for m in matches]
    print(f"'{text}' -> {cleaned}")
