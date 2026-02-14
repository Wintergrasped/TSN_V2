import re

PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z0-9]\s*){1,4})(?:[^A-Z0-9]|$)",
    re.IGNORECASE,
)

text = "Got it. KJ70X on. Not you listening."
matches = PATTERN.findall(text)
print(f"Matches: {matches}")

# The issue: (?:[A-Z0-9]\s*){1,4} matches:
# 0, X, o (from "on"), n (from "on")
# Because \s* allows the space between "X" and "on"

# We need to require the lookahead immediately after the last character
# without an optional space before it

# Better pattern: require non-letter-digit OR space+non-letter-digit after
BETTER_PATTERN = re.compile(
    r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z0-9]\s*){0,3}[A-Z0-9])(?:\s+[^A-Z0-9]|[^A-Z0-9]|$)",
    re.IGNORECASE,
)

matches2 = BETTER_PATTERN.findall(text)
print(f"Better matches: {matches2}")
