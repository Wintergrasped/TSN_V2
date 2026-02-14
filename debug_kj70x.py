import re

text = "Got it. KJ70X on. Not you listening."

# The problem: we want to match "KJ70X" but not "on"
# Let's test different patterns

patterns = [
    ("Current", r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z0-9]\s*){0,3}[A-Z0-9])(?=\s|[^A-Z0-9]|$)"),
    ("No spaces in suffix", r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*[A-Z0-9]{1,4})(?=\s|[^A-Z0-9]|$)"),
    ("With alternation", r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z0-9]{1,4}|(?:[A-Z0-9]\s*){0,3}[A-Z0-9]))(?=\s|[^A-Z0-9]|$)"),
]

for name, pattern_str in patterns:
    pattern = re.compile(pattern_str, re.IGNORECASE)
    matches = pattern.findall(text)
    cleaned = [m.replace(' ', '') for m in matches]
    print(f"{name:20s}: {cleaned}")

# The issue is that "(?:[A-Z0-9]\s*){0,3}[A-Z0-9]" matches:
# 0 (with space after)
# X (with space after)  
# o (with space after? no, but it continues)
# n

# Actually I think the issue is simpler - let me trace through:
print("\nDetailed trace:")
pattern = re.compile(r"(?:^|[^A-Z0-9])([A-Z]{1,2}\s*\d\s*(?:[A-Z0-9]\s*){0,3}[A-Z0-9])(?=\s|[^A-Z0-9]|$)", re.IGNORECASE)
import regex
for match in pattern.finditer(text):
    print(f"Match: '{match.group(1)}' at position {match.start()}-{match.end()}")
    print(f"Preceding: '{text[max(0, match.start()-5):match.start()]}'")
    print(f"Following: '{text[match.end():min(len(text), match.end()+5)]}'")
