"""
Shared utilities for SWE-SYNTH environment.
"""

import re

# Source code file extensions for git diff
# Used to filter out non-code files when extracting patches
DIFF_EXTENSIONS = (
    "'*.js' '*.ts' '*.jsx' '*.tsx' '*.py' '*.java' '*.go' "
    "'*.c' '*.cpp' '*.h' '*.rs' '*.rb' '*.php' '*.cs' "
    "'*.swift' '*.kt' '*.scala' '*.vue' '*.svelte'"
)

# Git history sanitization script
# Used to prevent cheating by removing commit history that could reveal the fix
SANITIZE_GIT_SCRIPT = """
cd /app
git config user.email "agent@swe-synth.local"
git config user.name "SWE-SYNTH Agent"
git checkout --orphan sanitized_branch
git add -A
git commit -m "Initial state"
git branch -D main 2>/dev/null || git branch -D master 2>/dev/null || true
git branch -m main
rm -rf .git/logs
rm -rf .git/refs/original
git reflog expire --expire=now --all 2>/dev/null || true
git gc --prune=now 2>/dev/null || true
echo "Git history sanitized"
"""

# Normalize all file timestamps to the same value to prevent fingerprinting via mtime
NORMALIZE_TIMESTAMPS_SCRIPT = """
cd /app
find . -not -path './.git/*' -not -path './node_modules/*' -not -path './.venv/*' -not -path './vendor/*' -exec touch -t 202001010000 {} + 2>/dev/null || true
echo "Timestamps normalized"
"""

# Regex patterns for commands that fingerprint the codebase instead of solving the task
_BLACKLISTED_PATTERNS = [
    re.compile(r'\bsha256sum\b'),
    re.compile(r'\bmd5sum\b'),
    re.compile(r'\bsha1sum\b'),
    re.compile(r'\bsha512sum\b'),
    re.compile(r'find\b.*-mmin\b'),
    re.compile(r'find\b.*-mtime\b'),
    re.compile(r'find\b.*-newer\b'),
]


def is_blacklisted_command(cmd: str) -> bool:
    """Return True if the command matches a known fingerprinting/cheating pattern."""
    for pattern in _BLACKLISTED_PATTERNS:
        if pattern.search(cmd):
            return True
    return False
