#!/bin/bash
# Post-edit lint hook: auto-fixes formatting, reports remaining errors to Claude.
# Exit 0 = clean, Exit 2 = errors fed back to Claude for self-correction.

INPUT=$(cat)

# Guard: jq is required to parse hook input
if ! command -v jq &>/dev/null; then
  exit 0
fi

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

PROJECT_ROOT="$(pwd)"

# --- Python files: ruff ---
if [[ "$FILE_PATH" == *.py ]]; then
  RUFF="$PROJECT_ROOT/.venv/bin/ruff"
  if [ ! -x "$RUFF" ]; then
    exit 0
  fi

  # Auto-fix: format + fixable lint errors
  "$RUFF" format "$FILE_PATH" 2>/dev/null
  "$RUFF" check --fix --quiet "$FILE_PATH" 2>/dev/null

  # Check for remaining errors
  ERRORS=$("$RUFF" check --quiet "$FILE_PATH" 2>&1)
  if [ -n "$ERRORS" ]; then
    echo "$ERRORS" >&2
    exit 2
  fi
  exit 0
fi

# Other file types: skip
exit 0
