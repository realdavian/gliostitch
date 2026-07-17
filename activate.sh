#!/usr/bin/env bash
# Activate the project venv (created by `uv venv` + `uv sync`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
