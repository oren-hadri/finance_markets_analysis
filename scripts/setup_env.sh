#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    pip install --upgrade pip
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements ..."
pip install -r requirements.txt

echo "Done."