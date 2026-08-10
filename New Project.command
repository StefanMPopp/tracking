#!/bin/bash
# Double-click this file in Finder to open the new-project window.
# (macOS runs .command files in Terminal; the window opens from there.)

cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on PATH."
    echo "Install Python 3.11 or newer, then double-click this file again."
    read -r -p "Press Enter to close…"
    exit 1
fi

python3 new_project.py

# Only pause if something went wrong, so the window closes cleanly on success.
status=$?
if [ $status -ne 0 ]; then
    echo ""
    echo "The new-project window exited with an error (status $status)."
    read -r -p "Press Enter to close…"
fi
