#!/bin/bash

# Git hooks installation script
# This script installs the project's git hooks for all developers

set -e

HOOKS_DIR="$(pwd)/hooks"
GIT_HOOKS_DIR="$(pwd)/.git/hooks"

echo "Installing git hooks..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository root directory."
    exit 1
fi

# Check if hooks directory exists
if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: hooks directory not found at $HOOKS_DIR"
    exit 1
fi

# Install each hook (excluding README and other non-hook files)
for hook in "$HOOKS_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        # Skip README and other documentation files
        if [[ "$hook_name" == "README"* ]] || [[ "$hook_name" == *.md ]] || [[ "$hook_name" == *.txt ]]; then
            continue
        fi
        echo "Installing $hook_name hook..."
        cp "$hook" "$GIT_HOOKS_DIR/$hook_name"
        chmod +x "$GIT_HOOKS_DIR/$hook_name"
        echo "✅ $hook_name hook installed successfully!"
    fi
done

echo "🎉 All git hooks have been installed!"
echo ""
echo "The following hooks are now active:"
ls -la "$GIT_HOOKS_DIR" | grep -v sample | grep -E "^-.*x.*"
echo ""
echo "To update hooks in the future, simply run this script again."
