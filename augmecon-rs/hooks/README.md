# Git Hooks Setup

This project includes pre-commit hooks to ensure code quality and consistency.

## Installation

To install the git hooks, run the installation script from the project root:

```bash
./install-hooks.sh
```

This will install the following hooks:
- **pre-commit**: Runs `cargo fmt --check`, `cargo check`, and `cargo clippy` before each commit

## What the hooks do

### Pre-commit Hook
The pre-commit hook ensures that:
1. **Code formatting**: All code is properly formatted with `cargo fmt`
2. **Compilation**: All code compiles without errors with `cargo check`
3. **Linting**: All code passes clippy checks without warnings

If any of these checks fail, the commit will be rejected with a helpful error message.

## Manual execution

You can also run the pre-commit checks manually at any time:

```bash
./hooks/pre-commit
```

## Bypassing hooks (not recommended)

In rare cases where you need to commit despite hook failures, you can bypass them with:

```bash
git commit --no-verify
```

However, this should only be used in exceptional circumstances.

## Updating hooks

When hooks are updated in the repository, team members can reinstall them by running:

```bash
./install-hooks.sh
```
