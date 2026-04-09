# Contributing

Thanks for contributing to `traceroot-py`.

## Development environment

You will need:

- Python 3.11 or newer
- `uv`
- `git`

## Fork and clone

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/traceroot-py.git
cd traceroot-py
```

3. Add the main repository as `upstream`:

```bash
git remote add upstream https://github.com/traceroot-ai/traceroot-py.git
```

## Local setup

```bash
uv sync --dev
```

To confirm the package imports correctly:

```bash
uv run python -c "import traceroot; print(traceroot.__version__)"
```

## Running locally

This repository is a Python SDK, not a standalone CLI or service. In practice, "running locally"
usually means either running the test suite or importing the SDK in a small script.

Quick smoke test:

```python
import traceroot

client = traceroot.initialize(enabled=False)
print(traceroot.__version__)
print(client.enabled)
traceroot.shutdown()
```

Run it with:

If you want to send real traces while testing locally, set `TRACEROOT_API_KEY` first.

macOS/Linux example:

```bash
export TRACEROOT_API_KEY="your_api_key"
uv run python smoke_test.py
```

PowerShell example:

```powershell
$env:TRACEROOT_API_KEY = "your_api_key"
uv run python smoke_test.py
```

## Quality checks

Run the basic checks locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

If you use pre-commit, you can also run:

```bash
uv tool run pre-commit run --all-files
```

## Best practices

- Keep changes focused and easy to review.
- Add or update tests when behavior changes.
- Update docs when public APIs or behavior change.
- Follow existing code patterns instead of introducing new styles unnecessarily.
- Let Ruff handle formatting instead of formatting by hand.
- Be mindful of backwards compatibility for public SDK behavior.

## Branch naming

Create branches from `main` and use short descriptive names:

- `feat/add-openai-agent-instrumentation`
- `fix/span-flush-timeout`
- `docs/update-contributing`
- `chore/bump-dev-dependencies`

Suggested format:

```text
<type>/<short-description>
```

Avoid committing directly to `main`.

## Conventional commits

Use Conventional Commits for commit messages and, when helpful, PR titles:

```text
feat: add google genai instrumentation
fix: handle missing git remote gracefully
docs: expand contributing guide
test: cover disabled client initialization
chore: bump development dependencies
```

Common commit types:

- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`
- `ci`
- `build`

## Pull requests

- Keep PRs focused and small enough to review quickly.
- Fill out the pull request template clearly.
- Make sure linting and tests pass locally before opening the PR.
- Add or update tests and documentation where appropriate.

## License

This project is licensed under [Apache 2.0](LICENSE).

When contributing to the TraceRoot codebase, you need to agree to the
[Contributor License Agreement](https://cla-assistant.io/traceroot-ai/traceroot-py).
You only need to do this once.
