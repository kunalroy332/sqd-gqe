# Contributing

Thank you for your interest in contributing to `sqd-gqe`. This project is under active
development (see the note in [README.md](README.md)); contributions, issues and feedback are welcome.

## Setup

```bash
uv venv && source .venv/bin/activate     # or: python -m venv .venv
uv pip install -e ".[dev]"
cp .env.example .env.local               # only needed for real IBM hardware runs; never commit it
```

## Running checks

```bash
pytest -q                                # unit tests (no hardware, no GPU needed)
ruff check .                             # lint (syntax errors and undefined names)
python tools/verify_headers.py           # every Python file must carry the IBM copyright header
python tools/verify_headers.py --fix     # add the header to new files
```

## Pull requests

- Keep pull requests focused on a single change.
- Add or update tests for any behaviour change.
- Make sure `pytest`, `ruff check .` and `python tools/verify_headers.py` pass before requesting review.
- New Python files need the IBM header (`tools/verify_headers.py --fix` adds it).
- Do not commit credentials, `.env*` files, cluster account names, or experiment outputs.

## Code of Conduct

This project adheres to the [Qiskit Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold this code.
