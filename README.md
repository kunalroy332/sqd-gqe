# Python with Ruff Template Repo

This is a template for a repo that
- Defines a Python package
- Uses `ruff` for linting and formatting. Ruff is a fast all-in-one tool that replaces
  - `pylint`
  - `black`
  - `flake8`
  - `isort`
  - `...`
- Uses `towncrier` for changelog management.
- Uses `setuptools_scm` for versioning.

To use this template, do a find-all-replace for each of:
 - `Project Name`
 - `project_name`
 - `project-name`

## Installation

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management. Install uv first:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Installing the Package

You can install Project Name using uv:

```bash
uv pip install .
```

Developers should install in editable mode with the development requirements:

```bash
uv pip install -e ".[dev]"
```

Alternatively, use uv's sync command to install from the lock file:

```bash
uv sync --all-extras
```

### Virtual Environment Management

uv automatically manages virtual environments. To create and activate a virtual environment:

```bash
# Create a virtual environment (if not already created)
uv venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

You can also run commands directly without activating the environment using `uv run`:

```bash
uv run python script.py
uv run pytest
```

## Development

### Testing

Testing is done with `pytest`, with tests located in the `test` folder:

```bash
uv run pytest
```

Or run pytest directly if you've activated a virtual environment:

```bash
pytest
```

### Linting and Formatting

We use ruff for linting and formatting. Run the following to manually apply checks:

```bash
uv run ruff check .
```

Or if you have ruff installed globally or in an activated environment:

```bash
ruff check .
```

More conveniently, set up ruff in your IDE to run-on-save.

### Releasing a version

Versions are managed completely with git tags, no files nede to be written to.
To create a new version pointing at your current state of `main`,

```bash
git checkout main  # make sure you're on head
git tag -a 1.2.3   # make an annotated tag 1.2.3
git push --tags    # push the tags to origin
```

Note that the second step uses `-a` which causes the tag to be annotated with some text. This text will appear on the github tag/release pages. Alternatively, github has a GUI for adding annotated tags which may be more convenient because you can see markdown rendered. This GUI can be used instead of the three commands above.

### Managing the Changelog with Towncrier

We use [Towncrier](https://towncrier.readthedocs.io/) for changelog management.
For a pull request with number `42`,

1. Create a [new fragment](https://towncrier.readthedocs.io/en/stable/tutorial.html#creating-news-fragments) file in the `changelog.d/` directory:

```bash
echo "Added a new particle weighting system." > changelog.d/42.feature.rst
```

2. Before releasing a new version, generate the changelog:

```bash
uv run towncrier build --yes
```

### Pre-commit Hooks

We use `pre-commit` to ensure code quality and consistency before committing changes.
It automatically runs linters and formatters on staged files during `git commit`.
This will have been installed as a part of the developer requirements.


1. Install the hooks defined in the .pre-commit-config.yaml file:

```bash
pre-commit install
```

2. Test the hooks on all files (optional):

```bash
pre-commit run --all-files
```

3. To update the pre-commit hooks to their latest versions, run:

```bash
pre-commit autoupdate
```

### VSCode Setup

Here are my recommended settings for `.vscode/settings.json` for this project.
They cause ruff to format and fix on save, they setup the testing tab, and they disable
other linters and formatters from running because ruff does it all.

```json
{
    "python.testing.pytestArgs": [
        "test"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```
