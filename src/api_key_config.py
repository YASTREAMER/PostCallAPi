import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_DIR / ".env"
API_KEY_VARIABLE = "POSTCALL_API_KEY"


def _read_env_file_value(path: Path, variable: str) -> str | None:
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != variable:
            continue

        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value

    return None


def load_api_key() -> str:
    """Load the single raw API key from the environment or project .env."""
    api_key = os.environ.get(API_KEY_VARIABLE)
    if api_key is None:
        api_key = _read_env_file_value(ENV_FILE, API_KEY_VARIABLE)

    if api_key is None or not api_key.strip():
        raise RuntimeError(
            f"{API_KEY_VARIABLE} is required. Set it in the environment or "
            f"in {ENV_FILE}."
        )
    if api_key != api_key.strip() or any(char.isspace() for char in api_key):
        raise RuntimeError(f"{API_KEY_VARIABLE} must not contain whitespace")
    if len(api_key) < 32:
        raise RuntimeError(
            f"{API_KEY_VARIABLE} must contain at least 32 characters"
        )
    return api_key
