from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[0]


def get_profiles_path() -> Path:
    return (TESTS_DIR / "profiles").resolve()
