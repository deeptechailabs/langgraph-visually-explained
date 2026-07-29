"""Run every published course example and verify the intentional failure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_FAILURE = Path(
    "chapters/02-state-and-reducers/code/06_parallel_crash.py"
)


def example_files() -> list[Path]:
    """Return published example files in chapter and filename order."""
    return sorted((ROOT / "chapters").glob("*/code/*.py"))


def main() -> int:
    failures: list[str] = []

    for path in example_files():
        relative = path.relative_to(ROOT)
        print(f"\n{'=' * 72}\nRUN {relative}\n{'=' * 72}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout.rstrip())

        if relative == EXPECTED_FAILURE:
            combined = f"{result.stdout}\n{result.stderr}"
            passed = (
                result.returncode != 0
                and "InvalidUpdateError" in combined
            )
            if passed:
                print("PASS: raised the expected InvalidUpdateError")
            else:
                failures.append(
                    f"{relative}: expected InvalidUpdateError"
                )
            continue

        if result.returncode == 0:
            print("PASS")
        else:
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            failures.append(
                f"{relative}: exited with {result.returncode}"
            )

    print(f"\n{'=' * 72}")
    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"ALL {len(example_files())} EXAMPLES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

