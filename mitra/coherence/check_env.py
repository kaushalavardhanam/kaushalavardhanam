"""Check that judge-model credentials are configured."""

import os
import sys

REQUIRED = ["OPENAI_API_KEY", "MITRA_JUDGE_MODEL"]
OPTIONAL = ["OPENAI_BASE_URL", "MITRA_SPOKEN_LOG"]


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    for name in REQUIRED:
        print(f"{name}: {'set' if name not in missing else 'MISSING'}")
    for name in OPTIONAL:
        print(f"{name}: {'set' if os.environ.get(name) else 'not set (optional)'}")
    if missing:
        print("Missing required variables:", ", ".join(missing))
        return 1
    print("Judge-model configuration looks complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())