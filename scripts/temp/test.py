from __future__ import annotations

import sys
import subprocess

def main():
    try:
        res = subprocess.run(
            [sys.executable, "-u", "scripts/user_profile.py"],
            check=True,
            # text=True,
            # capture_output=True
        )
        # print(res.stdout)
        # print(res.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"user_profile.py failed with exit code {e.returncode}")
        raise
    except FileNotFoundError:
        print(f"Python executable or script not found")
        raise


if __name__ == "__main__":
    main()