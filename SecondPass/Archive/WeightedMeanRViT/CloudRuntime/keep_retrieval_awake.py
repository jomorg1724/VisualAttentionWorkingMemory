"""Keep this Mac awake only until the authorized cloud retrieval ends."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    pod = json.loads((ROOT / 'pod.json').read_text())
    deadline = pod['hard_deadline'] + 180
    awake = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(os.getpid())])
    try:
        while time.time() < deadline:
            if (ROOT / 'cleanup_verified.json').exists() or (ROOT / 'mirror_result.json').exists():
                return
            time.sleep(5)
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == '__main__':
    main()
