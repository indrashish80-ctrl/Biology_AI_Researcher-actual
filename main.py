"""
main.py
-------
Starting point for anyone who opens this folder and runs the file called "main".

It used to be a leftover 25-line Gemini connection test that raised
"GEMINI_API_KEY was not found. Check your .env file." the moment it ran - which made a working
program look broken, and contradicted the rule that keys are entered in the web page and never
stored in a file.

It now starts the real application, exactly as run.bat / run.sh do.

    python main.py          ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import runpy
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000"


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    print("BioSynthetica - starting the web application.")
    print(f"When it is ready, open:  {URL}")
    print("Enter your own API keys on the 'API keys' page. Nothing is read from a .env file.")
    print("Press Ctrl+C to stop.\n")
    if os.environ.get("BIOSYN_NO_BROWSER") != "1":
        try:
            webbrowser.open(URL)
        except Exception:
            pass                      # no desktop session: the URL is printed above
    try:
        runpy.run_path(str(ROOT / "server.py"), run_name="__main__")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
