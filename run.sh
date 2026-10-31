#!/usr/bin/env bash
# Start BioSynthetica locally: http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
# Fast start: reinstall packages only when requirements.txt changed
HASH=$(python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())")
if [ "$(cat .venv/req.sha256 2>/dev/null)" != "$HASH" ]; then
  echo "Installing packages (first run takes a few minutes)..."
  pip install -q -r requirements.txt
  pip install -q -r requirements-lab.txt || echo "The Simulator package (Tellurium) could not be installed - everything else works."
  echo "$HASH" > .venv/req.sha256
else
  echo "Packages already installed - skipping."
fi
python check_setup.py || exit 1
python server.py
