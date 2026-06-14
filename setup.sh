#!/usr/bin/env bash
# One-time setup. Run from project root:  bash setup.sh
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name .venv --display-name "Python (.venv)"
echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Open notebooks/Notebook.ipynb and select the 'Python (.venv)' kernel."
