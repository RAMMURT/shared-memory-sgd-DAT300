#!/bin/bash

python3 -m venv .venv-viewer
source .venv-viewer/bin/activate
python -m pip install -r requirements-viewer.txt
python experiment_viewer.py
