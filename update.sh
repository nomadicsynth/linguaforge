#!/bin/bash
source .venv/bin/activate
pip install -U ninja wheel packaging
pip install -U -r requirements.txt
pip uninstall -y flash-attn
pip install flash-attn --no-build-isolation
