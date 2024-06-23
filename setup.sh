#!/bin/bash
source .venv/bin/activate
pip install ninja wheel packaging
pip install torch
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
wget https://raw.githubusercontent.com/ironjr/grokfast/main/grokfast.py
