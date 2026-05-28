#!/bin/bash
set -e
cd /app

# SPURS source is mounted at /app at runtime.
# Tell Python where to find the spurs/ package.
export PYTHONPATH=/app:${PYTHONPATH:-}

python predict.py
