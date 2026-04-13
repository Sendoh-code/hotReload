#!/bin/bash
# Start the GMS (Global Metadata Service) server on port 8500.
# Run this before starting vLLM workers.
#
# Usage:
#   ./start_gms.sh             # INFO logging
#   ./start_gms.sh --debug     # DEBUG logging
#   ./start_gms.sh --port 8501 # custom port

source ~/LMCache_minIO/LMCache/.venv/bin/activate
python3 ~/LMCache_minIO/gms/gms_server.py "$@"
