#!/usr/bin/env bash
set -e

cd /workspace/G.O.D
source .venv/bin/activate

export NETUID=56
export NETWORK=finney
export SUBTENSOR_NETWORK=finney
export WALLET_NAME=coldwallet
export HOTKEY=hotwallet
export BT_WALLET_NAME=coldwallet
export BT_WALLET_HOTKEY=hotwallet
export BT_AXON_EXTERNAL_IP=195.142.145.66
export BT_AXON_EXTERNAL_PORT=12964

echo "ENV ustawione. Odpalam uvicorn w tle przez nohup..."
nohup uvicorn miner.server:app --host 0.0.0.0 --port 7999 --log-level info > miner.log 2>&1 &
echo "Miner wystartował z PID: $!"
