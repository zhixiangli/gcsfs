#!/usr/bin/env bash
set -euo pipefail

source "$HOME/ssb_cloudbuild.env"
cd "$HOME/gcsfs"
source env/bin/activate

export PYTHONPATH="$HOME/gcsfs"

# TCP diagnostic sampler: sample ss -Htin for storage.googleapis.com (port 443) every 2s.
# Note: cwnd, retrans, delivery_rate, app_limited and busy describe this host's *sending*
# (small HTTP requests). For download diagnosis use the receive-side fields:
# bytes_received deltas, rcv_rtt, rcv_space, rcv_ssthresh, rcv_ooopack, and recvq
# (bytes received but not yet read by the application).
python -u -c '
import datetime, subprocess, time

def sample_once():
    proc = subprocess.run(
        ["ss", "-Htin", "state", "established", "( dport = :443 or dport = :https )"],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    cur_lport, cur_peer, cur_q = None, None, ""
    for line in proc.stdout.splitlines():
        if not line:
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            parts = line.split()
            if len(parts) >= 4:
                cur_lport = parts[2].rsplit(":", 1)[-1]
                cur_peer = parts[3]
                cur_q = f"recvq={parts[0]} sendq={parts[1]}"
        else:
            if cur_lport:
                print(f"tcp_sample ts={ts} lport={cur_lport} peer={cur_peer} {cur_q} {line.strip()}", flush=True)
                cur_lport = None

while True:
    sample_once()
    time.sleep(2)
' &
TCP_PID=$!
trap 'kill $TCP_PID 2>/dev/null || true' EXIT

python gcsfs/tests/perf/subsystembenchmarks/run.py \
  "--group=$GROUP" \
  "--sweep-axes=$SWEEP_AXES" \
  "--filter=$FILTER" \
  "--bucket-prefix=$BUCKET_PREFIX" \
  "--bucket-type=$BUCKET_TYPE" \
  "--project=$PROJECT_ID" \
  "--location=$REGION" \
  "--zone=$ZONE" \
  "--rapid-cache-timeout=${RAPID_CACHE_TIMEOUT:-3600}" \
  "--model-id=${MODEL_ID:-}" \
  --require-amplification
