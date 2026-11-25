#!/usr/bin/env bash
# During a traced gsm8k run: one 2-s profiler window on the decode server (PORT, default 9020) once it has
# drafted 20,000 tokens.  The server writes the traces to the launcher's torch_traces/ directory.
P=${PORT:-9020}
until [[ $(curl -s -m 10 http://127.0.0.1:$P/debug/mtp_stats | python3 -c 'import json,sys;print(json.load(sys.stdin)["total_draft_tokens"]>20000)' 2>/dev/null) == True ]]; do sleep 10; done
curl -s -X POST -m 120 http://127.0.0.1:$P/start_profile; sleep 2; curl -s -X POST -m 900 http://127.0.0.1:$P/stop_profile; echo
