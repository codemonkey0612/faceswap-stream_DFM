#!/usr/bin/env bash
# Debug helper: create (and optionally close) an Akool live face swap session
# from the command line, without the browser UI. Useful for verifying creds /
# source image before going live.
#
# Usage:
#   CLIENT_ID=... API_KEY=... SOURCE_URL=... bash akool_session.sh
#   CLIENT_ID=... API_KEY=... CLOSE_ID=<session_id> bash akool_session.sh   # close one
set -euo pipefail

AKOOL="https://openapi.akool.com"
: "${CLIENT_ID:?set CLIENT_ID}"
: "${API_KEY:?set API_KEY}"

echo "== getToken =="
TOKEN=$(curl -s -X POST "$AKOOL/api/open/v3/getToken" \
  -H "Content-Type: application/json" \
  -d "{\"clientId\":\"$CLIENT_ID\",\"clientSecret\":\"$API_KEY\"}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
echo "token ok"

if [ -n "${CLOSE_ID:-}" ]; then
  echo "== close session $CLOSE_ID =="
  curl -s -X POST "$AKOOL/api/open/v3/faceswap/live/close" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d "{\"_id\":\"$CLOSE_ID\"}"
  echo; exit 0
fi

: "${SOURCE_URL:?set SOURCE_URL}"

echo "== detect face =="
OPTS=$(curl -s -X POST "$AKOOL/interface/detect-api/detect_faces" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"url\":\"$SOURCE_URL\",\"single_face\":true,\"return_face_url\":true}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['faces_obj']['0']['landmarks_str'][0])")
echo "opts: $OPTS"

echo "== create session =="
curl -s -X POST "$AKOOL/api/open/v3/faceswap/live/create" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"sourceImage\":[{\"path\":\"$SOURCE_URL\",\"opts\":\"$OPTS\"}]}" \
  | python3 -m json.tool
echo
echo "NOTE: this session is now LIVE and billing. Close it with:"
echo "  CLIENT_ID=$CLIENT_ID API_KEY=*** CLOSE_ID=<_id from above> bash akool_session.sh"
