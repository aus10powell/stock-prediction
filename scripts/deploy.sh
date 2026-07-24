#!/usr/bin/env bash
# Non-interactive production deploy for local machines, CI, and remote agents.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .vercel/project.json ]]; then
  echo "Missing .vercel/project.json — repo is not linked to a Vercel project." >&2
  exit 1
fi

TOKEN_ARGS=()
if [[ -n "${VERCEL_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--token "$VERCEL_TOKEN")
fi

if ! npx vercel whoami "${TOKEN_ARGS[@]}" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Not authenticated with Vercel.

Remote / cloud agent options (pick one):
  1) Push to main — GitHub↔Vercel auto-deploys (no token needed on the agent)
  2) Export VERCEL_TOKEN from https://vercel.com/account/tokens and rerun
  3) Run: npx vercel login   (device OAuth)

EOF
  exit 1
fi

echo "Deploying stock-prediction to Vercel production..."
npx vercel deploy --prod --yes "${TOKEN_ARGS[@]}"
