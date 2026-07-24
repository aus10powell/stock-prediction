<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# Deploy (remote / cloud agents)

Production app: https://workspace-lemon-two.vercel.app  
Vercel project: `aus10powells-projects/stock-prediction` (see `.vercel/project.json`)

## Preferred: push to `main`

This GitHub repo is connected to Vercel. From any remote server:

```bash
git push origin main
```

Vercel builds and deploys production automatically. No CLI login needed on the agent.

## Explicit CLI deploy (optional)

Needs auth via `VERCEL_TOKEN` (create at https://vercel.com/account/tokens):

```bash
export VERCEL_TOKEN=...   # or set in the cloud environment secrets
npm run deploy
```

`.vercel/project.json` is committed so `vercel --prod --yes` is non-interactive once authenticated.
