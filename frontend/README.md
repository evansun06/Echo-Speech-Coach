# Frontend

Minimal React + TypeScript + Vite frontend scaffold for Speech Coach.

## Run with Docker

From the repository root:

```bash
docker compose up frontend
```

Then open:

`http://localhost:5173`

The compose setup mounts the entire local `frontend/` directory into the
container, so local file edits hot-reload in the browser.

## Stop

```bash
docker compose down
```

## Local (without Docker)

```bash
npm install
npm run dev
```

## Runtime Flags

The API layer defaults to real backend mode. Optional Vite flags:

```bash
VITE_DEMO_AI_WALKTHROUGH=false
VITE_SESSIONS_USE_MOCK=false
VITE_CHAT_USE_MOCK=false
```
