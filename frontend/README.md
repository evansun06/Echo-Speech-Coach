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
