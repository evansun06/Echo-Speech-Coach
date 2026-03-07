# frontend.md — Speech Coach (SC) Frontend Agent Context

## What This Project Is

SC is a speech annotation and coaching web app. Users record or upload a presentation video, optionally attach slides/scripts/context, and receive AI-generated feedback: a timestamped interactive timeline of ML-detected events (pacing, filler words, posture, eye contact) plus a structured coach panel with collapsible notes and a chatbot for follow-up questions.

---

## Tech Stack

- **React** (component-based UI, hooks)
- **TypeScript** (strict types throughout — no `any`)
- **Material UI (MUI)** (component library — use MUI components first before building custom)
- **React Router** (client-side routing)
- **Axios or Fetch** (API calls to Django REST backend)

---

## Pages & Routes

| Route | Page | Description |
|---|---|---|
| `/login` | LoginPage | Email + password login form |
| `/signup` | SignupPage | New account creation |
| `/` | HomePage | List of prior sessions + "Start Coaching Session" CTA |
| `/sessions/:id` | DashboardPage | Full coaching dashboard for one session |

---

## Key UI Components

### HomePage
- Session list cards (session name, date, status badge)
- "Start Coaching Session" button → opens `SessionConfigModal`

### SessionConfigModal
- Step 1: Choose video source — Upload file OR Record now
- Step 2: Optional inputs — slide deck file, script file/text, free-text context
- Confirm button → creates session and navigates to `/sessions/:id`

### DashboardPage (3-panel layout)
- **Left panel**: `CoachPanel` — stage progress list + collapsible note cards + `ChatBot` input (disabled until `ready`)
- **Right panel**: `VideoPlayer` — standard playback controls, synced to timeline scrubber
- **Bottom**: `AnnotationTimeline` — layered hoverable track with timestamped ML event markers (hidden until `ml_ready`)

### CoachPanel
- Shows ordered stage list with status icons (`pending`, `processing`, `completed`, `failed`)
- Each stage has collapsible note cards with `title`, `body`, `evidence_refs`
- `default_collapsed` controls initial open/closed state

### ChatBot
- Message input disabled until session status is `ready`
- Streams response tokens via SSE
- Shows chat history

### AnnotationTimeline
- Hidden until session status is `ml_ready`
- Layered tracks (audio events, body language events)
- Each annotation has: `event_type`, `source`, `start_ms`, `end_ms`, `severity`, `summary`
- Hovering a marker shows a tooltip with `summary`
- Clicking a marker seeks the video to `start_ms`

---

## Session Lifecycle States

The backend drives all state — the frontend only reads and renders it. Poll `GET /api/v1/sessions/:id` every few seconds until terminal state.

```
draft → media_attached → queued_ml → processing_ml → ml_ready → processing_coach → ready
                                                                                  ↘ coach_failed
                                                         ↘ failed
```

| State | What to show |
|---|---|
| `draft` | "Waiting for video..." |
| `media_attached` | "Ready to analyze" |
| `queued_ml` | Loading spinner — "Analysis queued..." |
| `processing_ml` | Loading spinner — "Analyzing audio & body language..." |
| `ml_ready` | Show video + timeline. Coach panel still loading. |
| `processing_coach` | Show stage statuses animating in coach panel |
| `ready` | Full dashboard. Chat input enabled. |
| `coach_failed` | Timeline + video available. Coach panel shows error + retry. Chat disabled. |
| `failed` | Error state. Show message. Allow return to Home. |

---

## API Endpoints (What Frontend Consumes)

All endpoints are prefixed `/api/v1/`.

### Auth
```
POST /clients/signup        { email, password, name }
POST /clients/login         { email, password }
POST /clients/logout
GET  /clients/me
```

### Sessions
```
POST /sessions                          → create draft session
POST /sessions/:id/video                → attach video file (multipart)
POST /sessions/:id/assets               → attach slides/script/context (optional)
POST /sessions/:id/start-analysis       → enqueue analysis
GET  /sessions                          → list all sessions (for HomePage)
GET  /sessions/:id                      → session detail + status + coach_progress
GET  /sessions/:id/timeline             → array of Annotation objects
GET  /sessions/:id/chat-context         → assembled context (used internally by chat)
GET  /sessions/:id/video-stream         → video bytes (range requests for seeking)
```

### Chat
```
POST /sessions/:id/chat/messages        → send message
GET  /sessions/:id/chat/streams/:rid    → SSE stream (events: start, token, complete, error, heartbeat)
GET  /sessions/:id/chat/history         → past messages
```

---

## Key TypeScript Types

```ts
type SessionStatus =
  | 'draft' | 'media_attached' | 'queued_ml' | 'processing_ml'
  | 'ml_ready' | 'processing_coach' | 'ready' | 'coach_failed' | 'failed';

interface CoachingSession {
  id: string;
  status: SessionStatus;
  created_at: string;
  coach_progress: CoachProgress;
}

interface CoachProgress {
  status: 'pending' | 'processing_coach' | 'completed' | 'failed';
  current_stage: string;
  stages: CoachStage[];
}

interface CoachStage {
  stage_key: string;
  label: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  notes: CoachNote[];
}

interface CoachNote {
  note_id: string;
  title: string;
  body: string;
  evidence_refs: string[];
  default_collapsed: boolean;
}

interface Annotation {
  id: string;
  event_type: string;
  source: 'audio' | 'video';
  start_ms: number;
  end_ms: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  summary: string;
  metadata: Record<string, unknown>;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
}
```

---

## Placeholder / Mock Data Strategy

All API calls should be abstracted behind a service layer (e.g. `src/services/sessions.ts`). During development, swap real fetch calls for mock data by toggling a `USE_MOCK` flag. This means when the backend is ready, integration = flip the flag. No component logic changes needed.

```ts
// src/services/sessions.ts
const USE_MOCK = true;

export async function getSession(id: string): Promise<CoachingSession> {
  if (USE_MOCK) return MOCK_SESSION;
  const res = await fetch(`/api/v1/sessions/${id}`);
  return res.json();
}
```

---

## Frontend Conventions

- All components in `src/components/`, pages in `src/pages/`
- Use MUI `sx` prop for inline styles, `theme` for global tokens
- No `any` types — define interfaces for all API responses
- Polling: use `setInterval` inside `useEffect` to poll session status, clear on unmount
- SSE: use `EventSource` for chat streaming, close on component unmount
- Auth: store session cookie (Django handles it), redirect to `/login` on 401
- Video: use native `<video>` element with `src` pointing to `/video-stream` endpoint

---

## Critical UI Rules (from User Journey doc)

- Timeline is **hidden** until status is `ml_ready`
- Chat input is **disabled** until status is `ready`
- User can navigate to Home while analysis runs — never block navigation
- Failures must be **explicit** — never silent. Always show what failed and offer retry
- One video per session — do not allow multiple video uploads
