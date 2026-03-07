# Speech Coach

## 1. Summary
Speech Coach is an AI-powered communication coaching platform that analyzes recorded presentations and delivers timestamped, actionable feedback on delivery quality, confidence, and clarity.

The system is built for hackathon-speed execution with production-minded architecture: asynchronous ML processing, explicit lifecycle states, and coach-memory synthesis that turns raw multimodal signals into practical coaching guidance.

## 2. Features
- Single-session coaching flow: create session, attach one video, and run analysis.
- Optional context enrichment: slides, script, and free-text speaking context.
- Timeline-first feedback: timestamped events for audio and body-language signals.
- Two-phase analysis pipeline:
  - Phase 1 (`ml`): transcript/audio/visual artifacts become timeline-ready.
  - Phase 2 (`llm`): coach-memory orchestration produces stage notes and final coaching output.
- Session-aware coaching chat: enabled once session reaches `ready`.
- Failure-tolerant UX: timeline remains available when coach phase fails (`coach_failed`).

## 3. Tech Stack
### Frontend
- React + MUI + TypeScript

### Backend
- Django + Django REST Framework + Python 3.13

### Async + Data + Infra
- Redis (broker/cache)
- Celery (async orchestration)
- PostgreSQL (system of record)
- Docker (containerized runtime)

### Powered by Google Cloud
- Gemini model APIs for coaching intelligence (lite and frontier-grade model paths based on task complexity).
- Google Cloud speech services for transcription and voice-oriented coaching flows.
- Current architecture uses Google Cloud primarily for AI-service integration while core runtime remains container-agnostic.

## 4. Backend Architecture
This section is derived from the project design docs, especially:
- `BACKEND_ARCHITECTURE.md`
- `AI_ORCHESTRATION.md`
- `COACHING_SESSION_USER_JOURNEY.md`

### 4.1 Architecture Model
- Monolithic Django service with bounded domain apps plus internal orchestration packages.
- Domain apps:
  - `clients`: auth/account endpoints and account access logic.
  - `sessions`: coaching session lifecycle, media/assets, timeline ownership.
  - `chat`: session-scoped chat APIs and durable message history.
- Internal packages:
  - `ml/`: Celery workflow composition and phase-1 analysis stages.
  - `llm/`: coach-memory orchestration, provider integration, and response shaping.

### 4.2 Two-Phase Orchestration
- Single enqueue entrypoint: `POST /sessions/{id}/start-analysis`.
- Phase 1 (`ml`): complete transcript/audio/visual analysis and persist timeline artifacts (`TranscriptSegment`, `Annotation`).
- Phase 2 (`llm`): run coach-memory loop over semantic chunks and emit stage notes + final coaching output.
- Benefit: timeline can appear at `ml_ready` while coach synthesis continues, reducing perceived latency.

### 4.3 Session Lifecycle (Server-Authoritative)
Canonical session states:
- `draft`
- `media_attached`
- `queued_ml`
- `processing_ml`
- `ml_ready`
- `processing_coach`
- `ready`
- `coach_failed`
- `failed`

Key behavior:
- Timeline is gated until `ml_ready`.
- Chat input is gated until `ready`.
- If phase 2 fails, session may transition to `coach_failed` while keeping timeline review available.

### 4.4 API Surface (V1 Capability-Level)
- Auth (`clients`): signup, login, logout, current user.
- Session lifecycle (`sessions`): create session, attach video/assets, start analysis, list/get sessions.
- Review endpoints (`sessions`): timeline, chat-context, video streaming.
- Chat (`chat`): send messages, read history, stream responses.

Representative v1 routes from design docs:
- `POST /api/v1/sessions/{id}/start-analysis`
- `GET /api/v1/sessions/{id}` (includes `coach_progress`)
- `GET /api/v1/sessions/{id}/timeline`
- `POST /api/v1/sessions/{id}/chat/messages`

### 4.5 Data Contracts and Ownership
Core entities:
- `CoachingSession`: lifecycle authority for one coaching attempt.
- `SessionAsset`: optional slides/script/context payloads.
- `TranscriptSegment`: timestamped transcript chunks.
- `Annotation`: normalized timeline events (`event_type`, timing, severity/confidence, metadata).
- `CoachProgress`: phase-2 stage status and note cards.
- `ChatMessage`: session-scoped durable conversation history.

Contract constraints (hackathon v1):
- One video per session.
- Optional assets must not block analysis.
- Annotation metadata evolves via versioned/type-specific schema handling.

### 4.6 Reliability, Failure Modes, and Operations
- Async isolation keeps API responsive during heavy ML/LLM workloads.
- Explicit terminal states prevent silent failure behavior.
- Partial-value fallback preserves timeline artifacts when coach phase fails.
- Recommended observability baseline:
  - Correlated logs via request/session IDs.
  - Queue depth, task latency, phase duration, and failure-rate metrics.
  - Stage-based triage using lifecycle states.

### 4.7 Hackathon Constraints and Non-Goals
In scope:
- Single video per session.
- Happy-path-first reliability for demo.
- Polling-compatible progress model.

Deferred:
- Multi-video comparison sessions.
- Complex partial-recovery orchestration.
- Full real-time push infrastructure as a hard requirement.
- Advanced feedback taxonomy personalization.
