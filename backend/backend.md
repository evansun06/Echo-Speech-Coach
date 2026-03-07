# Backend Architecture (Hackathon V1)

## 1. Architecture Decision Summary

### Proposed Setup

- Keep **3 Django apps**:
    - `clients` for auth/account endpoints.
    - `sessions` for coaching session lifecycle, assets, timeline, and browser video streaming.
    - `chat` for user-facing chat endpoints and chat history ownership.
- Add a top-level **`ml/` Python package** (not a Django app) for queueable Celery workflows/stages.
- Add a top-level **`llm/` Python package** for LangGraph-based coach-memory orchestration, prompt policy, and stream shaping.
- Use Django built-in auth with session cookies + CSRF for v1 browser login/signup.
- Use a single `start-analysis` enqueue that runs phase 1 (ML) then phase 2 (LLM coach memory).

### Impact

- Fastest path for hackathon delivery without over-splitting into microservices.
- Clear separation of concerns with low migration friction post-hackathon.
- Reduces coupling by separating chat API (`chat`), ML workflow (`ml`), and coach orchestration (`llm`).

## 2. Primary Backend File Structure

```
backend/
  manage.py
  config/
    __init__.py
    asgi.py
    wsgi.py
    urls.py
    celery.py
    settings/
      __init__.py
      base.py
      local.py
      production.py    # keep for now
  apps/
    clients/
      __init__.py
      apps.py
      urls.py
      views.py
      serializers.py   # serializers are similar to pydantic contracts
      services.py
      tests/
    sessions/
      __init__.py
      apps.py
      models.py
      urls.py
      views.py
      serializers.py
      services.py
      streaming.py
      selectors.py
      tests/
    chat/
      __init__.py
      apps.py
      models.py
      urls.py
      views.py
      serializers.py
      services.py
      streaming.py
      tests/
  ml/
    __init__.py
    workflows/
      __init__.py
      analyze_session.py
    stages/
      __init__.py
      ...
  llm/
    __init__.py
    provider.py
    orchestrator.py
    coach_graph.py
    prompts.py
    streaming.py
    schemas.py
  requirements.txt
  Dockerfile
```

## 3. App Responsibilities and Ownership

- `clients` owns:
    - Signup/login/logout/current-user endpoints.
    - Authentication service logic (`authenticate`, `login`, `logout`).
    - Account-facing serializers and request validation.
- `sessions` owns:
    - `CoachingSession`, `SessionAsset`, `TranscriptSegment`, `Annotation` model ownership.
    - Session lifecycle transitions (`draft` -> `queued_ml` -> `processing_ml` -> `ml_ready` -> `processing_coach` -> `ready` / `coach_failed` / `failed`).
    - Timeline read APIs and video range-stream endpoint for browser seeking.
    - Enqueue trigger into two-phase analysis workflow.
- `ml` package owns:
    - Queueable Celery workflow composition (chains/chords/groups).
    - Phase-1 stage workers (audio, video, transcript, annotation synthesis).
    - Stage-to-session status update contracts.
- `chat` owns:
    - User-facing chat APIs (send message, stream response, fetch history).
    - Session-scoped chat message persistence in Postgres.
    - Authz checks and chat request/response DTO validation.
- `llm` package owns:
    - LangGraph coach-memory orchestration for phase 2.
    - Provider client integration and model request orchestration.
    - Prompt policy for session chat responses.
    - SSE stream token/event shaping for the chat API layer.

## 4. V1 API Surface (Locked)

- `clients`
    - `POST /api/v1/clients/signup`
    - `POST /api/v1/clients/login`
    - `POST /api/v1/clients/logout`
    - `GET /api/v1/clients/me`
- `sessions`
    - `POST /api/v1/sessions`
    - `POST /api/v1/sessions/{id}/video`
    - `POST /api/v1/sessions/{id}/assets`
    - `POST /api/v1/sessions/{id}/start-analysis`
    - `GET /api/v1/sessions` (includes session status)
    - `GET /api/v1/sessions/{id}` (includes `coach_progress`)
    - `GET /api/v1/sessions/{id}/timeline`
    - `GET /api/v1/sessions/{id}/chat-context`
    - `GET /api/v1/sessions/{id}/video-stream`
- `chat`
    - `POST /api/v1/sessions/{id}/chat/messages`
    - `GET /api/v1/sessions/{id}/chat/streams/{response_id}` (SSE)
    - `GET /api/v1/sessions/{id}/chat/history`
- `chat` SSE events (v1)
    - `start`
    - `token`
    - `complete`
    - `error`
    - `heartbeat`

## 5. Scope, Non-Goals, Risks, and Compatibility

### Scope

- Monolithic Django service with domain apps + internal ML workflow package.
- Local media storage and range streaming for hackathon v1.
- Session-scoped dashboard chat with SSE streaming and durable history.
- Two-phase lifecycle: phase 1 ML readiness for timeline, then phase 2 coach-memory generation.

### Non-Goals (V1)

- No microservice split.
- No multi-video-per-session support.
- No websocket push requirement (polling acceptable).
- No advanced retry orchestration beyond clear failure status.

### Key Risks and Mitigations

- Risk: `sessions` app grows too broad.
    - Mitigation: keep workflow orchestration in `ml/` and use selectors/services layering.
- Risk: SPA auth friction with CSRF.
    - Mitigation: standard CSRF cookie/header handling documented in frontend integration.
- Risk: browser seek/playback issues.
    - Mitigation: enforce HTTP range-response test coverage for streaming endpoint.
- Risk: SSE interruption and reconnection gaps.
    - Mitigation: response IDs + dedicated stream endpoint + terminal event coverage in tests.
- Risk: phase-2 coach failure blocks useful outputs.
    - Mitigation: introduce `coach_failed` state while keeping timeline accessible.

### Rollout and Compatibility Notes

- Start with scaffold and endpoint stubs, then incrementally implement:
    1. auth/session creation,
    2. media upload + stream,
    3. phase-1 ML workflow + timeline availability,
    4. phase-2 LangGraph coach orchestration + `coach_progress`,
    5. chat API + SSE streaming with `ready`state gating.
- Align endpoint behavior and lifecycle states with `COACHING_SESSION_USER_JOURNEY.md`.