## Current State
- Auth: complete, connected to real Django backend
- Home page: complete, session list connected to real backend
- Session creation modal: complete, connected to real backend, named sessions
- Session card navigation: complete, clicking cards opens dashboard
- Dashboard: complete — 3 panel layout, status polling, live notes feed, 
  final coach review placeholder (pending coach_progress from backend)
- Chat panel: built with mock SSE reasoning stream (pending backend)
- Older session inspection: complete
- Backend integration: sessions fully connected (SESSIONS_USE_MOCK = false)
- Record Now: complete — MediaRecorder API, camera preview, webm blob 
  upload, backend updated to accept webm

## USE_MOCK Status
- api.auth → AUTH_USE_MOCK = false (real backend ✅)
- api.sessions → SESSIONS_USE_MOCK = false (real backend ✅)
- api.chat → USE_MOCK = true (pending backend)

## Known TODOs
- coach_progress not yet returned by backend — dashboard shows placeholder 
  with TODO comment, wire when backend adds it
- duration_seconds not returned by backend — hidden in UI
- Chat SSE not yet connected to real backend
- RAG/PDF implementation delegated task incoming
- UI polish pass needed
- Error states for failed/coach_failed

## Next Steps
1. Connect chat to real backend when ready
2. Wire coach_progress when backend adds it
3. RAG/PDF implementation
4. UI polish pass
5. Error states for failed/coach_failed