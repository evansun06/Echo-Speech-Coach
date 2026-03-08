## Current State
- Google Workspace-inspired global MUI theme applied (`src/theme.ts`) with shared palette, typography, shape, and component overrides
- App-wide font loading added for Google Sans + Roboto (Google Fonts in `index.html`)
- First-load splash screen added (`src/components/SplashScreen.tsx`) and wired in `App.tsx`
- Dashboard status animations added via inline `StatusIndicator` in `DashboardPage.tsx`
- `processing_coach` synthesis loading indicator added in `CoachPanelContent` above coach panel content
- Auth: complete, connected to real Django backend
- Home page: complete, session list connected to real backend
- Session creation modal: complete, connected to real backend, named sessions
- Session card navigation: complete, clicking cards opens dashboard
- Dashboard: complete — 3 panel layout, status polling, live notes feed, 
  coach review now wired to `coach_progress` stages/notes in ready state
- Chat panel: connected to real backend chat endpoints (messages/history/SSE stream)
- Older session inspection: complete
- Backend integration: sessions fully connected (SESSIONS_USE_MOCK = false)
- Record Now: complete — MediaRecorder API, camera preview, webm blob 
  upload, backend updated to accept webm
- Dashboard VideoPlayer: now uses session `video_file_url` directly
- Dashboard timeline: click-to-seek implemented on annotation markers
- Dashboard timeline: duration spans, playhead sync, and severity legend 
  implemented
- `/api/v1/sessions/:id/video-stream` is unused by frontend and can be 
  removed by backend team
- Demo AI walkthrough mode added in `src/api/index.ts` via 
  `DEMO_AI_WALKTHROUGH` constant (frontend-only lifecycle simulation)
- Demo mode video playback expects a real file at 
  `frontend/public/demo/sample.mp4`

## USE_MOCK Status
- api.auth → AUTH_USE_MOCK = false (real backend ✅)
- api.sessions → SESSIONS_USE_MOCK = false (real backend ✅)
- api.chat → CHAT_USE_MOCK = false (real backend ✅)

## Known TODOs
- duration_seconds not returned by backend — hidden in UI
- timeline endpoint is still stubbed on backend (no real annotations yet)
- RAG/PDF implementation delegated task incoming
- UI polish pass needed
- Error states for failed/coach_failed

## Next Steps
1. Backend timeline implementation (real annotation payloads)
2. RAG/PDF implementation
3. UI polish pass
4. Error states for failed/coach_failed
