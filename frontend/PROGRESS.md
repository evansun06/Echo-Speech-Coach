## Current State
- Auth: complete, connected to real Django backend
- Home page: complete, session list with mock data
- Session creation modal: complete
- Dashboard: in progress — coach panel redesign just implemented
- Chat panel: built with mock SSE reasoning stream
- Older session inspection: complete

## USE_MOCK Status
- api.auth → USE_MOCK = false (real backend)
- api.sessions → USE_MOCK = true (pending backend)
- api.chat → USE_MOCK = true (pending backend)

## Next Steps
- Real API swap for sessions when backend ready
- UI polish pass
- Error states for failed/coach_failed