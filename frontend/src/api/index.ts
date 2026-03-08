// Hackathon hard-switch: force real backend paths, ignore Vite mock/demo flags.
export const CHAT_USE_MOCK = false
export const SESSIONS_USE_MOCK = false
export const DEMO_AI_WALKTHROUGH = false
const AUTH_USE_MOCK = false
export const API_BASE_URL = 'http://localhost:8000'

export interface AuthUser {
  id: string
  name: string
  email: string
}

export type LoginResponse = AuthUser
export type SignupResponse = AuthUser
export type CurrentUserResponse = AuthUser

export type SessionStatus =
  | 'draft'
  | 'media_attached'
  | 'queued_ml'
  | 'processing_ml'
  | 'ml_ready'
  | 'processing_coach'
  | 'ready'
  | 'coach_failed'
  | 'failed'

export type CoachProgressStatus = 'pending' | 'processing_coach' | 'completed' | 'failed'

export type CoachStageStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type CoachNoteKind = 'window_impression' | 'event_note' | 'general'

export interface CoachNote {
  note_id: string
  title: string
  body: string
  evidence_refs: string[]
  default_collapsed: boolean
  note_kind: CoachNoteKind
  event_id: string | null
  event_type: string | null
  sequence: number | null
}

export interface CoachStage {
  stage_key: string
  label: string
  status: CoachStageStatus
  notes: CoachNote[]
}

export interface CoachAgentProgress {
  agent_execution_id: string
  execution_index: number
  agent_kind: 'subagent' | 'flagship_periodic' | 'flagship_final'
  agent_name: string
  status: CoachStageStatus
  window_start_ms: number | null
  window_end_ms: number | null
  input_seq_from: number | null
  input_seq_to: number | null
  output_seq_to: number | null
  started_at: string | null
  completed_at: string | null
  last_heartbeat_at: string | null
  model_name: string
  window_label: string
  window_impression: CoachNote | null
  reasoning_events: CoachNote[]
}

export interface CoachLedgerEntry {
  sequence: number
  entry_kind: string
  agent_kind: 'subagent' | 'flagship_periodic' | 'flagship_final' | null
  agent_name: string
  window_start_ms: number | null
  window_end_ms: number | null
  content: string
  payload: Record<string, unknown>
  created_at: string
}

export interface CoachFinalReconciliation {
  agent_name: string
  model_name: string
  overall_impression: string
  strengths: string[]
  improvements: string[]
  priority_actions: string[]
  created_at: string
}

export interface CoachProgress {
  status: CoachProgressStatus
  current_stage: string
  active_run_id?: string | null
  run_index?: number | null
  latest_ledger_sequence?: number
  updated_at?: string
  agent_progress?: CoachAgentProgress[]
  stages: CoachStage[]
  ledger_entries?: CoachLedgerEntry[]
  final_reconciliation?: CoachFinalReconciliation | null
}

export interface Annotation {
  id: string
  event_type: string
  source: 'audio' | 'video'
  start_ms: number
  end_ms: number
  severity: 'low' | 'medium' | 'high'
  confidence: number
  summary: string
  metadata: Record<string, unknown>
}

export interface CoachingSessionListItem {
  id: string
  title: string
  status: SessionStatus
  created_at: string
  updated_at: string
}

export interface CoachingSessionDetail {
  id: string
  title: string
  status: SessionStatus
  created_at: string
  updated_at: string
  video_file_url: string | null
  supplementary_pdf_1_url: string | null
  supplementary_pdf_2_url: string | null
  supplementary_pdf_3_url: string | null
  speaker_context: string
  coach_progress: CoachProgress | null
}

export interface SessionCreateResponse {
  id: string
  title: string
  status: SessionStatus
  created_at: string
}

export interface SessionAssetsPayload {
  pdfFiles?: File[] | null
  speakerContext?: string
}

export type ChatRole = 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  created_at: string
}

export interface ChatSendResponse {
  response_id: string
}

export interface ChatStreamCallbacks {
  onReasoningToken?: (token: string) => void
  onAnswerToken?: (token: string) => void
  onComplete?: () => void
  onError?: (message: string) => void
}

export interface ApiError {
  message: string
  status?: number
}

interface MockAuthData {
  login_success: LoginResponse
  signup_success: SignupResponse
  current_user: CurrentUserResponse
  errors: {
    invalid_credentials: string
    email_already_exists: string
  }
}

interface MockSessionsData {
  sessions: CoachingSessionListItem[]
  create_session_response?: {
    id: string
    title: string
  }
  session_details?: Record<string, CoachingSessionDetail>
  timelines_by_session?: Record<string, Annotation[]>
  live_notes_by_session?: Record<string, string[]>
  chat_history_by_session?: Record<string, ChatMessage[]>
}

const MOCK_DELAY_MS = 1000
const demoSessionStartTimes = new Map<string, number>()

const DEMO_STATUS_SCHEDULE: Array<{ status: SessionStatus; fromMs: number }> = [
  { status: 'draft', fromMs: 0 },
  { status: 'media_attached', fromMs: 6000 },
  { status: 'queued_ml', fromMs: 12000 },
  { status: 'processing_ml', fromMs: 18000 },
  { status: 'ml_ready', fromMs: 30000 },
  { status: 'processing_coach', fromMs: 42000 },
  { status: 'ready', fromMs: 54000 },
]

function getDemoStatus(sessionId: string): SessionStatus {
  const startedAt = demoSessionStartTimes.get(sessionId)
  if (!startedAt) {
    return 'draft'
  }

  const elapsed = Date.now() - startedAt
  let current: SessionStatus = 'draft'
  for (const entry of DEMO_STATUS_SCHEDULE) {
    if (elapsed >= entry.fromMs) {
      current = entry.status
    }
  }
  return current
}

function getDemoElapsedMs(sessionId: string): number {
  const startedAt = demoSessionStartTimes.get(sessionId)
  if (!startedAt) {
    return 0
  }
  return Date.now() - startedAt
}

function getCookieValue(name: string): string | null {
  const cookies = document.cookie ? document.cookie.split('; ') : []
  const targetCookie = cookies.find((cookie) => cookie.startsWith(`${name}=`))

  return targetCookie ? decodeURIComponent(targetCookie.split('=').slice(1).join('=')) : null
}

function getCsrfTokenFromCookie(): string {
  const token = getCookieValue('csrftoken')

  if (!token) {
    throw {
      message: 'Missing CSRF token cookie. Refresh and try again.',
    } satisfies ApiError
  }

  return token
}

async function getCsrfTokenForPost(): Promise<string> {
  const existingToken = getCookieValue('csrftoken')
  if (existingToken) {
    return existingToken
  }

  const csrfResponse = await fetch(`${API_BASE_URL}/api/v1/clients/csrf`, {
    method: 'GET',
    credentials: 'include',
  })

  if (!csrfResponse.ok) {
    throw {
      message: `Failed to initialize CSRF token (status ${csrfResponse.status}).`,
      status: csrfResponse.status,
    } satisfies ApiError
  }

  return getCsrfTokenFromCookie()
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const data = (await response.json().catch(() => null)) as T | null

  if (!response.ok) {
    const errorData = data && typeof data === 'object' ? (data as Record<string, unknown>) : null
    const errorMessage =
      errorData && typeof errorData.message === 'string'
        ? errorData.message
        : errorData && typeof errorData.detail === 'string'
          ? errorData.detail
        : `Request failed with status ${response.status}`

    throw {
      message: errorMessage,
      status: response.status,
    } satisfies ApiError
  }

  if (!data) {
    throw {
      message: 'Empty response from server.',
      status: response.status,
    } satisfies ApiError
  }

  return data
}

async function ensureSuccessResponse(response: Response): Promise<void> {
  if (response.ok) {
    return
  }

  const data = (await response.json().catch(() => null)) as { message?: string } | null
  const errorMessage =
    data && typeof data.message === 'string' ? data.message : `Request failed with status ${response.status}`

  throw {
    message: errorMessage,
    status: response.status,
  } satisfies ApiError
}

async function delay(milliseconds: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function getMockAuthData(): Promise<MockAuthData> {
  const mockUrl = new URL('./mock/auth.json', import.meta.url).href
  const response = await fetch(mockUrl)

  return parseJsonResponse<MockAuthData>(response)
}

async function getMockSessionsData(): Promise<MockSessionsData> {
  const mockUrl = new URL('./mock/sessions.json', import.meta.url).href
  const response = await fetch(mockUrl)

  return parseJsonResponse<MockSessionsData>(response)
}

function getMockSessionDetail(mockData: MockSessionsData, sessionId: string): CoachingSessionDetail {
  const detailedSession = mockData.session_details?.[sessionId]
  if (detailedSession) {
    return detailedSession
  }

  const listSession = mockData.sessions.find((session) => session.id === sessionId)
  if (listSession) {
    return {
      id: listSession.id,
      title: listSession.title,
      status: listSession.status,
      created_at: listSession.created_at,
      updated_at: listSession.updated_at,
      video_file_url: null,
      supplementary_pdf_1_url: null,
      supplementary_pdf_2_url: null,
      supplementary_pdf_3_url: null,
      speaker_context: '',
      coach_progress: null,
    }
  }

  throw {
    message: 'Session not found.',
    status: 404,
  } satisfies ApiError
}

function getMockTimeline(mockData: MockSessionsData, sessionId: string): Annotation[] {
  return mockData.timelines_by_session?.[sessionId] ?? []
}

function getMockLiveNotes(mockData: MockSessionsData, sessionId: string): string[] {
  const mappedNotes = mockData.live_notes_by_session?.[sessionId]
  if (mappedNotes) {
    return mappedNotes
  }

  const sessionDetail = mockData.session_details?.[sessionId] as { live_notes?: string[] } | undefined
  return sessionDetail?.live_notes ?? []
}

function getMockChatHistory(mockData: MockSessionsData, sessionId: string): ChatMessage[] {
  return mockData.chat_history_by_session?.[sessionId] ?? []
}

function buildMockResponseId(): string {
  return `mock-resp-${Math.random().toString(36).slice(2, 10)}`
}

export const api = {
  auth: {
    async login(email: string, password: string): Promise<LoginResponse> {
      if (AUTH_USE_MOCK) {
        const mockData = await getMockAuthData()
        if (!email || !password || email === 'fail@example.com') {
          throw { message: mockData.errors.invalid_credentials } satisfies ApiError
        }
        return mockData.login_success
      }

      const csrfToken = await getCsrfTokenForPost()

      // TODO: real endpoint - POST /api/v1/clients/login
      const response = await fetch(`${API_BASE_URL}/api/v1/clients/login`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ email, password }),
      })

      return parseJsonResponse<LoginResponse>(response)
    },

    async signup(name: string, email: string, password: string): Promise<SignupResponse> {
      if (AUTH_USE_MOCK) {
        const mockData = await getMockAuthData()
        if (!name || !email || !password || email === 'taken@example.com') {
          throw { message: mockData.errors.email_already_exists } satisfies ApiError
        }
        return mockData.signup_success
      }

      const csrfToken = await getCsrfTokenForPost()

      // TODO: real endpoint - POST /api/v1/clients/signup
      const response = await fetch(`${API_BASE_URL}/api/v1/clients/signup`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ name, email, password }),
      })

      return parseJsonResponse<SignupResponse>(response)
    },

    async getCurrentUser(): Promise<CurrentUserResponse> {
      if (AUTH_USE_MOCK) {
        const mockData = await getMockAuthData()
        return mockData.current_user
      }

      // TODO: real endpoint - GET /api/v1/clients/me
      const response = await fetch(`${API_BASE_URL}/api/v1/clients/me`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<CurrentUserResponse>(response)
    },

    async logout(): Promise<void> {
      if (AUTH_USE_MOCK) {
        return
      }

      const csrfToken = await getCsrfTokenForPost()
      const response = await fetch(`${API_BASE_URL}/api/v1/clients/logout`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRFToken': csrfToken,
        },
      })

      if (!response.ok) {
        const errorData = (await response.json().catch(() => null)) as Record<string, unknown> | null
        const errorMessage =
          errorData && typeof errorData.message === 'string'
            ? errorData.message
            : errorData && typeof errorData.detail === 'string'
              ? errorData.detail
              : `Request failed with status ${response.status}`

        throw {
          message: errorMessage,
          status: response.status,
        } satisfies ApiError
      }
    },
  },
  sessions: {
    async list(): Promise<CoachingSessionListItem[]> {
      if (DEMO_AI_WALKTHROUGH) {
        return []
      }

      if (SESSIONS_USE_MOCK) {
        const mockData = await getMockSessionsData()
        return mockData.sessions
      }

      // TODO: real endpoint - GET /api/v1/sessions
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<CoachingSessionListItem[]>(response)
    },

    async create(title: string): Promise<SessionCreateResponse> {
      if (DEMO_AI_WALKTHROUGH) {
        const demoSessionId = `demo-${Date.now()}`
        demoSessionStartTimes.set(demoSessionId, Date.now())
        return {
          id: demoSessionId,
          title,
          status: 'draft',
          created_at: new Date().toISOString(),
        }
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        const mockData = await getMockSessionsData()
        return {
          id: mockData.create_session_response?.id ?? 'mock-123',
          title: mockData.create_session_response?.title ?? title,
          status: 'draft',
          created_at: new Date().toISOString(),
        }
      }

      const csrfToken = getCsrfTokenFromCookie()
      // TODO: real endpoint - POST /api/v1/sessions
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ title }),
      })

      return parseJsonResponse<SessionCreateResponse>(response)
    },

    async uploadVideo(sessionId: string, videoFile: File | null): Promise<void> {
      if (DEMO_AI_WALKTHROUGH) {
        await delay(800)
        return
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        return
      }

      if (!videoFile) {
        throw {
          message: 'Video file is required.',
        } satisfies ApiError
      }

      const csrfToken = getCsrfTokenFromCookie()
      const formData = new FormData()
      formData.append('video_file', videoFile)

      // TODO: real endpoint - POST /api/v1/sessions/:id/video
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/video`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRFToken': csrfToken,
        },
        body: formData,
      })

      await ensureSuccessResponse(response)
    },

    async uploadAssets(sessionId: string, payload: SessionAssetsPayload): Promise<void> {
      if (DEMO_AI_WALKTHROUGH) {
        await delay(400)
        return
      }

      const speakerContext = payload.speakerContext?.trim() ?? ''
      const pdfFiles = payload.pdfFiles ?? []
      const hasAssets = pdfFiles.length > 0 || Boolean(speakerContext)

      if (!hasAssets) {
        return
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        return
      }

      const csrfToken = getCsrfTokenFromCookie()
      const formData = new FormData()
      const [pdf1, pdf2, pdf3] = pdfFiles
      if (pdf1) {
        formData.append('supplementary_pdf_1', pdf1)
      }
      if (pdf2) {
        formData.append('supplementary_pdf_2', pdf2)
      }
      if (pdf3) {
        formData.append('supplementary_pdf_3', pdf3)
      }
      if (speakerContext) {
        formData.append('speaker_context', speakerContext)
      }

      // TODO: real endpoint - POST /api/v1/sessions/:id/assets
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/assets`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRFToken': csrfToken,
        },
        body: formData,
      })

      await ensureSuccessResponse(response)
    },

    async startAnalysis(sessionId: string): Promise<void> {
      if (DEMO_AI_WALKTHROUGH) {
        demoSessionStartTimes.set(sessionId, Date.now())
        await delay(300)
        return
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        return
      }

      const csrfToken = getCsrfTokenFromCookie()
      // TODO: real endpoint - POST /api/v1/sessions/:id/start-analysis
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/start-analysis`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRFToken': csrfToken,
        },
      })

      await ensureSuccessResponse(response)
    },

    async getById(sessionId: string): Promise<CoachingSessionDetail> {
      if (DEMO_AI_WALKTHROUGH) {
        const mockData = await getMockSessionsData()
        const base = Object.values(mockData.session_details ?? {})[0] as Partial<CoachingSessionDetail> | undefined

        if (!base) {
          throw {
            message: 'Session not found.',
            status: 404,
          } satisfies ApiError
        }

        const status = getDemoStatus(sessionId)
        const nowIso = new Date().toISOString()

        return {
          ...base,
          id: sessionId,
          title: base.title ?? 'Demo Session',
          status,
          created_at: base.created_at ?? nowIso,
          updated_at: nowIso,
          video_file_url: '/demo/sample.mp4',
          supplementary_pdf_1_url: base.supplementary_pdf_1_url ?? null,
          supplementary_pdf_2_url: base.supplementary_pdf_2_url ?? null,
          supplementary_pdf_3_url: base.supplementary_pdf_3_url ?? null,
          speaker_context: base.speaker_context ?? '',
          coach_progress: base.coach_progress ?? null,
        }
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        const mockData = await getMockSessionsData()
        return getMockSessionDetail(mockData, sessionId)
      }

      // TODO: real endpoint - GET /api/v1/sessions/:id
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<CoachingSessionDetail>(response)
    },

    async getTimeline(sessionId: string): Promise<Annotation[]> {
      if (DEMO_AI_WALKTHROUGH) {
        const status = getDemoStatus(sessionId)
        const TIMELINE_VISIBLE: SessionStatus[] = ['ml_ready', 'processing_coach', 'ready', 'coach_failed']
        if (!TIMELINE_VISIBLE.includes(status)) {
          return []
        }

        const mockData = await getMockSessionsData()
        return getMockTimeline(mockData, Object.keys(mockData.timelines_by_session ?? {})[0] ?? sessionId)
      }

      if (SESSIONS_USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        const mockData = await getMockSessionsData()
        return getMockTimeline(mockData, sessionId)
      }

      // TODO: real endpoint - GET /api/v1/sessions/:id/timeline
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/timeline`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<Annotation[]>(response)
    },

    async getLiveNotes(sessionId: string): Promise<string[]> {
      if (DEMO_AI_WALKTHROUGH) {
        const elapsed = getDemoElapsedMs(sessionId)
        const status = getDemoStatus(sessionId)
        const LIVE_NOTE_STATUSES: SessionStatus[] = ['processing_ml', 'ml_ready', 'processing_coach']
        if (!LIVE_NOTE_STATUSES.includes(status)) {
          return []
        }

        const mockData = await getMockSessionsData()
        const firstLiveNotesKey = Object.keys(mockData.live_notes_by_session ?? {})[0] ?? sessionId
        const notes = getMockLiveNotes(mockData, firstLiveNotesKey)

        return notes.slice(0, Math.ceil((elapsed / 54000) * notes.length))
      }

      if (SESSIONS_USE_MOCK) {
        await delay(250)
        const mockData = await getMockSessionsData()
        return getMockLiveNotes(mockData, sessionId)
      }

      // TODO: real endpoint - replace with backend live note stream/source when available
      return []
    },
  },

  chat: {
    async getHistory(sessionId: string): Promise<ChatMessage[]> {
      if (CHAT_USE_MOCK) {
        await delay(400)
        const mockData = await getMockSessionsData()
        return getMockChatHistory(mockData, sessionId)
      }

      // TODO: real endpoint - GET /api/v1/sessions/:id/chat/history
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/chat/history`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<ChatMessage[]>(response)
    },

    async sendMessage(sessionId: string, message: string): Promise<ChatSendResponse> {
      if (CHAT_USE_MOCK) {
        await delay(300)
        if (!message.trim()) {
          throw {
            message: 'Message is required.',
            status: 400,
          } satisfies ApiError
        }

        return { response_id: buildMockResponseId() }
      }

      const csrfToken = getCsrfTokenFromCookie()
      // TODO: real endpoint - POST /api/v1/sessions/:id/chat/messages
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/chat/messages`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
        },
        body: JSON.stringify({ content: message }),
      })

      return parseJsonResponse<ChatSendResponse>(response)
    },

    async streamResponse(sessionId: string, responseId: string, callbacks: ChatStreamCallbacks): Promise<void> {
      if (CHAT_USE_MOCK) {
        const reasoningTokens = [
          'Scanning',
          'delivery',
          'patterns',
          'and',
          'timing,',
          'comparing',
          'voice',
          'pace',
          'against',
          'strong',
          'segments,',
          'then',
          'checking',
          'eye-contact',
          'drift',
          'and',
          'filler-word',
          'clusters.',
        ]
        const answerTokens = [
          'You',
          'improved',
          'clarity,',
          'slow',
          'transitions,',
          'and',
          'hold',
          'camera',
          'focus',
          'one',
          'beat',
          'longer.',
        ]

        for (const token of reasoningTokens) {
          await delay(110)
          callbacks.onReasoningToken?.(`${token} `)
        }

        for (const token of answerTokens) {
          await delay(100)
          callbacks.onAnswerToken?.(`${token} `)
        }

        callbacks.onComplete?.()
        return
      }

      await new Promise<void>((resolve, reject) => {
        let resolved = false
        // TODO: real endpoint - GET /api/v1/sessions/:id/chat/streams/:response_id
        const eventSource = new EventSource(`${API_BASE_URL}/api/v1/sessions/${sessionId}/chat/streams/${responseId}`, {
          withCredentials: true,
        })

        const handleComplete = () => {
          if (resolved) {
            return
          }
          resolved = true
          eventSource.close()
          callbacks.onComplete?.()
          resolve()
        }

        const handleTokenPayload = (payload: unknown) => {
          if (!payload || typeof payload !== 'object') {
            return
          }

          const token = 'token' in payload && typeof payload.token === 'string' ? payload.token : ''
          const phase = 'phase' in payload && typeof payload.phase === 'string' ? payload.phase : 'answer'

          if (!token) {
            return
          }

          if (phase === 'reasoning') {
            callbacks.onReasoningToken?.(token)
          } else {
            callbacks.onAnswerToken?.(token)
          }
        }

        eventSource.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as unknown
            handleTokenPayload(payload)
          } catch {
            callbacks.onAnswerToken?.(event.data)
          }
        }

        eventSource.addEventListener('token', (event) => {
          const messageEvent = event as MessageEvent<string>
          try {
            const payload = JSON.parse(messageEvent.data) as unknown
            handleTokenPayload(payload)
          } catch {
            callbacks.onAnswerToken?.(messageEvent.data)
          }
        })

        eventSource.addEventListener('complete', () => {
          handleComplete()
        })

        eventSource.addEventListener('error', (event) => {
          const messageEvent = event as MessageEvent<string>
          if (messageEvent.data) {
            callbacks.onError?.(messageEvent.data)
          } else {
            callbacks.onError?.('Chat stream failed.')
          }
          if (!resolved) {
            resolved = true
            eventSource.close()
            reject(
              {
                message: 'Chat stream failed.',
              } satisfies ApiError,
            )
          }
        })

        eventSource.onerror = () => {
          if (!resolved) {
            resolved = true
            eventSource.close()
            callbacks.onError?.('Chat stream connection lost.')
            reject(
              {
                message: 'Chat stream connection lost.',
              } satisfies ApiError,
            )
          }
        }
      })
    },
  },
}

export default api
