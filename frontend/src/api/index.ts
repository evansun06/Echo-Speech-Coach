export const USE_MOCK = true
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

export interface CoachNote {
  note_id: string
  title: string
  body: string
  evidence_refs: string[]
  default_collapsed: boolean
}

export interface CoachStage {
  stage_key: string
  label: string
  status: CoachStageStatus
  notes: CoachNote[]
}

export interface CoachProgress {
  status: CoachProgressStatus
  current_stage: string
  stages: CoachStage[]
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
  created_at: string
  duration_seconds: number
  status: SessionStatus
}

export interface CoachingSessionDetail extends CoachingSessionListItem {
  coach_progress: CoachProgress
}

export interface SessionCreateResponse {
  id: string
}

export interface SessionAssetsPayload {
  slidesFile?: File | null
  scriptText?: string
  contextText?: string
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
  session_details?: Record<string, CoachingSessionDetail>
  timelines_by_session?: Record<string, Annotation[]>
}

const MOCK_DELAY_MS = 1000

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
      ...listSession,
      coach_progress: {
        status: 'pending',
        current_stage: '',
        stages: [],
      },
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
      if (USE_MOCK) {
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

    async create(): Promise<SessionCreateResponse> {
      if (USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        return { id: 'mock-123' }
      }

      const csrfToken = getCsrfTokenFromCookie()
      // TODO: real endpoint - POST /api/v1/sessions
      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRFToken': csrfToken,
        },
      })

      return parseJsonResponse<SessionCreateResponse>(response)
    },

    async uploadVideo(sessionId: string, videoFile: File | null): Promise<void> {
      if (USE_MOCK) {
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
      formData.append('video', videoFile)

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
      const scriptText = payload.scriptText?.trim()
      const contextText = payload.contextText?.trim()
      const hasAssets = Boolean(payload.slidesFile || scriptText || contextText)

      if (!hasAssets) {
        return
      }

      if (USE_MOCK) {
        await delay(MOCK_DELAY_MS)
        return
      }

      const csrfToken = getCsrfTokenFromCookie()
      const formData = new FormData()

      if (payload.slidesFile) {
        formData.append('slides', payload.slidesFile)
      }

      if (scriptText) {
        formData.append('script_text', scriptText)
      }

      if (contextText) {
        formData.append('context', contextText)
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
      if (USE_MOCK) {
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
      if (USE_MOCK) {
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
      if (USE_MOCK) {
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
  },
}

export default api
