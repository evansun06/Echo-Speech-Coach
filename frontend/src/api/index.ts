export const USE_MOCK = true
export const API_BASE_URL = 'http://localhost:8000'

export interface AuthUser {
  id: string
  name: string
  email: string
}

export interface LoginResponse {
  message: string
  user: AuthUser
}

export interface SignupResponse {
  message: string
  user: AuthUser
}

export interface CurrentUserResponse {
  user: AuthUser
}

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

export interface CoachingSessionListItem {
  id: string
  title: string
  created_at: string
  duration_seconds: number
  status: SessionStatus
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

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const data = (await response.json().catch(() => null)) as T | null

  if (!response.ok) {
    const errorMessage =
      data && typeof data === 'object' && 'message' in data && typeof data.message === 'string'
        ? data.message
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

export const api = {
  auth: {
    async login(email: string, password: string): Promise<LoginResponse> {
      if (USE_MOCK) {
        const mockData = await getMockAuthData()
        if (!email || !password || email === 'fail@example.com') {
          throw { message: mockData.errors.invalid_credentials } satisfies ApiError
        }
        return mockData.login_success
      }

      const csrfToken = getCsrfTokenFromCookie()

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
      if (USE_MOCK) {
        const mockData = await getMockAuthData()
        if (!name || !email || !password || email === 'taken@example.com') {
          throw { message: mockData.errors.email_already_exists } satisfies ApiError
        }
        return mockData.signup_success
      }

      const csrfToken = getCsrfTokenFromCookie()

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
      if (USE_MOCK) {
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
  },
  sessions: {
    async list(): Promise<CoachingSessionListItem[]> {
      if (USE_MOCK) {
        const mockData = await getMockSessionsData()
        return mockData.sessions
      }

      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<CoachingSessionListItem[]>(response)
    },
  },
}

export default api
