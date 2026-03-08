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

      const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
        method: 'GET',
        credentials: 'include',
      })

      return parseJsonResponse<CoachingSessionListItem[]>(response)
    },
  },
}

export default api
