import { CircularProgress, Stack, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import api from './api'
import SplashScreen from './components/SplashScreen'
import DashboardPage from './pages/DashboardPage'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'

type AuthStatus = 'authenticated' | 'unauthenticated'

function AuthLoadingScreen() {
  return (
    <Stack minHeight="100vh" alignItems="center" justifyContent="center" spacing={2}>
      <CircularProgress />
      <Typography color="text.secondary">Checking authentication...</Typography>
    </Stack>
  )
}

function AppRoutes() {
  const location = useLocation()
  const [authStatus, setAuthStatus] = useState<AuthStatus>('unauthenticated')
  const [isCheckingAuth, setIsCheckingAuth] = useState(true)
  const [checkedPath, setCheckedPath] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const checkAuth = async () => {
      setIsCheckingAuth(true)

      try {
        const user = await api.auth.getCurrentUser()
        console.log(`Authenticated user: ${user.name || user.email}`)
        if (isMounted) {
          setAuthStatus('authenticated')
        }
      } catch {
        if (isMounted) {
          setAuthStatus('unauthenticated')
        }
      } finally {
        if (isMounted) {
          setCheckedPath(location.pathname)
          setIsCheckingAuth(false)
        }
      }
    }

    void checkAuth()

    return () => {
      isMounted = false
    }
  }, [location.pathname])

  if (isCheckingAuth || checkedPath !== location.pathname) {
    return <AuthLoadingScreen />
  }

  return (
    <Routes>
      <Route path="/" element={authStatus === 'authenticated' ? <HomePage /> : <Navigate to="/login" replace />} />
      <Route path="/sessions/:id" element={authStatus === 'authenticated' ? <DashboardPage /> : <Navigate to="/login" replace />} />
      <Route path="/login" element={authStatus === 'authenticated' ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route path="/signup" element={authStatus === 'authenticated' ? <Navigate to="/" replace /> : <SignupPage />} />
      <Route path="*" element={<Navigate to={authStatus === 'authenticated' ? '/' : '/login'} replace />} />
    </Routes>
  )
}

function App() {
  const [showSplash, setShowSplash] = useState(true)

  if (showSplash) {
    return <SplashScreen onDone={() => setShowSplash(false)} />
  }

  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}

export default App
