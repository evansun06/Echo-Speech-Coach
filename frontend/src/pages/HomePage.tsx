import { useEffect, useState } from 'react'
import { Alert, Box, Button, Card, CardContent, Chip, CircularProgress, Container, Stack, Typography } from '@mui/material'
import api from '../api'
import type { ApiError, CoachingSessionListItem } from '../api'

function formatCreatedDate(value: string): string {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds}s`
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60

  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds.toString().padStart(2, '0')}s`
  }

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60

  return `${hours}h ${remainingMinutes.toString().padStart(2, '0')}m ${remainingSeconds.toString().padStart(2, '0')}s`
}

function formatStatusLabel(status: string): string {
  return status
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function HomePage() {
  const [sessions, setSessions] = useState<CoachingSessionListItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadSessions = async () => {
      setIsLoading(true)
      setError(null)

      try {
        const data = await api.sessions.list()
        if (isMounted) {
          setSessions(data)
        }
      } catch (loadError) {
        const apiError = loadError as ApiError
        if (isMounted) {
          setError(apiError.message || 'Failed to load sessions.')
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    void loadSessions()

    return () => {
      isMounted = false
    }
  }, [])

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Stack spacing={3}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Typography component="h1" variant="h4">
            Coaching Sessions
          </Typography>
          <Button variant="contained">New Session</Button>
        </Stack>

        {isLoading ? (
          <Stack spacing={2} alignItems="center" sx={{ py: 8 }}>
            <CircularProgress />
            <Typography color="text.secondary">Loading sessions...</Typography>
          </Stack>
        ) : (
          <>
            {error && <Alert severity="error">{error}</Alert>}

            {!error && sessions.length === 0 ? (
              <Box
                sx={{
                  py: 8,
                  border: '1px dashed',
                  borderColor: 'divider',
                  borderRadius: 2,
                  textAlign: 'center',
                }}
              >
                <Typography variant="h6">No coaching sessions yet</Typography>
                <Typography color="text.secondary">Create a new session to get started.</Typography>
              </Box>
            ) : (
              <Stack spacing={2}>
                {sessions.map((session) => (
                  <Card key={session.id} variant="outlined">
                    <CardContent>
                      <Stack spacing={1.5}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
                          <Typography variant="h6">{session.title}</Typography>
                          <Chip label={formatStatusLabel(session.status)} size="small" />
                        </Stack>
                        <Typography color="text.secondary" variant="body2">
                          Created: {formatCreatedDate(session.created_at)}
                        </Typography>
                        <Typography color="text.secondary" variant="body2">
                          Duration: {formatDuration(session.duration_seconds)}
                        </Typography>
                      </Stack>
                    </CardContent>
                  </Card>
                ))}
              </Stack>
            )}
          </>
        )}
      </Stack>
    </Container>
  )
}

export default HomePage
