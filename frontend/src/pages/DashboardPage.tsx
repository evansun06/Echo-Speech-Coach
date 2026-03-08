import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Container,
  Divider,
  LinearProgress,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'
import { useNavigate, useParams } from 'react-router-dom'
import api, { API_BASE_URL } from '../api'
import type { Annotation, ApiError, CoachStage, CoachStageStatus, CoachingSessionDetail, SessionStatus } from '../api'

const TERMINAL_STATUSES: SessionStatus[] = ['ready', 'failed', 'coach_failed']
const TIMELINE_VISIBLE_STATUSES: SessionStatus[] = ['ml_ready', 'processing_coach', 'ready', 'coach_failed']

function formatStatusLabel(status: string): string {
  return status
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function isTerminalStatus(status: SessionStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}

function shouldShowTimeline(status: SessionStatus): boolean {
  return TIMELINE_VISIBLE_STATUSES.includes(status)
}

function getStageColor(status: CoachStageStatus): string {
  switch (status) {
    case 'completed':
      return 'success.main'
    case 'processing':
      return 'warning.main'
    case 'failed':
      return 'error.main'
    default:
      return 'grey.500'
  }
}

function StageStatusIcon({ status }: { status: CoachStageStatus }) {
  if (status === 'processing') {
    return <CircularProgress size={16} />
  }

  return (
    <Box
      sx={{
        width: 18,
        height: 18,
        borderRadius: '50%',
        bgcolor: getStageColor(status),
        color: 'common.white',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 12,
        fontWeight: 700,
      }}
    >
      {status === 'completed' ? '✓' : status === 'failed' ? '!' : ''}
    </Box>
  )
}

function CoachPanel({
  session,
  noteExpanded,
  onToggleNote,
}: {
  session: CoachingSessionDetail
  noteExpanded: Record<string, boolean>
  onToggleNote: (noteId: string) => void
}) {
  const stages = session.coach_progress?.stages ?? []

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ p: 2.5 }}>
        <Stack spacing={2}>
          <Stack spacing={1}>
            <Typography variant="h6">Coach Panel</Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography color="text.secondary" variant="body2">
                Status:
              </Typography>
              <Chip label={formatStatusLabel(session.status)} size="small" />
            </Stack>
          </Stack>

          {session.status === 'processing_coach' && (
            <Stack spacing={1}>
              <LinearProgress />
              <Typography variant="body2" color="text.secondary">
                Coach notes are still being generated...
              </Typography>
            </Stack>
          )}

          {stages.length === 0 ? (
            <Alert severity="info">Coach stages are not available yet.</Alert>
          ) : (
            <Stack spacing={1.5}>
              {stages.map((stage: CoachStage) => (
                <Card key={stage.stage_key} variant="outlined">
                  <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Stack spacing={1.25}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <StageStatusIcon status={stage.status} />
                        <Typography variant="subtitle2">{stage.label}</Typography>
                        <Chip size="small" label={formatStatusLabel(stage.status)} sx={{ ml: 'auto' }} />
                      </Stack>

                      {stage.notes.length === 0 ? (
                        <Typography variant="caption" color="text.secondary">
                          No notes yet for this stage.
                        </Typography>
                      ) : (
                        <Stack spacing={1}>
                          {stage.notes.map((note) => {
                            const isExpanded = noteExpanded[note.note_id] ?? !note.default_collapsed

                            return (
                              <Card key={note.note_id} variant="outlined" sx={{ bgcolor: 'grey.50' }}>
                                <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
                                  <Stack spacing={1}>
                                    <Stack direction="row" alignItems="center" justifyContent="space-between">
                                      <Typography variant="body2" fontWeight={600}>
                                        {note.title}
                                      </Typography>
                                      <Button size="small" onClick={() => onToggleNote(note.note_id)}>
                                        {isExpanded ? 'Hide' : 'Show'}
                                      </Button>
                                    </Stack>
                                    <Collapse in={isExpanded}>
                                      <Stack spacing={1}>
                                        <Typography variant="body2" color="text.secondary">
                                          {note.body}
                                        </Typography>
                                        {note.evidence_refs.length > 0 && (
                                          <Typography variant="caption" color="text.secondary">
                                            Evidence: {note.evidence_refs.join(', ')}
                                          </Typography>
                                        )}
                                      </Stack>
                                    </Collapse>
                                  </Stack>
                                </CardContent>
                              </Card>
                            )
                          })}
                        </Stack>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

function VideoPlayer({ sessionId }: { sessionId: string }) {
  const streamUrl = `${API_BASE_URL}/api/v1/sessions/${sessionId}/video-stream`

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ p: 2.5 }}>
        <Stack spacing={2}>
          <Typography variant="h6">Video Player</Typography>
          <Box
            sx={{
              borderRadius: 1.5,
              overflow: 'hidden',
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'common.black',
            }}
          >
            <video controls style={{ width: '100%', display: 'block' }} preload="metadata">
              {/* TODO: real endpoint - GET /api/v1/sessions/:id/video-stream */}
              <source src={streamUrl} type="video/mp4" />
            </video>
          </Box>
          <Typography variant="body2" color="text.secondary">
            Native controls provide play/pause and timeline scrubbing in this placeholder phase.
          </Typography>
        </Stack>
      </CardContent>
    </Card>
  )
}

function AnnotationTimeline({
  status,
  annotations,
  timelineError,
}: {
  status: SessionStatus
  annotations: Annotation[]
  timelineError: string | null
}) {
  if (!shouldShowTimeline(status)) {
    return (
      <Card variant="outlined">
        <CardContent sx={{ p: 2.5 }}>
          <Stack spacing={1}>
            <Typography variant="h6">Annotation Timeline</Typography>
            <Typography variant="body2" color="text.secondary">
              Timeline is hidden until status reaches ml_ready.
            </Typography>
          </Stack>
        </CardContent>
      </Card>
    )
  }

  const audioTrack = annotations.filter((annotation) => annotation.source === 'audio')
  const videoTrack = annotations.filter((annotation) => annotation.source === 'video')
  const maxEndMs = Math.max(1, ...annotations.map((annotation) => annotation.end_ms))

  return (
    <Card variant="outlined">
      <CardContent sx={{ p: 2.5 }}>
        <Stack spacing={2}>
          <Typography variant="h6">Annotation Timeline</Typography>
          {timelineError && <Alert severity="warning">{timelineError}</Alert>}

          {annotations.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No annotations available yet.
            </Typography>
          ) : (
            <Stack spacing={2}>
              <TimelineTrack label="Audio" color="info.main" annotations={audioTrack} maxEndMs={maxEndMs} />
              <TimelineTrack label="Video" color="secondary.main" annotations={videoTrack} maxEndMs={maxEndMs} />
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

function TimelineTrack({
  label,
  color,
  annotations,
  maxEndMs,
}: {
  label: string
  color: string
  annotations: Annotation[]
  maxEndMs: number
}) {
  return (
    <Stack spacing={1}>
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="subtitle2">{label}</Typography>
        <Typography variant="caption" color="text.secondary">
          0:00 - {formatTimestamp(maxEndMs)}
        </Typography>
      </Stack>

      <Box
        sx={{
          position: 'relative',
          height: 44,
          borderRadius: 1.5,
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: 'grey.100',
          overflow: 'hidden',
        }}
      >
        <Divider sx={{ position: 'absolute', insetX: 0, top: '50%' }} />
        {annotations.map((annotation) => {
          const position = Math.min(100, Math.max(0, (annotation.start_ms / maxEndMs) * 100))

          return (
            <Tooltip
              key={annotation.id}
              title={`${formatTimestamp(annotation.start_ms)} • ${annotation.summary}`}
              placement="top"
              arrow
            >
              <Box
                sx={{
                  position: 'absolute',
                  left: `${position}%`,
                  top: '50%',
                  transform: 'translate(-50%, -50%)',
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  bgcolor: color,
                  border: '2px solid',
                  borderColor: 'common.white',
                  boxShadow: 1,
                  cursor: 'pointer',
                }}
              />
            </Tooltip>
          )
        })}
      </Box>
    </Stack>
  )
}

function DashboardPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [session, setSession] = useState<CoachingSessionDetail | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [timelineError, setTimelineError] = useState<string | null>(null)
  const [noteExpanded, setNoteExpanded] = useState<Record<string, boolean>>({})

  useEffect(() => {
    if (!id) {
      setError('Missing session id.')
      setIsLoading(false)
      return
    }

    let isMounted = true
    let intervalId: ReturnType<typeof setInterval> | null = null

    const stopPolling = () => {
      if (intervalId !== null) {
        clearInterval(intervalId)
        intervalId = null
      }
    }

    const loadTimelineIfVisible = async (status: SessionStatus) => {
      if (!shouldShowTimeline(status)) {
        if (isMounted) {
          setAnnotations([])
          setTimelineError(null)
        }
        return
      }

      try {
        const timelineData = await api.sessions.getTimeline(id)
        if (isMounted) {
          setAnnotations(timelineData)
          setTimelineError(null)
        }
      } catch (timelineLoadError) {
        const apiError = timelineLoadError as ApiError
        if (isMounted) {
          setTimelineError(apiError.message || 'Failed to load timeline.')
        }
      }
    }

    const loadSession = async (initialLoad: boolean): Promise<SessionStatus | null> => {
      if (initialLoad && isMounted) {
        setIsLoading(true)
      }

      try {
        const sessionData = await api.sessions.getById(id)
        if (!isMounted) {
          return null
        }

        setSession(sessionData)
        setError(null)
        await loadTimelineIfVisible(sessionData.status)
        return sessionData.status
      } catch (loadError) {
        const apiError = loadError as ApiError
        if (isMounted) {
          setError(apiError.message || 'Failed to load session.')
        }
        return null
      } finally {
        if (initialLoad && isMounted) {
          setIsLoading(false)
        }
      }
    }

    const startPolling = async () => {
      const initialStatus = await loadSession(true)
      if (!isMounted || !initialStatus || isTerminalStatus(initialStatus)) {
        return
      }

      intervalId = setInterval(() => {
        void loadSession(false).then((status) => {
          if (status && isTerminalStatus(status)) {
            stopPolling()
          }
        })
      }, 3000)
    }

    void startPolling()

    return () => {
      isMounted = false
      stopPolling()
    }
  }, [id])

  useEffect(() => {
    if (!session) {
      return
    }

    setNoteExpanded((previous) => {
      const next = { ...previous }
      for (const stage of session.coach_progress.stages) {
        for (const note of stage.notes) {
          if (next[note.note_id] === undefined) {
            next[note.note_id] = !note.default_collapsed
          }
        }
      }
      return next
    })
  }, [session])

  const handleToggleNote = (noteId: string) => {
    setNoteExpanded((previous) => ({
      ...previous,
      [noteId]: !previous[noteId],
    }))
  }

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Stack spacing={2.5}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Stack spacing={0.5}>
            <Typography component="h1" variant="h4">
              {session?.title || 'Session Dashboard'}
            </Typography>
            {session && (
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  Session ID: {session.id}
                </Typography>
                <Chip size="small" label={formatStatusLabel(session.status)} />
              </Stack>
            )}
          </Stack>
          <Button variant="outlined" onClick={() => navigate('/')}>
            Back to Home
          </Button>
        </Stack>

        {isLoading ? (
          <Stack spacing={2} alignItems="center" sx={{ py: 8 }}>
            <CircularProgress />
            <Typography color="text.secondary">Loading session...</Typography>
          </Stack>
        ) : error ? (
          <Alert severity="error">{error}</Alert>
        ) : !session ? (
          <Alert severity="info">No session data available.</Alert>
        ) : (
          <Box
            sx={{
              display: 'grid',
              gap: 2,
              gridTemplateColumns: { xs: '1fr', lg: '360px 1fr' },
              gridTemplateAreas: {
                xs: '"coach" "video" "timeline"',
                lg: '"coach video" "timeline timeline"',
              },
            }}
          >
            <Box sx={{ gridArea: 'coach' }}>
              <CoachPanel session={session} noteExpanded={noteExpanded} onToggleNote={handleToggleNote} />
            </Box>
            <Box sx={{ gridArea: 'video' }}>
              <VideoPlayer sessionId={session.id} />
            </Box>
            <Box sx={{ gridArea: 'timeline' }}>
              <AnnotationTimeline status={session.status} annotations={annotations} timelineError={timelineError} />
            </Box>
          </Box>
        )}
      </Stack>
    </Container>
  )
}

export default DashboardPage
