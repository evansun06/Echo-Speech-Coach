import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'
import api, { USE_MOCK } from '../api'
import type { ApiError, CoachingSessionListItem, SessionAssetsPayload } from '../api'

type VideoSource = 'upload' | 'record'


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
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<CoachingSessionListItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isLoggingOut, setIsLoggingOut] = useState(false)
  const [logoutError, setLogoutError] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalStep, setModalStep] = useState(1)
  const [videoSource, setVideoSource] = useState<VideoSource | null>(null)
  const [sessionName, setSessionName] = useState('')
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [supportingPdfFiles, setSupportingPdfFiles] = useState<File[]>([])
  const [supportingText, setSupportingText] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [modalError, setModalError] = useState<string | null>(null)

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

  const handleLogout = async () => {
    setLogoutError(null)
    setIsLoggingOut(true)

    try {
      await api.auth.logout()
      navigate('/login')
    } catch (logoutRequestError) {
      const apiError = logoutRequestError as ApiError
      setLogoutError(apiError.message || 'Logout failed. Please try again.')
    } finally {
      setIsLoggingOut(false)
    }
  }

  const resetModalState = () => {
    setModalStep(1)
    setVideoSource(null)
    setSessionName('')
    setVideoFile(null)
    setSupportingPdfFiles([])
    setSupportingText('')
    setIsSubmitting(false)
    setModalError(null)
  }

  const openNewSessionModal = () => {
    resetModalState()
    setIsModalOpen(true)
  }

  const closeNewSessionModal = () => {
    if (isSubmitting) {
      return
    }

    setIsModalOpen(false)
    resetModalState()
  }

  const handleVideoFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    setVideoFile(file)
    setModalError(null)
  }

  const handleSupportingPdfFilesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const incomingFiles = Array.from(event.target.files ?? [])
    const allPdfFiles = incomingFiles.filter(
      (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'),
    )

    if (allPdfFiles.length !== incomingFiles.length) {
      setModalError('Only PDF files are allowed for supporting materials.')
      return
    }

    setSupportingPdfFiles((previous) => {
      const next = [...previous, ...allPdfFiles].slice(0, 3)
      if (previous.length + allPdfFiles.length > 3) {
        setModalError('You can upload up to 3 PDFs.')
      } else {
        setModalError(null)
      }
      return next
    })

    event.target.value = ''
  }

  const handleRemoveSupportingPdf = (indexToRemove: number) => {
    setSupportingPdfFiles((previous) => previous.filter((_, index) => index !== indexToRemove))
    setModalError(null)
  }

  const handleNextStep = () => {
    if (!videoSource) {
      setModalError('Choose a video source to continue.')
      return
    }

    setModalError(null)
    setModalStep(2)
  }

  const handleBackStep = () => {
    setModalError(null)
    setModalStep(1)
  }

  const handleConfirmSession = async () => {
    if (!videoSource) {
      setModalError('Choose a video source before confirming.')
      return
    }

    const trimmedSessionName = sessionName.trim()
    if (!trimmedSessionName) {
      setModalError('Session name is required.')
      return
    }

    if (videoSource === 'upload' && !videoFile) {
      setModalError('Select one video file before confirming.')
      return
    }

    if (videoSource === 'record' && !USE_MOCK) {
      setModalError('Record Now is coming soon. Please use Upload Video for now.')
      return
    }

    setIsSubmitting(true)
    setModalError(null)

    try {
      const createResponse = await api.sessions.create(trimmedSessionName)
      const sessionId = createResponse.id

      await api.sessions.uploadVideo(sessionId, videoSource === 'upload' ? videoFile : null)

      const optionalAssets: SessionAssetsPayload = {
        pdfFiles: supportingPdfFiles,
        supportingText,
      }
      const hasOptionalAssets = Boolean(supportingPdfFiles.length > 0 || supportingText.trim())

      if (hasOptionalAssets) {
        await api.sessions.uploadAssets(sessionId, optionalAssets)
      }

      await api.sessions.startAnalysis(sessionId)

      setIsModalOpen(false)
      resetModalState()
      navigate(`/sessions/${sessionId}`)
    } catch (submitError) {
      const apiError = submitError as ApiError
      setModalError(apiError.message || 'Failed to create session. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Stack spacing={3}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Typography component="h1" variant="h4">
            Coaching Sessions
          </Typography>
          <Stack direction="row" spacing={1.5}>
            
            <Button variant="contained" onClick={openNewSessionModal}>
              New Session
            </Button>

            <Button variant="outlined" color="inherit" onClick={handleLogout} disabled={isLoggingOut}>
              {isLoggingOut ? 'Logging out...' : 'Log out'}
            </Button>
          </Stack>
        </Stack>

        {isLoading ? (
          <Stack spacing={2} alignItems="center" sx={{ py: 8 }}>
            <CircularProgress />
            <Typography color="text.secondary">Loading sessions...</Typography>
          </Stack>
        ) : (
          <>
            {logoutError && <Alert severity="error">{logoutError}</Alert>}
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

      <Dialog open={isModalOpen} onClose={closeNewSessionModal} fullWidth maxWidth="md">
        <DialogTitle>Start Coaching Session</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={3} sx={{ pt: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Step {modalStep} of 2
            </Typography>

            {modalStep === 1 ? (
              <Stack spacing={2}>
                <Typography variant="h6">Choose Video Source</Typography>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <Card
                    variant="outlined"
                    sx={{
                      flex: 1,
                      borderColor: videoSource === 'upload' ? 'primary.main' : 'divider',
                      borderWidth: videoSource === 'upload' ? 2 : 1,
                    }}
                  >
                    <CardActionArea onClick={() => setVideoSource('upload')} sx={{ p: 2, height: '100%' }}>
                      <Stack spacing={1}>
                        <Typography variant="h6">Upload Video</Typography>
                        <Typography color="text.secondary" variant="body2">
                          Select a single video file from your device.
                        </Typography>
                      </Stack>
                    </CardActionArea>
                  </Card>

                  <Card
                    variant="outlined"
                    sx={{
                      flex: 1,
                      borderColor: videoSource === 'record' ? 'primary.main' : 'divider',
                      borderWidth: videoSource === 'record' ? 2 : 1,
                    }}
                  >
                    <CardActionArea onClick={() => setVideoSource('record')} sx={{ p: 2, height: '100%' }}>
                      <Stack spacing={1}>
                        <Typography variant="h6">Record Now</Typography>
                        <Typography color="text.secondary" variant="body2">
                          Placeholder for in-browser recording.
                        </Typography>
                      </Stack>
                    </CardActionArea>
                  </Card>
                </Stack>

                {videoSource === 'upload' && (
                  <Typography color="text.secondary" variant="body2">
                    Continue to add your video and optional materials.
                  </Typography>
                )}
              </Stack>
            ) : (
              <Stack spacing={2}>
                <TextField
                  label="Session name"
                  value={sessionName}
                  onChange={(event) => setSessionName(event.target.value.slice(0, 100))}
                  placeholder="e.g. Q3 Investor Pitch, Team Standup Practice"
                  helperText={`${sessionName.length}/100`}
                  required
                  inputProps={{ maxLength: 100 }}
                  fullWidth
                />

                <Stack spacing={1}>
                  <Typography variant="h6">Upload your presentation video</Typography>
                  <Button component="label" variant="outlined" sx={{ alignSelf: 'flex-start' }}>
                    Choose Video
                    <input hidden type="file" accept="video/*" onChange={handleVideoFileChange} />
                  </Button>
                  <Typography color="text.secondary" variant="body2">
                    {videoFile ? `Selected: ${videoFile.name}` : 'No file selected'}
                  </Typography>
                  <Typography color="text.secondary" variant="caption">
                    One video per session. Multiple uploads are not allowed.
                  </Typography>
                </Stack>

                {videoSource === 'record' && (
                  <Alert severity="info">
                    Record Now is still a placeholder. In mock mode you can continue without uploading a video file.
                  </Alert>
                )}

                <Stack spacing={1}>
                  <Typography variant="subtitle2">Supporting materials (optional)</Typography>
                  <Button
                    component="label"
                    variant="outlined"
                    sx={{ alignSelf: 'flex-start' }}
                    disabled={supportingPdfFiles.length >= 3}
                  >
                    Upload PDFs
                    <input
                      hidden
                      type="file"
                      accept=".pdf,application/pdf"
                      multiple
                      onChange={handleSupportingPdfFilesChange}
                    />
                  </Button>
                  <Typography color="text.secondary" variant="caption">
                    Add up to 3 PDFs and/or any notes or context.
                  </Typography>

                  {supportingPdfFiles.length > 0 ? (
                    <Stack spacing={0.75}>
                      {supportingPdfFiles.map((file, index) => (
                        <Stack
                          key={`${file.name}-${index}`}
                          direction="row"
                          spacing={1}
                          alignItems="center"
                          justifyContent="space-between"
                          sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, px: 1.25, py: 0.75 }}
                        >
                          <Typography variant="body2">{file.name}</Typography>
                          <IconButton
                            aria-label={`Remove ${file.name}`}
                            size="small"
                            onClick={() => handleRemoveSupportingPdf(index)}
                          >
                            ×
                          </IconButton>
                        </Stack>
                      ))}
                    </Stack>
                  ) : (
                    <Typography color="text.secondary" variant="body2">
                      No PDFs selected
                    </Typography>
                  )}
                </Stack>

                <TextField
                  label="Supporting notes (optional)"
                  multiline
                  minRows={5}
                  value={supportingText}
                  onChange={(event) => setSupportingText(event.target.value)}
                  placeholder="Add any notes, script excerpts, audience context, or speaking goals"
                  helperText="Add up to 3 PDFs and/or any notes or context"
                  fullWidth
                />
              </Stack>
            )}

            {modalError && <Alert severity="error">{modalError}</Alert>}
          </Stack>
        </DialogContent>

        <DialogActions>
          <Button onClick={closeNewSessionModal} disabled={isSubmitting}>
            Cancel
          </Button>
          {modalStep === 2 && (
            <Button onClick={handleBackStep} disabled={isSubmitting}>
              Back
            </Button>
          )}

          {modalStep === 1 ? (
            <Button onClick={handleNextStep} variant="contained" disabled={isSubmitting}>
              Next
            </Button>
          ) : (
            <Button
              onClick={handleConfirmSession}
              variant="contained"
              disabled={isSubmitting || !sessionName.trim()}
              startIcon={isSubmitting ? <CircularProgress size={16} color="inherit" /> : undefined}
            >
              {isSubmitting ? 'Starting...' : 'Confirm'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Container>
  )
}

export default HomePage
