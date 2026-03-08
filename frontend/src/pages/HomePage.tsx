import { useEffect, useRef, useState } from 'react'
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
import api from '../api'
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
  const cameraPreviewRef = useRef<HTMLVideoElement | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

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
  const [isCameraLoading, setIsCameraLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDurationSeconds, setRecordingDurationSeconds] = useState(0)

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

  const clearRecordingTimer = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
  }

  const getMediaErrorMessage = (error: unknown): string => {
    if (error instanceof DOMException) {
      if (error.name === 'NotAllowedError') {
        return 'Camera/microphone permission was denied. Allow access and try again.'
      }
      if (error.name === 'NotFoundError') {
        return 'No camera device was found.'
      }
      if (error.name === 'NotReadableError') {
        return 'Camera is currently in use by another application.'
      }
      return error.message || 'Unable to initialize camera preview.'
    }

    if (error instanceof Error) {
      return error.message || 'Unable to initialize camera preview.'
    }

    return 'Unable to initialize camera preview.'
  }

  const waitForPreviewElement = async (): Promise<HTMLVideoElement> => {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      if (cameraPreviewRef.current) {
        return cameraPreviewRef.current
      }
      await new Promise<void>((resolve) => {
        window.setTimeout(resolve, 50)
      })
    }

    throw new Error('Camera preview is not ready yet.')
  }

  const attachStreamToPreview = async (stream: MediaStream): Promise<void> => {
    const previewElement = await waitForPreviewElement()

    const videoTrack = stream.getVideoTracks()[0]
    if (!videoTrack || videoTrack.readyState !== 'live') {
      throw new Error('Camera video track is unavailable.')
    }

    previewElement.srcObject = stream
    await previewElement.play()
  }

  const cleanupRecordingResources = () => {
    clearRecordingTimer()

    const mediaRecorder = mediaRecorderRef.current
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.ondataavailable = null
      mediaRecorder.onstop = null
      mediaRecorder.onerror = null
      mediaRecorder.stop()
    }
    mediaRecorderRef.current = null

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => {
        track.stop()
      })
      mediaStreamRef.current = null
    }

    if (cameraPreviewRef.current) {
      cameraPreviewRef.current.srcObject = null
    }

    recordedChunksRef.current = []
    setIsCameraLoading(false)
    setIsRecording(false)
    setRecordingDurationSeconds(0)
  }

  const startCameraPreview = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setModalError('Camera access is not supported in this browser.')
      return
    }

    if (mediaStreamRef.current) {
      try {
        await attachStreamToPreview(mediaStreamRef.current)
      } catch (error) {
        setModalError(getMediaErrorMessage(error))
      }
      return
    }

    setIsCameraLoading(true)

    try {
      let stream: MediaStream
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        })
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })
      }

      mediaStreamRef.current = stream
      stream.getVideoTracks().forEach((track) => {
        track.enabled = true
      })

      try {
        await attachStreamToPreview(stream)
      } catch (error) {
        stream.getTracks().forEach((track) => {
          track.stop()
        })
        mediaStreamRef.current = null
        throw error
      }

      setModalError(null)
    } catch (error) {
      setModalError(getMediaErrorMessage(error))
    } finally {
      setIsCameraLoading(false)
    }
  }

  const handleStartRecording = () => {
    if (isRecording) {
      return
    }

    if (typeof MediaRecorder === 'undefined') {
      setModalError('Recording is not supported in this browser.')
      return
    }

    const stream = mediaStreamRef.current
    if (!stream) {
      setModalError('Camera is not ready. Wait for preview before recording.')
      return
    }

    try {
      const supportedMimeTypes = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm']
      const mimeType =
        supportedMimeTypes.find((candidate) => typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(candidate)) ||
        ''

      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recordedChunksRef.current = []
      setVideoFile(null)
      setRecordingDurationSeconds(0)
      setModalError(null)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = () => {
        setModalError('Recording failed. Please try again.')
        clearRecordingTimer()
        setIsRecording(false)
      }

      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: recorder.mimeType || 'video/webm' })
        if (blob.size === 0) {
          setModalError('No video was captured. Please record again.')
          return
        }

        const recordedFile = new File([blob], `recording-${Date.now()}.webm`, {
          type: blob.type || 'video/webm',
        })

        setVideoFile(recordedFile)
        setModalError(null)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)

      clearRecordingTimer()
      recordingTimerRef.current = setInterval(() => {
        setRecordingDurationSeconds((previous) => previous + 1)
      }, 1000)
    } catch {
      setModalError('Unable to start recording. Please try again.')
    }
  }

  const handleStopRecording = () => {
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state !== 'recording') {
      return
    }

    recorder.stop()
    setIsRecording(false)
    clearRecordingTimer()
  }

  useEffect(() => {
    if (!isModalOpen || modalStep !== 2 || videoSource !== 'record') {
      cleanupRecordingResources()
      return
    }

    let isCancelled = false

    const bootCameraPreview = async () => {
      await startCameraPreview()
      if (isCancelled) {
        cleanupRecordingResources()
      }
    }

    void bootCameraPreview()

    return () => {
      isCancelled = true
      cleanupRecordingResources()
    }
  }, [isModalOpen, modalStep, videoSource])

  const resetModalState = () => {
    cleanupRecordingResources()
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

    if (videoSource === 'record' && !videoFile) {
      setModalError('Record and stop your video before confirming.')
      return
    }

    setIsSubmitting(true)
    setModalError(null)

    try {
      const createResponse = await api.sessions.create(trimmedSessionName)
      const sessionId = createResponse.id

      await api.sessions.uploadVideo(sessionId, videoFile)

      const optionalAssets: SessionAssetsPayload = {
        pdfFiles: supportingPdfFiles,
        speakerContext: supportingText,
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
                {sessions.map((session) => {
                  const sessionDuration = (session as unknown as { duration_seconds?: number }).duration_seconds

                  return (
                    <Card key={session.id} variant="outlined">
                      <CardActionArea onClick={() => navigate(`/sessions/${session.id}`)} sx={{ cursor: 'pointer' }}>
                        <CardContent>
                          <Stack spacing={1.5}>
                            <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
                              <Typography variant="h6">{session.title}</Typography>
                              <Chip label={formatStatusLabel(session.status)} size="small" />
                            </Stack>
                            <Typography color="text.secondary" variant="body2">
                              Created: {formatCreatedDate(session.created_at)}
                            </Typography>
                            {typeof sessionDuration === 'number' && (
                              <Typography color="text.secondary" variant="body2">
                                Duration: {formatDuration(sessionDuration)}
                              </Typography>
                            )}
                          </Stack>
                        </CardContent>
                      </CardActionArea>
                    </Card>
                  )
                })}
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
                  {videoSource === 'upload' ? (
                    <>
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
                    </>
                  ) : (
                    <>
                      <Typography variant="h6">Record your presentation video</Typography>
                      <Box
                        sx={{
                          position: 'relative',
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 2,
                          overflow: 'hidden',
                          bgcolor: 'common.black',
                          minHeight: 220,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <video
                          ref={cameraPreviewRef}
                          autoPlay
                          muted
                          playsInline
                          style={{
                            width: '100%',
                            minHeight: 220,
                            maxHeight: 360,
                            display: 'block',
                            objectFit: 'cover',
                            backgroundColor: '#000',
                          }}
                        />
                        {isCameraLoading && (
                          <Box
                            sx={{
                              position: 'absolute',
                              inset: 0,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              bgcolor: 'rgba(0, 0, 0, 0.5)',
                            }}
                          >
                            <CircularProgress />
                          </Box>
                        )}
                      </Box>

                      <Stack direction="row" spacing={1}>
                        <Button
                          variant="contained"
                          onClick={handleStartRecording}
                          disabled={isRecording || isCameraLoading || isSubmitting || videoFile !== null}
                        >
                          Record
                        </Button>
                        <Button variant="outlined" onClick={handleStopRecording} disabled={!isRecording || isSubmitting}>
                          Stop
                        </Button>
                      </Stack>

                      <Typography color="text.secondary" variant="body2">
                        {isRecording ? `Recording: ${formatDuration(recordingDurationSeconds)}` : 'Not recording'}
                      </Typography>
                      <Typography color="text.secondary" variant="body2">
                        {videoFile ? `Recorded: ${videoFile.name}` : 'No recording captured yet'}
                      </Typography>
                      <Typography color="text.secondary" variant="caption">
                        One video per session. Multiple uploads are not allowed.
                      </Typography>
                    </>
                  )}
                </Stack>

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
