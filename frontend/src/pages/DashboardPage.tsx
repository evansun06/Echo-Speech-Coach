import { useEffect, useRef, useState } from 'react'
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
  Fade,
  LinearProgress,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import { useNavigate, useParams } from 'react-router-dom'
import api from '../api'
import type {
  Annotation,
  ApiError,
  ChatMessage,
  CoachAgentProgress,
  CoachFinalReconciliation,
  CoachProgress,
  CoachingSessionDetail,
  SessionStatus,
} from '../api'

const TERMINAL_STATUSES: SessionStatus[] = ['ready', 'failed', 'coach_failed']
const TIMELINE_VISIBLE_STATUSES: SessionStatus[] = ['ml_ready', 'processing_coach', 'ready', 'coach_failed']
const CONFIDENCE_COLOR = {
  low: '#ef4444',
  medium: '#f59e0b',
  high: '#22c55e',
} as const
const CONFIDENCE_BADGE_BG = {
  low: 'rgba(239, 68, 68, 0.22)',
  medium: 'rgba(245, 158, 11, 0.22)',
  high: 'rgba(34, 197, 94, 0.22)',
} as const
const STAGE_STATUS_COLOR = {
  pending: '#6b7280',
  processing: '#f59e0b',
  completed: '#22c55e',
  failed: '#ef4444',
} as const
const TIMELINE_LANE_HEIGHT = 20
const TIMELINE_LANE_GAP = 8
const TIMELINE_VERTICAL_PADDING = 8
const TIMELINE_BAR_HEIGHT = 14

type LeftPanelTab = 'coach' | 'chat'
type ConfidenceLevel = 'low' | 'medium' | 'high'
type AnnotationSeverity = 'low' | 'medium' | 'high'

type StackedTimelineAnnotation = {
  annotation: Annotation
  laneIndex: number
  resolvedConfidence: number
  confidenceLevel: ConfidenceLevel
}

type ChatUiMessage = ChatMessage & {
  reasoning?: string
  reasoningCollapsed?: boolean
}

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

function clampUnitInterval(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function parseUnknownNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }

  if (typeof value === 'string') {
    const parsedValue = Number(value)
    if (Number.isFinite(parsedValue)) {
      return parsedValue
    }
  }

  return null
}

function parseAnnotationSeverity(value: unknown): AnnotationSeverity | null {
  if (typeof value !== 'string') {
    return null
  }

  const normalized = value.trim().toLowerCase()
  if (normalized === 'low' || normalized === 'medium' || normalized === 'high') {
    return normalized
  }

  return null
}

function resolveAnnotationConfidence(annotation: Annotation): number {
  const metadataConfidence = parseUnknownNumber(annotation.metadata.confidence)
  if (metadataConfidence !== null) {
    return clampUnitInterval(metadataConfidence)
  }

  const fallbackConfidence = parseUnknownNumber(annotation.confidence)
  if (fallbackConfidence !== null) {
    return clampUnitInterval(fallbackConfidence)
  }

  return 0
}

function resolveAnnotationSeverity(annotation: Annotation): AnnotationSeverity | null {
  return parseAnnotationSeverity(annotation.metadata.severity)
}

function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.8) {
    return 'high'
  }

  if (confidence >= 0.6) {
    return 'medium'
  }

  return 'low'
}

function formatConfidencePercent(confidence: number): string {
  return `${Math.round(clampUnitInterval(confidence) * 100)}%`
}

function buildStackedTimelineAnnotations(annotations: Annotation[]): StackedTimelineAnnotation[] {
  const sortedAnnotations = [...annotations].sort((left, right) => {
    if (left.start_ms !== right.start_ms) {
      return left.start_ms - right.start_ms
    }

    if (left.end_ms !== right.end_ms) {
      return left.end_ms - right.end_ms
    }

    return left.id.localeCompare(right.id)
  })

  const laneEndTimes: number[] = []
  const stackedAnnotations: StackedTimelineAnnotation[] = []

  for (const annotation of sortedAnnotations) {
    const startMs = Math.max(0, annotation.start_ms)
    const endMs = Math.max(startMs, annotation.end_ms)
    let laneIndex = laneEndTimes.findIndex((laneEndMs) => laneEndMs <= startMs)

    if (laneIndex === -1) {
      laneIndex = laneEndTimes.length
      laneEndTimes.push(endMs)
    } else {
      laneEndTimes[laneIndex] = endMs
    }

    const resolvedConfidence = resolveAnnotationConfidence(annotation)
    stackedAnnotations.push({
      annotation,
      laneIndex,
      resolvedConfidence,
      confidenceLevel: getConfidenceLevel(resolvedConfidence),
    })
  }

  return stackedAnnotations
}

function formatChatTimestamp(value: string): string {
  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function isTerminalStatus(status: SessionStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}

function shouldShowTimeline(status: SessionStatus): boolean {
  return TIMELINE_VISIBLE_STATUSES.includes(status)
}

function sortAgentProgressChronologically(agentProgress: CoachAgentProgress[]): CoachAgentProgress[] {
  return [...agentProgress].sort((left, right) => {
    const leftStart = left.window_start_ms === null ? Number.POSITIVE_INFINITY : left.window_start_ms
    const rightStart = right.window_start_ms === null ? Number.POSITIVE_INFINITY : right.window_start_ms
    if (leftStart !== rightStart) {
      return leftStart - rightStart
    }

    const leftEnd = left.window_end_ms === null ? Number.POSITIVE_INFINITY : left.window_end_ms
    const rightEnd = right.window_end_ms === null ? Number.POSITIVE_INFINITY : right.window_end_ms
    if (leftEnd !== rightEnd) {
      return leftEnd - rightEnd
    }

    return left.execution_index - right.execution_index
  })
}

function ReconciliationHero({ finalReconciliation }: { finalReconciliation: CoachFinalReconciliation }) {
  return (
    <Card
      variant="outlined"
      sx={{
        borderColor: '#1d4ed8',
        bgcolor: 'transparent',
        background: 'linear-gradient(135deg, #0b1220 0%, #1e293b 100%)',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Stack spacing={1.25}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
            <Typography variant="subtitle1" sx={{ color: '#f8fafc', fontWeight: 700 }}>
              Final Reconciliation
            </Typography>
            <Chip
              size="small"
              label={finalReconciliation.model_name || finalReconciliation.agent_name}
              sx={{ bgcolor: 'rgba(59,130,246,0.18)', color: '#dbeafe', border: '1px solid rgba(96,165,250,0.45)' }}
            />
          </Stack>
          <Typography variant="body2" sx={{ color: '#e2e8f0' }}>
            {finalReconciliation.overall_impression}
          </Typography>
          {finalReconciliation.priority_actions.length > 0 && (
            <Stack spacing={0.5}>
              <Typography variant="caption" sx={{ color: '#93c5fd', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Priority Actions
              </Typography>
              {finalReconciliation.priority_actions.slice(0, 3).map((action, index) => (
                <Typography key={`${action}-${index}`} variant="body2" sx={{ color: '#e2e8f0' }}>
                  {index + 1}. {action}
                </Typography>
              ))}
            </Stack>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

function AgentReasoningCard({ agent }: { agent: CoachAgentProgress }) {
  const [showEvents, setShowEvents] = useState(false)
  const isPending = agent.status === 'pending'
  const isProcessing = agent.status === 'processing'
  const isWaiting = isPending || isProcessing
  const isCompleted = agent.status === 'completed'
  const hasReasoningEvents = agent.reasoning_events.length > 0
  const impression = agent.window_impression?.body?.trim() || ''

  return (
    <Card variant="outlined">
      <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
        <Stack spacing={1}>
          <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={1}>
            <Stack spacing={0.5}>
              <Stack direction="row" spacing={0.75} alignItems="center" useFlexGap flexWrap="wrap">
                <Chip size="small" label={agent.model_name || 'model'} />
                <Chip size="small" variant="outlined" label={agent.window_label} />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                {agent.agent_name}
              </Typography>
            </Stack>
            <Chip
              size="small"
              label={formatStatusLabel(agent.status)}
              sx={{
                bgcolor: `${STAGE_STATUS_COLOR[agent.status]}22`,
                border: '1px solid',
                borderColor: `${STAGE_STATUS_COLOR[agent.status]}66`,
              }}
            />
          </Stack>

          {isWaiting && (
            <Stack spacing={0.75}>
              <LinearProgress />
              <Typography variant="body2" color="text.secondary">
                {isProcessing ? 'Reasoning in progress...' : 'Queued...'}
              </Typography>
            </Stack>
          )}

          {isCompleted && (
            <Stack spacing={0.75}>
              <Typography variant="body2">{impression || 'No impression captured for this window yet.'}</Typography>
              {hasReasoningEvents && (
                <Stack spacing={0.5}>
                  <Button size="small" onClick={() => setShowEvents((previous) => !previous)} sx={{ alignSelf: 'flex-start', px: 0, minWidth: 0 }}>
                    {showEvents ? 'Hide reasoning events' : 'View reasoning events'}
                  </Button>
                  <Collapse in={showEvents}>
                    <Stack spacing={0.75}>
                      {agent.reasoning_events.map((event) => (
                        <Box key={event.note_id} sx={{ p: 1, borderRadius: 1, bgcolor: 'grey.50', border: '1px solid', borderColor: 'divider' }}>
                          <Typography variant="caption" color="text.secondary">
                            {event.title}
                          </Typography>
                          <Typography variant="body2">{event.body}</Typography>
                          {event.evidence_refs.length > 0 && (
                            <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                              {event.evidence_refs.map((evidenceRef) => (
                                <Chip key={`${event.note_id}-${evidenceRef}`} size="small" label={evidenceRef} />
                              ))}
                            </Stack>
                          )}
                        </Box>
                      ))}
                    </Stack>
                  </Collapse>
                </Stack>
              )}
            </Stack>
          )}

          {agent.status === 'failed' && (
            <Alert severity="warning" sx={{ py: 0 }}>
              This window failed before an impression was generated.
            </Alert>
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

function CoachPanelContent({
  session,
  coachProgress,
  onRetry,
  isRetrying,
  retryError,
  showReadyTransition,
}: {
  session: CoachingSessionDetail
  coachProgress: CoachProgress | null
  onRetry: () => void
  isRetrying: boolean
  retryError: string | null
  showReadyTransition: boolean
}) {
  const finalReconciliation = coachProgress?.final_reconciliation ?? null
  const orderedSubagents = sortAgentProgressChronologically(
    (coachProgress?.agent_progress ?? []).filter((agent) => agent.agent_kind === 'subagent'),
  )

  return (
    <Stack spacing={1.5}>
      {session.status === 'ready' && (
        <Fade in={showReadyTransition} timeout={600}>
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            sx={{
              py: 0.5,
              px: 1,
              borderRadius: 1,
              bgcolor: 'success.50',
              border: '1px solid',
              borderColor: 'success.200',
            }}
          >
            <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'success.main' }} />
            <Typography variant="caption" color="success.dark">
              Analysis complete
            </Typography>
          </Stack>
        </Fade>
      )}

      {session.status === 'coach_failed' && (
        <Alert
          severity="warning"
          action={
            <Button size="small" onClick={onRetry} disabled={isRetrying}>
              {isRetrying ? 'Retrying...' : 'Retry'}
            </Button>
          }
        >
          Coach review unavailable.
        </Alert>
      )}

      {retryError && <Alert severity="error">{retryError}</Alert>}

      {!coachProgress ? (
        <Alert severity="info">Coach progress is not available yet.</Alert>
      ) : (
        <Stack spacing={1.25}>
          {finalReconciliation && <ReconciliationHero finalReconciliation={finalReconciliation} />}

          <Stack spacing={0.5}>
            <Typography variant="h6">Reasoning Timeline</Typography>
            <Typography variant="body2" color="text.secondary">
              Windows are always rendered in chronological order.
            </Typography>
          </Stack>

          {orderedSubagents.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              Waiting for subagent windows...
            </Typography>
          ) : (
            <Box
              sx={{
                maxHeight: 420,
                overflowY: 'auto',
                pr: 0.5,
              }}
            >
              <Stack spacing={1}>
                {orderedSubagents.map((agent) => (
                  <AgentReasoningCard key={agent.agent_execution_id} agent={agent} />
                ))}
              </Stack>
            </Box>
          )}
        </Stack>
      )}
    </Stack>
  )
}

function ChatPanel({ sessionId, sessionStatus }: { sessionId: string; sessionStatus: SessionStatus }) {
  const [chatMessages, setChatMessages] = useState<ChatUiMessage[]>([])
  const [isHistoryLoading, setIsHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [draftMessage, setDraftMessage] = useState('')
  const [sendError, setSendError] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [streamingReasoning, setStreamingReasoning] = useState('')
  const [streamingAnswer, setStreamingAnswer] = useState('')
  const [streamingReasoningCollapsed, setStreamingReasoningCollapsed] = useState(false)

  const isLocked = sessionStatus !== 'ready'

  useEffect(() => {
    let isMounted = true

    const loadChatHistory = async () => {
      setIsHistoryLoading(true)
      setHistoryError(null)

      try {
        const history = await api.chat.getHistory(sessionId)
        if (isMounted) {
          setChatMessages(history)
        }
      } catch (historyLoadError) {
        const apiError = historyLoadError as ApiError
        if (isMounted) {
          setHistoryError(apiError.message || 'Failed to load chat history.')
        }
      } finally {
        if (isMounted) {
          setIsHistoryLoading(false)
        }
      }
    }

    void loadChatHistory()

    return () => {
      isMounted = false
    }
  }, [sessionId])

  const toggleMessageReasoning = (messageId: string) => {
    setChatMessages((previous) =>
      previous.map((message) =>
        message.id === messageId
          ? {
              ...message,
              reasoningCollapsed: !(message.reasoningCollapsed ?? true),
            }
          : message,
      ),
    )
  }

  const handleSendMessage = async () => {
    if (isLocked || isSending) {
      return
    }

    const trimmedMessage = draftMessage.trim()
    if (!trimmedMessage) {
      return
    }

    const userMessage: ChatUiMessage = {
      id: `local-user-${Date.now()}`,
      role: 'user',
      content: trimmedMessage,
      created_at: new Date().toISOString(),
    }

    setChatMessages((previous) => [...previous, userMessage])
    setDraftMessage('')
    setSendError(null)
    setIsSending(true)
    setStreamingReasoning('')
    setStreamingAnswer('')
    setStreamingReasoningCollapsed(false)

    let reasoningBuffer = ''
    let answerBuffer = ''

    try {
      const response = await api.chat.sendMessage(sessionId, trimmedMessage)

      await api.chat.streamResponse(sessionId, response.response_id, {
        onReasoningToken: (token) => {
          reasoningBuffer += token
          setStreamingReasoning((previous) => previous + token)
        },
        onAnswerToken: (token) => {
          answerBuffer += token
          setStreamingAnswer((previous) => previous + token)
        },
        onError: (message) => {
          setSendError(message)
        },
      })

      const assistantMessage: ChatUiMessage = {
        id: `local-assistant-${Date.now()}`,
        role: 'assistant',
        content: answerBuffer.trim() || 'No final answer generated.',
        created_at: new Date().toISOString(),
        reasoning: reasoningBuffer.trim() || undefined,
        reasoningCollapsed: true,
      }

      setChatMessages((previous) => [...previous, assistantMessage])
      setStreamingReasoning('')
      setStreamingAnswer('')
      setStreamingReasoningCollapsed(true)
    } catch (sendMessageError) {
      const apiError = sendMessageError as ApiError
      setSendError(apiError.message || 'Failed to send message.')
    } finally {
      setIsSending(false)
    }
  }

  const hasStreamingReasoning = streamingReasoning.trim().length > 0
  const showStreamingAssistant = isSending && streamingAnswer.length > 0

  return (
    <Stack spacing={1.5} sx={{ height: '100%' }}>
      {isLocked && <Alert severity="info">Chat unlocks when analysis is complete.</Alert>}
      {historyError && <Alert severity="warning">{historyError}</Alert>}
      {sendError && <Alert severity="error">{sendError}</Alert>}

      <Box
        sx={{
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1.5,
          p: 1.25,
          minHeight: 260,
          maxHeight: 320,
          overflowY: 'auto',
          bgcolor: 'grey.50',
        }}
      >
        {isHistoryLoading ? (
          <Stack direction="row" spacing={1} alignItems="center">
            <CircularProgress size={16} />
            <Typography variant="body2" color="text.secondary">
              Loading chat history...
            </Typography>
          </Stack>
        ) : chatMessages.length === 0 && !showStreamingAssistant ? (
          <Typography variant="body2" color="text.secondary">
            No messages yet.
          </Typography>
        ) : (
          <Stack spacing={1.25}>
            {chatMessages.map((message) => {
              const isAssistant = message.role === 'assistant'

              return (
                <Card
                  key={message.id}
                  variant="outlined"
                  sx={{
                    alignSelf: isAssistant ? 'stretch' : 'flex-end',
                    maxWidth: isAssistant ? '100%' : '88%',
                    bgcolor: isAssistant ? 'common.white' : 'primary.50',
                  }}
                >
                  <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
                    <Stack spacing={0.75}>
                      <Typography variant="caption" color="text.secondary">
                        {isAssistant ? 'Coach' : 'You'} • {formatChatTimestamp(message.created_at)}
                      </Typography>

                      {isAssistant && message.reasoning && (
                        <Stack spacing={0.5}>
                          <Button
                            size="small"
                            onClick={() => toggleMessageReasoning(message.id)}
                            sx={{ alignSelf: 'flex-start', px: 0, minWidth: 0 }}
                          >
                            {message.reasoningCollapsed ?? true ? 'Show Reasoning…' : 'Hide Reasoning…'}
                          </Button>
                          <Collapse in={!(message.reasoningCollapsed ?? true)}>
                            <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                              {message.reasoning}
                            </Typography>
                          </Collapse>
                        </Stack>
                      )}

                      <Typography variant="body2">{message.content}</Typography>
                    </Stack>
                  </CardContent>
                </Card>
              )
            })}

            {showStreamingAssistant && (
              <Card variant="outlined" sx={{ bgcolor: 'common.white' }}>
                <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
                  <Stack spacing={0.75}>
                    <Typography variant="caption" color="text.secondary">
                      Coach • {formatChatTimestamp(new Date().toISOString())}
                    </Typography>

                    {hasStreamingReasoning && (
                      <Stack spacing={0.5}>
                        <Button
                          size="small"
                          onClick={() => setStreamingReasoningCollapsed((previous) => !previous)}
                          sx={{ alignSelf: 'flex-start', px: 0, minWidth: 0 }}
                        >
                          {streamingReasoningCollapsed ? 'Show Reasoning…' : 'Hide Reasoning…'}
                        </Button>
                        <Collapse in={!streamingReasoningCollapsed}>
                          <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>
                            {streamingReasoning}
                          </Typography>
                        </Collapse>
                      </Stack>
                    )}

                    <Typography variant="body2">{streamingAnswer || 'Generating final answer...'}</Typography>
                  </Stack>
                </CardContent>
              </Card>
            )}
          </Stack>
        )}
      </Box>

      <Stack direction="row" spacing={1.25} alignItems="flex-end">
        <TextField
          label="Ask Coach"
          placeholder={isLocked ? 'Chat unlocks when analysis is complete' : 'Ask a follow-up question...'}
          multiline
          maxRows={4}
          value={draftMessage}
          onChange={(event: React.ChangeEvent<HTMLInputElement>) => setDraftMessage(event.target.value)}
          disabled={isLocked || isSending}
          fullWidth
        />
        <Button variant="contained" onClick={handleSendMessage} disabled={isLocked || isSending || !draftMessage.trim()}>
          {isSending ? 'Sending...' : 'Send'}
        </Button>
      </Stack>
    </Stack>
  )
}

function VideoPlayer({
  videoUrl,
  videoRef,
  onTimeUpdate,
}: {
  videoUrl: string | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  onTimeUpdate: () => void
}) {
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
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                preload="metadata"
                onTimeUpdate={onTimeUpdate}
                style={{ width: '100%', display: 'block' }}
              />
            ) : (
              <Box
                sx={{
                  minHeight: 220,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  px: 2,
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Video will appear once upload is complete.
                </Typography>
              </Box>
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  )
}

function AnnotationTimeline({
  status,
  annotations,
  timelineError,
  videoRef,
  currentMs,
}: {
  status: SessionStatus
  annotations: Annotation[]
  timelineError: string | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  currentMs: number
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

  const maxEndMs = Math.max(
    1,
    ...annotations.map((annotation) => Math.max(annotation.start_ms, annotation.end_ms))
  )

  return (
    <Card variant="outlined">
      <CardContent sx={{ p: 2.5 }}>
        <Stack spacing={2}>
          <Typography variant="h6">Annotation Timeline</Typography>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Typography variant="caption" color="text.secondary">
              Confidence
            </Typography>
            {(['high', 'medium', 'low'] as const).map((level) => (
              <Stack direction="row" spacing={0.5} alignItems="center" key={level}>
                <Box
                  sx={{
                    width: 10,
                    height: 10,
                    borderRadius: '2px',
                    bgcolor: CONFIDENCE_COLOR[level],
                  }}
                />
                <Typography variant="caption" color="text.secondary">
                  {level}
                </Typography>
              </Stack>
            ))}
          </Stack>
          {timelineError && <Alert severity="warning">{timelineError}</Alert>}

          {annotations.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No annotations available yet.
            </Typography>
          ) : (
            <TimelineTrack
              label="Events"
              annotations={annotations}
              maxEndMs={maxEndMs}
              totalDurationMs={maxEndMs}
              currentMs={currentMs}
              onSeek={(ms) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = ms / 1000
                }
              }}
            />
          )}
        </Stack>
      </CardContent>
    </Card>
  )
}

function TimelineTrack({
  label,
  annotations,
  maxEndMs,
  totalDurationMs,
  currentMs,
  onSeek,
}: {
  label: string
  annotations: Annotation[]
  maxEndMs: number
  totalDurationMs: number
  currentMs: number
  onSeek: (ms: number) => void
}) {
  const stackedAnnotations = buildStackedTimelineAnnotations(annotations)
  const laneCount = Math.max(1, ...stackedAnnotations.map((item) => item.laneIndex + 1))
  const trackHeight =
    TIMELINE_VERTICAL_PADDING * 2 +
    laneCount * TIMELINE_LANE_HEIGHT +
    Math.max(0, laneCount - 1) * TIMELINE_LANE_GAP
  const playheadPosition = Math.min(100, Math.max(0, (currentMs / totalDurationMs) * 100))

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
          height: trackHeight,
          borderRadius: 1.5,
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: 'grey.100',
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            width: '2px',
            bgcolor: 'primary.main',
            left: `${playheadPosition}%`,
            pointerEvents: 'none',
            zIndex: 2,
          }}
        />
        {stackedAnnotations.map((stackedAnnotation) => {
          const { annotation, confidenceLevel, laneIndex, resolvedConfidence } = stackedAnnotation
          const startMs = Math.max(0, annotation.start_ms)
          const endMs = Math.max(startMs, annotation.end_ms)
          const left = Math.min(100, Math.max(0, (startMs / totalDurationMs) * 100))
          const durationWidth = Math.max(0.8, ((endMs - startMs) / totalDurationMs) * 100)
          const confidenceColor = CONFIDENCE_COLOR[confidenceLevel]
          const confidenceLabel = formatStatusLabel(confidenceLevel)
          const severity = resolveAnnotationSeverity(annotation)
          const severityLabel = severity ? formatStatusLabel(severity) : null
          const top =
            TIMELINE_VERTICAL_PADDING +
            laneIndex * (TIMELINE_LANE_HEIGHT + TIMELINE_LANE_GAP) +
            (TIMELINE_LANE_HEIGHT - TIMELINE_BAR_HEIGHT) / 2

          return (
            <Tooltip
              key={annotation.id}
              title={
                <Stack spacing={0.25}>
                  <Typography variant="caption" sx={{ color: 'inherit', fontWeight: 600 }}>
                    {formatTimestamp(startMs)} - {formatTimestamp(endMs)}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'inherit' }}>
                    {annotation.summary}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'inherit' }}>
                    Source: {annotation.source}
                    {severityLabel ? ` • Severity: ${severityLabel}` : ''}
                    {' • '}Confidence: {confidenceLabel} ({formatConfidencePercent(resolvedConfidence)})
                  </Typography>
                </Stack>
              }
              placement="top"
              arrow
            >
              <Box
                onClick={() => onSeek(startMs)}
                sx={{
                  position: 'absolute',
                  left: `${left}%`,
                  top,
                  width: `${durationWidth}%`,
                  height: TIMELINE_BAR_HEIGHT,
                  borderRadius: '3px',
                  bgcolor: confidenceColor,
                  display: 'flex',
                  alignItems: 'center',
                  px: 0.5,
                  overflow: 'hidden',
                  cursor: 'pointer',
                }}
              >
                <Box
                  sx={{
                    px: 0.5,
                    borderRadius: '999px',
                    bgcolor: CONFIDENCE_BADGE_BG[confidenceLevel],
                    color: confidenceColor,
                    border: '1px solid',
                    borderColor: confidenceColor,
                    fontSize: 9,
                    fontWeight: 700,
                    lineHeight: 1.3,
                    textTransform: 'uppercase',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    maxWidth: '100%',
                  }}
                >
                  <Typography
                    component="span"
                    variant="caption"
                    sx={{
                      display: 'block',
                      color: 'inherit',
                      fontSize: 'inherit',
                      fontWeight: 'inherit',
                      lineHeight: 'inherit',
                      textTransform: 'inherit',
                      whiteSpace: 'inherit',
                      overflow: 'inherit',
                      textOverflow: 'inherit',
                    }}
                  >
                    {confidenceLabel}
                  </Typography>
                </Box>
              </Box>
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
  const videoRef = useRef<HTMLVideoElement>(null)
  const [currentMs, setCurrentMs] = useState(0)

  const [session, setSession] = useState<CoachingSessionDetail | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [timelineError, setTimelineError] = useState<string | null>(null)
  const [retryError, setRetryError] = useState<string | null>(null)

  const [isRetryingCoach, setIsRetryingCoach] = useState(false)
  const [pollRestartKey, setPollRestartKey] = useState(0)
  const [forceCoachFailedPolling, setForceCoachFailedPolling] = useState(false)

  const [leftPanelTab, setLeftPanelTab] = useState<LeftPanelTab>('coach')
  const [showReadyTransition, setShowReadyTransition] = useState(false)
  const previousStatus = useRef<SessionStatus | null>(null)

  useEffect(() => {
    if (!session) {
      return
    }

    if (previousStatus.current !== 'ready' && session.status === 'ready') {
      setShowReadyTransition(true)
      const timeoutId = setTimeout(() => {
        setShowReadyTransition(false)
      }, 1200)

      previousStatus.current = session.status
      return () => {
        clearTimeout(timeoutId)
      }
    }

    previousStatus.current = session.status
  }, [session])

  useEffect(() => {
    if (!id) {
      setError('Missing session id.')
      setIsLoading(false)
      return
    }

    let isMounted = true
    let intervalId: ReturnType<typeof setInterval> | null = null

    const shouldStopPolling = (status: SessionStatus): boolean => {
      if (status === 'coach_failed') {
        return !forceCoachFailedPolling
      }

      return isTerminalStatus(status)
    }

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
      if (!isMounted || !initialStatus || shouldStopPolling(initialStatus)) {
        return
      }

      intervalId = setInterval(() => {
        void loadSession(false).then((status) => {
          if (!status) {
            return
          }

          if (forceCoachFailedPolling && status !== 'coach_failed') {
            setForceCoachFailedPolling(false)
          }

          if (shouldStopPolling(status)) {
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
  }, [forceCoachFailedPolling, id, pollRestartKey])

  const handleRetryCoach = async () => {
    if (!session) {
      return
    }

    setIsRetryingCoach(true)
    setRetryError(null)

    try {
      await api.sessions.startAnalysis(session.id)
      setForceCoachFailedPolling(true)
      setPollRestartKey((previous) => previous + 1)
    } catch (retryRequestError) {
      const apiError = retryRequestError as ApiError
      setRetryError(apiError.message || 'Failed to retry analysis.')
    } finally {
      setIsRetryingCoach(false)
    }
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
              gridTemplateColumns: { xs: '1fr', lg: '380px 1fr' },
              gridTemplateAreas: {
                xs: '"left" "video"',
                lg: '"left video"',
              },
            }}
          >
            <Box sx={{ gridArea: 'left' }}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Stack spacing={1.5}>
                    <Tabs
                      value={leftPanelTab}
                      onChange={(_event: React.SyntheticEvent, nextTab: LeftPanelTab) => setLeftPanelTab(nextTab)}
                      variant="fullWidth"
                      sx={{ minHeight: 40 }}
                    >
                      <Tab label="Coach" value="coach" />
                      <Tab label="Chat" value="chat" />
                    </Tabs>

                    {leftPanelTab === 'coach' ? (
                      <CoachPanelContent
                        session={session}
                        coachProgress={session.coach_progress}
                        onRetry={handleRetryCoach}
                        isRetrying={isRetryingCoach}
                        retryError={retryError}
                        showReadyTransition={showReadyTransition}
                      />
                    ) : (
                      <ChatPanel sessionId={session.id} sessionStatus={session.status} />
                    )}
                  </Stack>
                </CardContent>
              </Card>
            </Box>

            <Box sx={{ gridArea: 'video' }}>
              <Stack spacing={2}>
                <VideoPlayer
                  videoUrl={session.video_file_url}
                  videoRef={videoRef}
                  onTimeUpdate={() => {
                    if (videoRef.current) {
                      setCurrentMs(videoRef.current.currentTime * 1000)
                    }
                  }}
                />
                <AnnotationTimeline
                  status={session.status}
                  annotations={annotations}
                  timelineError={timelineError}
                  videoRef={videoRef}
                  currentMs={currentMs}
                />
              </Stack>
            </Box>
          </Box>
        )}
      </Stack>
    </Container>
  )
}

export default DashboardPage
