# Echo Spech Coach

Echo Speech Coach is a full-stack presentation coaching application that turns a recorded speaking session into a reviewable timeline of multimodal signals, structured coaching output, and grounded follow-up chat. It was built to show how a modern AI product can combine browser media capture, asynchronous ML analysis, LLM orchestration, and a clean review workflow in one cohesive system.

## Overview

The application is centered on a single presentation session. A user can create a session, upload or record a video, optionally attach supporting PDFs and speaker context, start analysis, and then review the result through a status-aware dashboard that unlocks the timeline first and chat second.

What the product does today:

- Creates authenticated coaching sessions with a server-authoritative lifecycle
- Supports video upload or in-browser recording for a single presentation
- Accepts up to 3 supplementary PDFs plus free-text speaker context
- Runs asynchronous multimodal analysis over the uploaded session
- Generates timestamped timeline events from audio and visual signals
- Produces coach progress with per-window notes and a final reconciliation
- Enables session-specific follow-up chat after coach output is ready
- Streams the stored session video back for review in the dashboard

## 🏗️ Tech Stack

### Frontend

- React 19
- TypeScript
- Vite
- MUI

### Backend

- Django 6
- Django REST Framework
- Python 3.12
- DRF Spectacular / Swagger for API schema docs

### Async, Data, and Infrastructure

- Celery for background orchestration
- Redis for broker/cache and live-ledger backing
- PostgreSQL as the system of record
- Docker and Docker Compose for local runtime

### ML and Analysis

- Google Cloud Speech-to-Text for word-level transcription
- OpenSMILE for interval-based acoustic features
- MediaPipe for frame/window-level visual signals
- OpenCV, pandas, and numpy for extraction, aggregation, and feature fusion

### LLM Orchestration

- LangGraph for workflow composition
- LangChain Google GenAI integration
- Gemini models for coach reasoning and chat

### Powered by Google

- Google Cloud Speech-to-Text powers the transcription layer used by the ML pipeline.
- Gemini powers the reasoning layer used for window-level coaching notes, final reconciliation, and chat responses.

## ⚙️ System Architecture

Speech Coach is implemented as a monolithic web app with clear internal boundaries. The frontend owns the session workflow and review experience, Django owns session state and API contracts, Celery workers own asynchronous processing, and PostgreSQL/Redis provide the persistence and orchestration backbone.

<p align="center">
  <img src="assets/Echo%20System%20Architecture.drawio.svg" alt="Speech Coach system architecture diagram" width="100%" />
</p>

Core components:

- **React frontend**: Handles authentication flow, session creation, recording/upload UX, dashboard review, timeline display, coach-progress panels, and chat.
- **Django API**: Exposes session, auth, timeline, video-stream, and chat endpoints while enforcing lifecycle transitions such as `draft`, `queued_ml`, `ml_ready`, `processing_coach`, and `ready`.
- **ML package (`backend/ml`)**: Runs the phase-1 analysis pipeline that extracts audio, computes speech and body-language features, aligns them to the transcript, and writes canonical artifacts back to the database.
- **LLM package (`backend/llm`)**: Runs the phase-2 coaching workflow that transforms ML events into window-level notes and a final coaching synthesis.
- **Celery + Redis**: Execute the asynchronous workload so analysis can continue outside the request cycle and session progress can be tracked over time.
- **PostgreSQL**: Stores sessions, orchestration runs, ledger entries, timeline events, and aligned analysis artifacts.

This separation keeps the application simple enough to ship quickly while still showing production-minded boundaries between UI, API, ML processing, and reasoning orchestration.

## 📊 Data Processing Pipeline: ML -> LLM

The product is intentionally split into two phases. The first phase creates objective timeline-ready artifacts from the video; the second phase turns those artifacts into coaching output.

<p align="center">
  <img src="assets/Echo%20Data%20Analysis.drawio.svg" alt="Speech Coach data analysis pipeline diagram" width="100%" />
</p>

### Phase 1: ML analysis

1. **Media preparation**
   The backend extracts audio from the uploaded session video and prepares the audio/video inputs used by the rest of the pipeline.

2. **Speech transcription**
   Google Cloud Speech-to-Text generates a word-level transcript with timestamps, durations, and confidence scores.

3. **Audio feature extraction**
   OpenSMILE computes interval-based acoustic features such as voiced ratio, pitch, loudness, and spectral flux.

4. **Visual feature extraction**
   MediaPipe processes frames from the presentation video to estimate attention, posture deviation, hand motion, fidgeting, expressiveness, and related visual signals.

5. **Feature fusion**
   The pipeline aligns transcript words with the OpenSMILE intervals and MediaPipe windows so each word can be analyzed in context with both audio and visual signals.

6. **Event generation**
   Deterministic event rules convert aligned signals into timestamped events such as hesitation, stuttering, rushing, slow delivery, engaged presence, expressive moments, and steady pacing.

7. **Persistence**
   The backend writes overall metrics, aligned word rows, and session events into the database so the timeline can be rendered independently of the coach phase.

### Phase 2: LLM orchestration

Once ML artifacts exist, the application builds fixed time windows from the canonical payload and sends each window to a subagent reasoning step. Each window includes the local `events` plus a `word_map`, which keeps the model grounded in exact timestamps and transcript spans instead of free-form summaries.

#### Subagent model

The subagent role is responsible for localized reasoning. It looks at one short window at a time, reads only the events and transcript context for that window, and emits notes tied back to specific `event_id` values so the reasoning stays auditable.

#### Flagship reconciliation model

The final flagship role is responsible for session-level synthesis. After window processing is finalized, it reads the accumulated ledger and produces the final reconciliation payload: overall impression, strengths, improvements, and priority actions.

#### Memory ledger

The ledger is the shared append-only memory surface for the coaching phase, and it stores model outputs rather than raw ML events. Subagents write localized notes into the live ledger as they finish each window, and those entries are then finalized into persistent run history. The final flagship model reads that accumulated ledger to generate a session-level coaching summary and provide a grounded evidence trail for downstream chat responses.

This two-phase design is one of the strongest parts of the project: the system can expose objective timeline artifacts as soon as ML finishes, while the higher-level coaching layer continues asynchronously on top of the same evidence base.

## Setup

For environment setup and local run instructions, see [SETUP.md](./SETUP.md).


## The Team

- [Evan Sun](https://github.com/evansun06) 
- [Ryan Liu](https://github.com/lyanriu8)
- [Paul Xu](https://github.com/paulxu6004)