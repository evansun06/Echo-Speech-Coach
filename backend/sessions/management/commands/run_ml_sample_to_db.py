from __future__ import annotations

from pathlib import Path
from uuid import UUID

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from ml.pipeline_facade import run_pipeline
from sessions.services import persist_canonical_payload


class Command(BaseCommand):
    help = "Run ML pipeline on sample media files and persist canonical payload into Postgres."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--session-uuid", required=True, type=str)
        parser.add_argument("--audio", type=str, default="ml/evan_test.wav")
        parser.add_argument("--video", type=str, default="ml/evan_test.mp4")
        parser.add_argument("--output-root", type=str, default="ml/run_outputs")
        parser.add_argument("--language-code", type=str, default="en-US")
        parser.add_argument("--sample-rate-hz", type=int, default=16000)
        parser.add_argument("--os-interval-sec", type=float, default=1.0)
        parser.add_argument("--mp-window-sec", type=float, default=1.0)
        parser.add_argument("--dry-run", action="store_true")

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return Path(settings.BASE_DIR) / path

    def handle(self, *args, **options) -> None:
        session_uuid = options["session_uuid"]
        try:
            parsed_uuid = UUID(session_uuid)
        except ValueError as exc:
            raise CommandError("--session-uuid must be a valid UUID") from exc

        audio_path = self._resolve_path(options["audio"])
        video_path = self._resolve_path(options["video"])
        output_root = self._resolve_path(options["output_root"])

        if not audio_path.exists():
            raise CommandError(f"Audio file not found: {audio_path}")
        if not video_path.exists():
            raise CommandError(f"Video file not found: {video_path}")

        self.stdout.write(
            f"Running pipeline for session {parsed_uuid} using:\n"
            f"  audio={audio_path}\n"
            f"  video={video_path}\n"
            f"  output_root={output_root}"
        )

        result = run_pipeline(
            audio_path=str(audio_path),
            video_path=str(video_path),
            session_uuid=str(parsed_uuid),
            language_code=options["language_code"],
            sample_rate_hz=options["sample_rate_hz"],
            os_interval_sec=options["os_interval_sec"],
            mp_window_sec=options["mp_window_sec"],
            output_root=str(output_root),
        )
        self.stdout.write(self.style.SUCCESS(f"Pipeline completed. run_dir={result['run_dir']}"))

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry-run mode enabled; skipping DB write."))
            return

        write_result = persist_canonical_payload(
            canonical_payload=result["canonical_payload"],
            skip_missing_session=True,
        )

        if not write_result["written"]:
            self.stdout.write(
                self.style.WARNING(
                    f"Skipped DB write for session {write_result['session_uuid']}: "
                    f"{write_result['reason']}"
                )
            )
            return

        self.stdout.write(self.style.SUCCESS("DB write completed."))
        self.stdout.write(
            "rows: "
            f"overall={write_result['overall_rows']} "
            f"aligned={write_result['aligned_rows']} "
            f"events={write_result['event_rows']}"
        )
