import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from content_extraction import CodexLLMClient, extract_key_concepts, save_concepts
from segment_transcript import segment_transcript, save_segments


ProgressCallback = Optional[Callable[[str, str], None]]

RUNS_DIR = Path("runs")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "lecture"


def emit_progress(callback: ProgressCallback, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def ensure_run_dir(lecture_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{timestamp}_{slugify(lecture_name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_uploaded_video(upload_name: str, file_bytes: bytes, run_dir: Path) -> Path:
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_path = uploads_dir / Path(upload_name).name
    upload_path.write_bytes(file_bytes)
    return upload_path


def save_uploaded_text(upload_name: str, file_text: str, run_dir: Path) -> Path:
    uploads_dir = run_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_path = uploads_dir / Path(upload_name).name
    upload_path.write_text(file_text, encoding="utf-8")
    return upload_path


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model_size: str = "base",
    language: Optional[str] = "en",
    chunk_minutes: int = 10,
) -> Dict[str, Any]:
    try:
        from preprocessing.transcribe import VideoTranscriber
    except ImportError as exc:
        error_text = str(exc)
        if "audioop" in error_text or "pyaudioop" in error_text:
            raise RuntimeError(
                "The transcription module failed to import because this environment is using Python 3.14, "
                "where `audioop` is no longer available for `pydub`. Run this project in Python 3.11 or 3.12 "
                "instead, then reinstall the requirements. You also still need `ffmpeg` installed and available in PATH."
            ) from exc

        raise RuntimeError(
            "The transcription module could not be imported. Install `openai-whisper`, `pydub`, and `torch`, "
            "and ensure `ffmpeg` is installed and available in PATH. "
            f"Original import error: {exc}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    transcriber = VideoTranscriber(model_size=model_size, language=language or None)
    transcriber.output_dir = output_dir
    result = transcriber.transcribe_video(str(video_path), max_chunk_minutes=chunk_minutes)

    transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
    transcript_json_path = output_dir / f"{video_path.stem}_transcript.json"
    subtitles_path = output_dir / f"{video_path.stem}_subtitles.srt"

    return {
        "result": result,
        "transcript_path": transcript_path,
        "transcript_json_path": transcript_json_path,
        "subtitles_path": subtitles_path,
    }


def count_total_concepts(concept_entries: List[Dict[str, Any]]) -> int:
    return sum(len(entry.get("concepts", [])) for entry in concept_entries)


def build_pipeline_summary(
    lecture_id: str,
    lecture_title: str,
    run_dir: Path,
    upload_path: Path,
    transcription_result: Dict[str, Any],
    segments: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    segments_path: Path,
    concepts_path: Path,
) -> Dict[str, Any]:
    stats = transcription_result["result"].get("statistics", {})
    transcript_path = transcription_result["transcript_path"]
    transcript_preview = transcript_path.read_text(encoding="utf-8")[:1500]

    return {
        "lecture_id": lecture_id,
        "lecture_title": lecture_title,
        "run_dir": str(run_dir),
        "upload_path": str(upload_path),
        "transcript_path": str(transcript_path),
        "transcript_json_path": str(transcription_result["transcript_json_path"]),
        "subtitles_path": str(transcription_result["subtitles_path"]),
        "segments_path": str(segments_path),
        "concepts_path": str(concepts_path),
        "statistics": {
            "word_count": int(stats.get("word_count", 0)),
            "total_segments": int(stats.get("total_segments", 0)),
            "duration_seconds": float(stats.get("total_duration_seconds", 0)),
            "duration": stats.get("total_duration", "0:00:00"),
            "detected_topics": len({seg.get("topic", "").strip() for seg in segments if seg.get("topic")}),
            "concept_count": count_total_concepts(concepts),
        },
        "topics": [seg.get("topic", "") for seg in segments if seg.get("topic")],
        "transcript_preview": transcript_preview,
        "segments": segments,
        "concepts": concepts,
    }


def save_pipeline_summary(summary: Dict[str, Any], run_dir: Path) -> Path:
    summary_path = run_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def process_uploaded_lecture(
    upload_name: str,
    file_bytes: bytes,
    lecture_title: str,
    lecture_id: Optional[str] = None,
    transcription_model: str = "base",
    language: Optional[str] = "en",
    chunk_minutes: int = 10,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    lecture_title = lecture_title.strip() or Path(upload_name).stem
    lecture_id = lecture_id or slugify(lecture_title)

    emit_progress(progress_callback, "prepare", "Creating run folder and saving the uploaded lecture.")
    run_dir = ensure_run_dir(lecture_title)
    upload_path = save_uploaded_video(upload_name, file_bytes, run_dir)

    emit_progress(progress_callback, "transcribe", "Transcribing the uploaded lecture video.")
    transcription_dir = run_dir / "transcription_output"
    transcription_result = transcribe_video(
        video_path=upload_path,
        output_dir=transcription_dir,
        model_size=transcription_model,
        language=language,
        chunk_minutes=chunk_minutes,
    )

    emit_progress(progress_callback, "segment", "Splitting the transcript into lecture segments.")
    segments_path = run_dir / "segments.json"
    segments = segment_transcript(
        transcript_path=str(transcription_result["transcript_path"]),
        lecture_id=lecture_id,
    )
    save_segments(segments, str(segments_path))

    emit_progress(progress_callback, "concepts", "Extracting key concepts from the segmented transcript.")
    client = CodexLLMClient()
    concepts_path = run_dir / "concepts.json"
    concepts = extract_key_concepts(segments=segments, client=client, sleep_between_calls=0.0)
    save_concepts(concepts, str(concepts_path))

    emit_progress(progress_callback, "finalize", "Saving the integrated lecture artifacts.")
    summary = build_pipeline_summary(
        lecture_id=lecture_id,
        lecture_title=lecture_title,
        run_dir=run_dir,
        upload_path=upload_path,
        transcription_result=transcription_result,
        segments=segments,
        concepts=concepts,
        segments_path=segments_path,
        concepts_path=concepts_path,
    )
    summary["summary_path"] = str(save_pipeline_summary(summary, run_dir))
    return summary


def process_transcript_text(
    transcript_name: str,
    transcript_text: str,
    lecture_title: str,
    lecture_id: Optional[str] = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    lecture_title = lecture_title.strip() or Path(transcript_name).stem
    lecture_id = lecture_id or slugify(lecture_title)

    emit_progress(progress_callback, "prepare", "Creating run folder and saving the transcript input.")
    run_dir = ensure_run_dir(lecture_title)
    transcript_path = save_uploaded_text(transcript_name, transcript_text, run_dir)

    emit_progress(progress_callback, "segment", "Splitting the transcript into lecture segments.")
    segments_path = run_dir / "segments.json"
    segments = segment_transcript(
        transcript_path=str(transcript_path),
        lecture_id=lecture_id,
    )
    save_segments(segments, str(segments_path))

    emit_progress(progress_callback, "concepts", "Extracting key concepts from the segmented transcript.")
    client = CodexLLMClient()
    concepts_path = run_dir / "concepts.json"
    concepts = extract_key_concepts(segments=segments, client=client, sleep_between_calls=0.0)
    save_concepts(concepts, str(concepts_path))

    emit_progress(progress_callback, "finalize", "Saving the integrated lecture artifacts.")
    summary = {
        "lecture_id": lecture_id,
        "lecture_title": lecture_title,
        "run_dir": str(run_dir),
        "upload_path": str(transcript_path),
        "transcript_path": str(transcript_path),
        "transcript_json_path": None,
        "subtitles_path": None,
        "segments_path": str(segments_path),
        "concepts_path": str(concepts_path),
        "statistics": {
            "word_count": len(transcript_text.split()),
            "total_segments": 0,
            "duration_seconds": 0.0,
            "duration": "N/A",
            "detected_topics": len({seg.get("topic", "").strip() for seg in segments if seg.get("topic")}),
            "concept_count": count_total_concepts(concepts),
        },
        "topics": [seg.get("topic", "") for seg in segments if seg.get("topic")],
        "transcript_preview": transcript_text[:1500],
        "segments": segments,
        "concepts": concepts,
    }
    summary["summary_path"] = str(save_pipeline_summary(summary, run_dir))
    return summary
