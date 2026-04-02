import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from lecture_quiz_pipeline import process_transcript_text, process_uploaded_lecture, slugify
from quiz_generation import generate_quiz_bank, save_quiz_bank


st.set_page_config(page_title="Lecture-to-Quiz Studio", page_icon="L", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f5f0e6;
            --panel-bg: rgba(255, 252, 247, 0.86);
            --ink: #172b22;
            --muted: #5d6e63;
            --accent: #1f6b52;
            --accent-soft: #dbe9df;
            --line: rgba(23, 43, 34, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(95, 152, 122, 0.18), transparent 28%),
                radial-gradient(circle at bottom left, rgba(203, 160, 87, 0.16), transparent 26%),
                linear-gradient(180deg, #f9f5ee 0%, var(--page-bg) 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255, 252, 247, 0.94), rgba(235, 244, 237, 0.94));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 40px rgba(45, 74, 58, 0.08);
            margin-bottom: 1.2rem;
        }

        .section-card {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.2rem 1.2rem 0.6rem 1.2rem;
            box-shadow: 0 10px 30px rgba(23, 43, 34, 0.05);
            margin-bottom: 1rem;
        }

        .eyebrow {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    st.session_state.setdefault("pipeline_summary", None)
    st.session_state.setdefault("quiz_bank", None)
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("submitted", False)


def clear_answer_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("quiz_answer_"):
            del st.session_state[key]


def reset_quiz_state() -> None:
    st.session_state.quiz_bank = None
    st.session_state.answers = {}
    st.session_state.submitted = False
    clear_answer_widget_state()


def get_provider_status() -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    key_name = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "nim": "NIM_API_KEY",
    }.get(provider, "API_KEY")
    return {
        "provider": provider,
        "key_name": key_name,
        "configured": bool(os.getenv(key_name)),
    }


def render_sidebar() -> None:
    status = get_provider_status()
    st.sidebar.header("Runtime")
    st.sidebar.caption(f"LLM provider: `{status['provider']}`")
    st.sidebar.caption(
        f"API key status: {'configured' if status['configured'] else 'missing'} (`{status['key_name']}`)"
    )
    st.sidebar.caption("Lecture transcription runs locally with Whisper and requires `ffmpeg`.")

    if st.session_state.pipeline_summary:
        summary = st.session_state.pipeline_summary
        st.sidebar.divider()
        st.sidebar.header("Current Run")
        st.sidebar.caption(f"Lecture: `{summary['lecture_title']}`")
        st.sidebar.caption(f"Run folder: `{summary['run_dir']}`")
        if st.sidebar.button("Clear loaded lecture", use_container_width=True):
            st.session_state.pipeline_summary = None
            reset_quiz_state()
            st.rerun()


def find_local_transcripts() -> List[str]:
    candidates = []
    for pattern in ["*.txt", "preprocessing/transcription_output/*_transcript.txt"]:
        for path in sorted(Path(".").glob(pattern)):
            if path.is_file():
                candidates.append(str(path))
    seen = set()
    ordered = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def save_quiz_if_possible(quiz_bank: Dict[str, Any], run_dir: str) -> Optional[str]:
    if not run_dir:
        return None
    output_path = Path(run_dir) / "quiz_bank.json"
    saved_path = save_quiz_bank(quiz_bank, str(output_path))
    return str(saved_path)


def render_artifact_downloads(summary: Dict[str, Any]) -> None:
    st.markdown("#### Artifacts")
    download_cols = st.columns(4)
    artifact_items = [
        ("Transcript TXT", summary.get("transcript_path")),
        ("Transcript JSON", summary.get("transcript_json_path")),
        ("Segments JSON", summary.get("segments_path")),
        ("Concepts JSON", summary.get("concepts_path")),
    ]

    for column, (label, path_str) in zip(download_cols, artifact_items):
        with column:
            if path_str and Path(path_str).exists():
                file_path = Path(path_str)
                st.download_button(
                    label,
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    use_container_width=True,
                )


def render_processed_summary(summary: Dict[str, Any]) -> None:
    stats = summary["statistics"]
    segments = summary["segments"]

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Processed Lecture")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Words", f"{stats['word_count']:,}")
    metric_cols[1].metric("Duration", stats["duration"])
    metric_cols[2].metric("Lecture Segments", f"{len(segments):,}")
    metric_cols[3].metric("Lecture Topics", stats["detected_topics"])
    metric_cols[4].metric("Extracted Concepts", stats["concept_count"])

    topic_rows = [
        {
            "Segment": seg["segment_id"],
            "Topic": seg.get("topic", "Untitled"),
            "Preview": f"{seg.get('text', '')[:160]}...",
        }
        for seg in segments
    ]

    with st.expander("Transcript preview", expanded=False):
        st.write(summary["transcript_preview"])

    with st.expander("Detected lecture segments", expanded=True):
        st.dataframe(topic_rows, use_container_width=True, hide_index=True)

    render_artifact_downloads(summary)
    st.markdown("</div>", unsafe_allow_html=True)


def compute_answered_count(questions: List[Dict[str, Any]]) -> int:
    answered = 0
    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id", f"q_{idx}")
        if st.session_state.answers.get(qid) is not None:
            answered += 1
    return answered


def render_quiz_player(quiz_bank: Dict[str, Any]) -> None:
    questions = quiz_bank.get("questions", [])
    if not questions:
        st.info("Generate a quiz to start answering questions.")
        return

    answered = compute_answered_count(questions)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Quiz")
    st.caption(f"Answered {answered} of {len(questions)} questions.")

    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id", f"q_{idx}")
        options = question.get("options", [])
        current_value = st.session_state.answers.get(qid)

        with st.container(border=True):
            st.markdown(f"#### Q{idx}. {question.get('question', '')}")
            st.caption(f"Topic: {question.get('topic', 'Lecture Concepts')} | Difficulty: {question.get('difficulty', 'medium')}")

            choice = st.radio(
                "Choose one answer",
                options=list(range(len(options))),
                index=current_value if current_value is not None else None,
                format_func=lambda option_index, option_list=options: option_list[option_index],
                key=f"quiz_answer_{qid}",
                disabled=st.session_state.submitted,
            )
            st.session_state.answers[qid] = choice

            if st.session_state.submitted:
                correct_index = question.get("answer_index")
                explanation = question.get("explanation", "")

                if choice is None:
                    st.warning("No answer selected for this question.")
                elif choice == correct_index:
                    st.success("Correct")
                else:
                    st.error(f"Incorrect. Correct answer: {options[correct_index]}")

                if explanation:
                    st.info(explanation)

    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        submit_clicked = st.button(
            "Submit Quiz",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.submitted,
        )
    with action_cols[1]:
        if st.button("Reset Answers", use_container_width=True):
            st.session_state.answers = {}
            st.session_state.submitted = False
            clear_answer_widget_state()
            st.rerun()

    if submit_clicked:
        st.session_state.submitted = True
        st.rerun()

    if st.session_state.submitted:
        correct = 0
        attempted = 0

        for idx, question in enumerate(questions, start=1):
            qid = question.get("question_id", f"q_{idx}")
            selected = st.session_state.answers.get(qid)
            answer_index = question.get("answer_index")
            if selected is None:
                continue
            attempted += 1
            if selected == answer_index:
                correct += 1

        st.markdown("#### Results")
        result_cols = st.columns(3)
        result_cols[0].metric("Correct", f"{correct}/{len(questions)}")
        result_cols[1].metric("Attempted", f"{attempted}/{len(questions)}")
        result_cols[2].metric(
            "Score",
            f"{(correct / len(questions)) * 100:.1f}%" if questions else "0.0%",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    init_session_state()
    render_sidebar()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Frontend / Interface / Integration</div>
            <h1 style="margin:0 0 0.4rem 0;">Lecture-to-Quiz Studio</h1>
            <p style="margin:0; color:#4d6157;">
                Upload an MP4 lecture recording, turn it into segmented concepts, and generate an interactive quiz that
                students can use to check their understanding immediately.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 1. Upload and Process Lecture")

    process_mode = st.radio(
        "Input mode",
        options=["Upload MP4", "Upload transcript", "Use local transcript"],
        horizontal=True,
    )

    left_col, right_col = st.columns([1.2, 0.8])
    uploaded_file = None
    uploaded_transcript = None
    selected_local_transcript = None

    with left_col:
        if process_mode == "Upload MP4":
            uploaded_file = st.file_uploader(
                "Attach a lecture recording",
                type=["mp4", "m4v", "mov"],
                help="MP4 is the main expected format for this project.",
            )
            default_title = Path(uploaded_file.name).stem if uploaded_file else "AAI3008 lecture"
        elif process_mode == "Upload transcript":
            uploaded_transcript = st.file_uploader(
                "Attach a transcript file",
                type=["txt"],
                help="Use this mode if transcription is already done or MP4 processing is blocked in your environment.",
            )
            default_title = Path(uploaded_transcript.name).stem if uploaded_transcript else "AAI3008 transcript"
        else:
            local_transcripts = find_local_transcripts()
            selected_local_transcript = st.selectbox(
                "Choose a local transcript",
                options=local_transcripts,
                index=0 if local_transcripts else None,
                placeholder="No transcript files found",
            )
            default_title = Path(selected_local_transcript).stem if selected_local_transcript else "AAI3008 transcript"

        lecture_title = st.text_input("Lecture title", value=default_title)
        lecture_id = st.text_input("Lecture ID", value=slugify(lecture_title or default_title))

    with right_col:
        transcription_model = st.selectbox(
            "Whisper model",
            options=["tiny", "base", "small", "medium"],
            index=1,
            disabled=process_mode != "Upload MP4",
        )
        language = st.selectbox(
            "Language",
            options=["en", "auto"],
            index=0,
            disabled=process_mode != "Upload MP4",
        )
        chunk_minutes = st.slider(
            "Chunk size (minutes)",
            min_value=5,
            max_value=20,
            value=10,
            disabled=process_mode != "Upload MP4",
        )
        process_clicked = st.button("Process Lecture", type="primary", use_container_width=True)

    provider_status = get_provider_status()
    if not provider_status["configured"]:
        st.warning(
            f"LLM concept extraction needs `{provider_status['key_name']}`. "
            "Transcription may still run, but the full quiz pipeline will fail without an API key."
        )
    elif process_mode != "Upload MP4":
        st.info("Transcript mode bypasses Whisper and ffmpeg. This is the fastest way to continue frontend integration work.")

    if process_clicked:
        reset_quiz_state()
        status_box = st.status("Processing lecture...", expanded=True)

        def progress_callback(stage: str, message: str) -> None:
            status_box.write(f"**{stage.title()}**: {message}")

        try:
            if process_mode == "Upload MP4":
                if uploaded_file is None:
                    st.error("Upload an MP4 lecture recording before processing.")
                    st.stop()

                summary = process_uploaded_lecture(
                    upload_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    lecture_title=lecture_title,
                    lecture_id=lecture_id,
                    transcription_model=transcription_model,
                    language=None if language == "auto" else language,
                    chunk_minutes=chunk_minutes,
                    progress_callback=progress_callback,
                )
            elif process_mode == "Upload transcript":
                if uploaded_transcript is None:
                    st.error("Upload a transcript `.txt` file before processing.")
                    st.stop()

                summary = process_transcript_text(
                    transcript_name=uploaded_transcript.name,
                    transcript_text=uploaded_transcript.getvalue().decode("utf-8", errors="ignore"),
                    lecture_title=lecture_title,
                    lecture_id=lecture_id,
                    progress_callback=progress_callback,
                )
            else:
                if not selected_local_transcript:
                    st.error("Select a local transcript before processing.")
                    st.stop()

                transcript_path = Path(selected_local_transcript)
                summary = process_transcript_text(
                    transcript_name=transcript_path.name,
                    transcript_text=transcript_path.read_text(encoding="utf-8"),
                    lecture_title=lecture_title,
                    lecture_id=lecture_id,
                    progress_callback=progress_callback,
                )

            st.session_state.pipeline_summary = summary
            status_box.update(label="Lecture processed successfully", state="complete", expanded=True)
        except Exception as exc:
            status_box.update(label="Lecture processing failed", state="error", expanded=True)
            status_box.write(str(exc))

    st.markdown("</div>", unsafe_allow_html=True)

    summary = st.session_state.pipeline_summary
    if summary:
        render_processed_summary(summary)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 2. Generate Quiz")

        topics = ["All"] + sorted({seg.get("topic", "") for seg in summary["segments"] if seg.get("topic")})
        concept_count = max(summary["statistics"]["concept_count"], 1)

        quiz_cols = st.columns([0.9, 0.9, 0.7, 0.7])
        with quiz_cols[0]:
            topic_filter = st.selectbox("Topic focus", options=topics)
        with quiz_cols[1]:
            difficulty = st.selectbox("Difficulty", options=["mixed", "easy", "medium", "hard"], index=0)
        with quiz_cols[2]:
            question_count = st.slider(
                "Question count",
                min_value=3,
                max_value=min(15, max(3, concept_count)),
                value=min(8, max(3, concept_count)),
            )
        with quiz_cols[3]:
            generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)

        if generate_clicked:
            with st.spinner("Generating live MCQs from the extracted concepts..."):
                try:
                    quiz_bank = generate_quiz_bank(
                        lecture_id=summary["lecture_id"],
                        segments=summary["segments"],
                        concept_entries=summary["concepts"],
                        question_count=question_count,
                        difficulty=difficulty,
                        topic_filter=topic_filter,
                    )
                    saved_path = save_quiz_if_possible(quiz_bank, summary["run_dir"])
                    if saved_path:
                        quiz_bank["metadata"]["saved_path"] = saved_path
                    st.session_state.quiz_bank = quiz_bank
                    st.session_state.answers = {}
                    st.session_state.submitted = False
                    clear_answer_widget_state()
                except Exception as exc:
                    st.error(f"Quiz generation failed: {exc}")

        if st.session_state.quiz_bank:
            metadata = st.session_state.quiz_bank.get("metadata", {})
            st.caption(
                f"Generated {metadata.get('question_count', 0)} questions "
                f"from {metadata.get('source_concept_count', 0)} concepts."
            )
            if metadata.get("fallback_used"):
                st.warning(
                    "Some questions were created from the extracted concept bank because the LLM returned too few valid MCQs."
                )
            saved_path = metadata.get("saved_path")
            if saved_path and Path(saved_path).exists():
                quiz_path = Path(saved_path)
                st.download_button(
                    "Download Quiz Bank",
                    data=quiz_path.read_bytes(),
                    file_name=quiz_path.name,
                    use_container_width=False,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.quiz_bank:
        render_quiz_player(st.session_state.quiz_bank)
    elif summary:
        st.info("The lecture is ready. Generate a quiz to start the student-facing quiz flow.")


if __name__ == "__main__":
    main()
