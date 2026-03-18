import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st


DEFAULT_QUIZ_PATHS = [
    "data/quiz_bank.json",
    "quiz_bank.json",
    "data/sample_quiz_bank.json",
]


def load_quiz_bank(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_existing_quiz_path() -> Optional[str]:
    for p in DEFAULT_QUIZ_PATHS:
        if Path(p).exists():
            return p
    return None


def filter_questions(
    questions: List[Dict[str, Any]],
    topic: str = "All",
    difficulty: str = "All",
) -> List[Dict[str, Any]]:
    out = []
    for q in questions:
        if topic != "All" and q.get("topic") != topic:
            continue
        if difficulty != "All" and q.get("difficulty") != difficulty:
            continue
        out.append(q)
    return out


def sample_questions(questions: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    return rng.sample(questions, n)


def get_unique_values(items: List[Dict[str, Any]], key: str) -> List[str]:
    vals = sorted({str(x.get(key, "")).strip() for x in items if str(x.get(key, "")).strip()})
    return vals


def init_session_state():
    st.session_state.setdefault("quiz_loaded_path", None)
    st.session_state.setdefault("quiz_bank", None)
    st.session_state.setdefault("selected_questions", [])
    st.session_state.setdefault("answers", {})  # question_id -> chosen_option_index
    st.session_state.setdefault("submitted", False)


def reset_quiz():
    st.session_state.selected_questions = []
    st.session_state.answers = {}
    st.session_state.submitted = False


st.set_page_config(page_title="Lecture Quiz Generator (MVP)", layout="wide")
init_session_state()

st.title("Lecture Quiz (MVP)")
st.caption("Frontend/Interface & Integration — loads a quiz bank JSON and runs an interactive MCQ quiz.")

# Sidebar: load quiz bank
st.sidebar.header("Quiz Bank")
existing_path = find_existing_quiz_path()
default_path = existing_path or "data/quiz_bank.json"

quiz_path = st.sidebar.text_input("Quiz bank JSON path", value=default_path)

col_load1, col_load2 = st.sidebar.columns(2)
with col_load1:
    load_btn = st.button("Load quiz bank", use_container_width=True)
with col_load2:
    reset_btn = st.button("Reset quiz", use_container_width=True)

if reset_btn:
    reset_quiz()

# Auto-load on first run if file exists
if st.session_state.quiz_bank is None and existing_path:
    try:
        st.session_state.quiz_bank = load_quiz_bank(existing_path)
        st.session_state.quiz_loaded_path = existing_path
    except Exception as e:
        st.sidebar.error(f"Failed to auto-load quiz bank: {e}")

if load_btn:
    try:
        st.session_state.quiz_bank = load_quiz_bank(quiz_path)
        st.session_state.quiz_loaded_path = quiz_path
        reset_quiz()
        st.sidebar.success(f"Loaded: {quiz_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to load quiz bank: {e}")

quiz_bank = st.session_state.quiz_bank
if not quiz_bank:
    st.warning(
        "No quiz bank loaded. Add a quiz bank JSON file (e.g., `data/quiz_bank.json`) "
        "or use the sample file `data/sample_quiz_bank.json`."
    )
    st.stop()

# Expected schema (flexible):
# {
#   "metadata": {...},
#   "questions": [ { question_id, lecture_id, topic, difficulty, question, options, answer_index, explanation } ... ]
# }
questions = quiz_bank.get("questions", [])
if not isinstance(questions, list) or len(questions) == 0:
    st.error("Quiz bank loaded but contains no questions. Expected `questions: [...]` in the JSON.")
    st.stop()

st.sidebar.caption(f"Loaded: {st.session_state.quiz_loaded_path}")
st.sidebar.caption(f"Total questions: {len(questions)}")

# Filters
st.sidebar.header("Quiz Settings")
topics = ["All"] + get_unique_values(questions, "topic")
difficulties = ["All"] + get_unique_values(questions, "difficulty")

selected_topic = st.sidebar.selectbox("Topic", options=topics)
selected_difficulty = st.sidebar.selectbox("Difficulty", options=difficulties)

filtered = filter_questions(questions, topic=selected_topic, difficulty=selected_difficulty)
st.sidebar.caption(f"Matching questions: {len(filtered)}")

num_q = st.sidebar.slider("Number of questions", min_value=1, max_value=min(30, max(1, len(filtered))), value=min(10, len(filtered)))
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42)

start_btn = st.sidebar.button("Start / Regenerate quiz", type="primary", use_container_width=True)

if start_btn:
    st.session_state.selected_questions = sample_questions(filtered, num_q, seed=seed)
    st.session_state.answers = {}
    st.session_state.submitted = False

# If quiz hasn't started yet, start with defaults
if not st.session_state.selected_questions:
    st.session_state.selected_questions = sample_questions(filtered, num_q, seed=seed)

selected_questions = st.session_state.selected_questions

# Main UI
st.subheader("Quiz")

with st.expander("Quiz bank metadata", expanded=False):
    st.json(quiz_bank.get("metadata", {}))

# Render questions
for idx, q in enumerate(selected_questions, start=1):
    qid = q.get("question_id") or f"q_{idx}"
    question_text = q.get("question", "")
    options = q.get("options", [])
    topic = q.get("topic", "")
    difficulty = q.get("difficulty", "")

    if not isinstance(options, list) or len(options) < 2:
        st.error(f"Question {qid} has invalid options list.")
        continue

    st.markdown(f"### Q{idx}. {question_text}")
    meta_line = " • ".join([x for x in [topic and f"Topic: {topic}", difficulty and f"Difficulty: {difficulty}"] if x])
    if meta_line:
        st.caption(meta_line)

    # Use session-state to persist selection
    current = st.session_state.answers.get(qid, None)
    choice = st.radio(
        "Select an answer",
        options=list(range(len(options))),
        format_func=lambda i: options[i],
        index=current if current is not None else 0,
        key=f"radio_{qid}",
        disabled=st.session_state.submitted,
    )
    st.session_state.answers[qid] = choice

    # Feedback if submitted
    if st.session_state.submitted:
        correct_index = q.get("answer_index", None)
        explanation = q.get("explanation", "")

        if correct_index is None:
            st.warning("No answer key provided for this question.")
        else:
            if choice == correct_index:
                st.success("Correct")
            else:
                st.error(f"Incorrect. Correct answer: {options[correct_index]}")
            if explanation:
                st.info(f"Explanation: {explanation}")

    st.divider()

# Submit + scoring
col1, col2 = st.columns([1, 2])
with col1:
    submit = st.button("Submit quiz", type="primary", disabled=st.session_state.submitted)
with col2:
    st.caption("After submission, you will see score + explanations (if provided).")

if submit:
    st.session_state.submitted = True

if st.session_state.submitted:
    # Compute score
    total = 0
    correct = 0
    for idx, q in enumerate(selected_questions, start=1):
        qid = q.get("question_id") or f"q_{idx}"
        correct_index = q.get("answer_index", None)
        chosen = st.session_state.answers.get(qid, None)
        if correct_index is None or chosen is None:
            continue
        total += 1
        if chosen == correct_index:
            correct += 1

    if total > 0:
        st.subheader("Results")
        st.metric("Score", f"{correct}/{total}", f"{(correct/total)*100:.1f}%")
    else:
        st.warning("Could not compute score (missing answer keys or answers).")