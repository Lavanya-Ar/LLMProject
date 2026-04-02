import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from content_extraction import CodexLLMClient, LLMClientProtocol


# -------------------------
# Pipeline 2 (existing) - UI friendly schema
# -------------------------
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def flatten_concepts(
    segments: List[Dict[str, Any]],
    concept_entries: List[Dict[str, Any]],
    topic_filter: str = "All",
) -> List[Dict[str, Any]]:
    segments_by_id = {seg["segment_id"]: seg for seg in segments}
    flattened: List[Dict[str, Any]] = []

    for entry in concept_entries:
        segment_id = entry.get("segment_id")
        segment = segments_by_id.get(segment_id, {})
        topic = segment.get("topic", "") or ""
        if topic_filter != "All" and topic != topic_filter:
            continue

        segment_excerpt = " ".join(segment.get("text", "").split())[:400]

        for concept in entry.get("concepts", []):
            flattened.append(
                {
                    "segment_id": segment_id,
                    "topic": concept.get("topic", topic) or topic,
                    "concept_id": concept.get("concept_id", ""),
                    "name": concept.get("name", ""),
                    "definition": concept.get("definition", ""),
                    "difficulty": str(concept.get("difficulty", "medium")).lower(),
                    "type": concept.get("type", "definition"),
                    "evidence": concept.get("evidence", ""),
                    "segment_excerpt": segment_excerpt,
                }
            )

    return flattened


def build_source_bundle(
    lecture_id: str,
    source_concepts: List[Dict[str, Any]],
    question_count: int,
    difficulty: str,
) -> str:
    payload = json.dumps(source_concepts, ensure_ascii=False, indent=2)
    difficulty_instruction = (
        f"Every question must be `{difficulty}` difficulty."
        if difficulty in VALID_DIFFICULTIES
        else "Use a balanced mix of easy, medium, and hard questions."
    )

    return f"""Lecture ID: {lecture_id}
Requested questions: {question_count}
Difficulty requirement: {difficulty_instruction}

Source concepts JSON:
{payload}

Create exactly {question_count} multiple-choice questions using only the source concepts.

Return ONLY valid JSON with this exact structure:
{{
  "questions": [
    {{
      "question_id": "short_unique_id",
      "topic": "topic label from the source material",
      "difficulty": "easy|medium|hard",
      "question": "clear question stem",
      "options": ["option A", "option B", "option C", "option D"],
      "answer_index": 0,
      "explanation": "2 to 3 sentences explaining why the correct answer is right and why the distractors are wrong or misleading",
      "source_concept_ids": ["concept_id_1"]
    }}
  ]
}}

Rules:
- Use only the provided concepts and excerpts.
- Do not invent outside facts.
- Every question must have exactly 4 answer options.
- Only one answer may be correct.
- Make distractors plausible but clearly wrong.
- Avoid repeated questions and repeated correct answers.
- Prefer precise wording over trick questions.
- `answer_index` must be an integer from 0 to 3.
- `source_concept_ids` must reference concept IDs from the source concepts.
""".strip()


def mcq_system_prompt() -> str:
    return """
You create assessment-quality lecture revision MCQs grounded only in the provided source material.

Return only valid JSON.
Do not include markdown.
Do not include commentary.
""".strip()


def safe_parse_quiz(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]

    try:
        data = json.loads(raw)
    except Exception:
        return []

    questions = data.get("questions", [])
    return questions if isinstance(questions, list) else []


def normalize_difficulty(value: Any, fallback: str) -> str:
    difficulty = str(value or "").strip().lower()
    if difficulty in VALID_DIFFICULTIES:
        return difficulty
    if fallback in VALID_DIFFICULTIES:
        return fallback
    return "medium"


def validate_questions(
    raw_questions: List[Dict[str, Any]],
    lecture_id: str,
    requested_count: int,
    requested_difficulty: str,
) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    seen_questions = set()

    for idx, question in enumerate(raw_questions, start=1):
        stem = str(question.get("question", "")).strip()
        options = question.get("options", [])
        explanation = str(question.get("explanation", "")).strip()
        topic = str(question.get("topic", "")).strip() or "Lecture Concepts"
        answer_index = question.get("answer_index")

        if not stem or stem.lower() in seen_questions:
            continue
        if not isinstance(options, list) or len(options) != 4:
            continue

        cleaned_options = [str(option).strip() for option in options]
        if any(not option for option in cleaned_options):
            continue
        if len({option.lower() for option in cleaned_options}) != 4:
            continue

        try:
            answer_index = int(answer_index)
        except Exception:
            continue

        if answer_index not in {0, 1, 2, 3}:
            continue

        seen_questions.add(stem.lower())
        validated.append(
            {
                "question_id": str(question.get("question_id", f"{lecture_id}_q{idx}")).strip() or f"{lecture_id}_q{idx}",
                "lecture_id": lecture_id,
                "topic": topic,
                "difficulty": normalize_difficulty(question.get("difficulty"), requested_difficulty),
                "question": stem,
                "options": cleaned_options,
                "answer_index": answer_index,
                "explanation": explanation or "This answer is supported by the lecture concepts used to generate the question.",
                "source_concept_ids": question.get("source_concept_ids", []),
            }
        )

        if len(validated) >= requested_count:
            break

    return validated


def build_fallback_questions(
    lecture_id: str,
    source_concepts: List[Dict[str, Any]],
    requested_count: int,
    requested_difficulty: str,
) -> List[Dict[str, Any]]:
    concepts_by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for concept in source_concepts:
        if concept.get("name") and concept.get("definition"):
            concepts_by_topic[concept.get("topic", "Lecture Concepts")].append(concept)

    all_concepts = [concept for concepts in concepts_by_topic.values() for concept in concepts]
    if len(all_concepts) < 4:
        return []

    rng = random.Random(42)
    fallback_questions: List[Dict[str, Any]] = []

    for idx, concept in enumerate(all_concepts, start=1):
        distractor_pool = [
            other["definition"]
            for other in all_concepts
            if other.get("concept_id") != concept.get("concept_id") and other.get("definition")
        ]
        if len(distractor_pool) < 3:
            continue

        distractors = rng.sample(distractor_pool, 3)
        options = distractors + [concept["definition"]]
        rng.shuffle(options)
        answer_index = options.index(concept["definition"])

        fallback_questions.append(
            {
                "question_id": f"{lecture_id}_fallback_q{idx}",
                "lecture_id": lecture_id,
                "topic": concept.get("topic", "Lecture Concepts"),
                "difficulty": normalize_difficulty(concept.get("difficulty"), requested_difficulty),
                "question": f"Which description best matches {concept['name']}?",
                "options": options,
                "answer_index": answer_index,
                "explanation": (
                    f"{concept['name']} is best described by the correct option because it matches the lecture definition. "
                    "The other options describe different concepts from the same lecture."
                ),
                "source_concept_ids": [concept.get("concept_id", "")],
            }
        )

        if len(fallback_questions) >= requested_count:
            break

    return fallback_questions


# -------------------------
# Pipeline 1 (imported) - segment-grounded, repair, A/B/C/D options
# NOTE: We keep it here but convert output to pipeline 2 schema for Streamlit.
# -------------------------
VALID_QUESTION_TYPES_V1 = {"definition", "understanding", "comparison", "application"}


def _quiz_system_prompt_v1() -> str:
    return """
You are an assistant that generates high-quality multiple-choice quiz questions from lecture transcript segments and extracted concepts.

Return ONLY valid JSON.

You must generate questions grounded strictly in the provided transcript segment and concept list.
Do not use outside knowledge.
Do not add facts that are not supported by the provided material.

Return JSON with this exact structure:
{
  "questions": [
    {
      "question": "string",
      "question_type": "definition|understanding|comparison|application",
      "difficulty": "easy|medium|hard",
      "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      },
      "correct_answer": "A|B|C|D",
      "explanation": "string",
      "evidence": "short supporting phrase from the segment",
      "concept_names": ["concept 1", "concept 2"]
    }
  ]
}

Rules:
- Generate only multiple-choice questions.
- Exactly 4 options per question: A, B, C, D.
- Exactly 1 correct answer.
- Questions must be directly answerable from the provided material.
- Distractors must be plausible but clearly incorrect.
- Avoid vague or ambiguous wording.
- Avoid "all of the above" and "none of the above".
- Avoid negative wording like "Which is NOT..." unless absolutely necessary.
- Prefer conceptual understanding over trivial copy-paste recall.
- Explanations must be short and clear for students.
- "evidence" must be a short phrase directly supported by the transcript segment.
- "concept_names" should match names from the provided concept list whenever possible.
- Return strictly valid JSON only.
""".strip()


def _repair_system_prompt_v1() -> str:
    return """
You are repairing a lecture-based MCQ.

Return ONLY valid JSON with this exact structure:
{
  "question": "string",
  "question_type": "definition|understanding|comparison|application",
  "difficulty": "easy|medium|hard",
  "options": {
    "A": "string",
    "B": "string",
    "C": "string",
    "D": "string"
  },
  "correct_answer": "A|B|C|D",
  "explanation": "string",
  "evidence": "short supporting phrase from the segment",
  "concept_names": ["concept 1"]
}

Rules:
- Keep the question grounded only in the provided material.
- Fix ambiguity.
- Ensure exactly one clearly correct answer.
- Make distractors plausible but clearly wrong.
- Use clean student-friendly wording.
- Return only JSON.
""".strip()


def _strip_code_fence_v1(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _safe_parse_questions_v1(raw: str) -> List[Dict[str, Any]]:
    raw = _strip_code_fence_v1(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
    except Exception:
        return []

    questions = data.get("questions", [])
    return questions if isinstance(questions, list) else []


def _normalize_question_type_v1(qtype: str) -> str:
    qtype = str(qtype).strip().lower()
    if qtype in VALID_QUESTION_TYPES_V1:
        return qtype
    return "understanding"


def _normalize_difficulty_v1(value: str, fallback: str = "medium") -> str:
    value = str(value).strip().lower()
    if value in VALID_DIFFICULTIES:
        return value
    return fallback


def _normalize_options_v1(options: Any) -> Optional[Dict[str, str]]:
    if not isinstance(options, dict):
        return None

    normalized = {}
    for key in ["A", "B", "C", "D"]:
        value = options.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        normalized[key] = value
    return normalized


def _normalize_text_v1(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _option_overlap_ratio_v1(a: str, b: str) -> float:
    a_words = set(_normalize_text_v1(a).split())
    b_words = set(_normalize_text_v1(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))


def _has_too_similar_options_v1(options: Dict[str, str]) -> bool:
    keys = ["A", "B", "C", "D"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _option_overlap_ratio_v1(options[keys[i]], options[keys[j]]) > 0.75:
                return True
    return False


def _looks_transcripty_v1(text: str) -> bool:
    t = _normalize_text_v1(text)
    bad_phrases = ["okay", "right", "you know", "kind of", "sort of"]
    return any(p in t for p in bad_phrases)


def _question_is_too_generic_v1(question: str) -> bool:
    q = _normalize_text_v1(question)
    generic_starts = [
        "what is the primary purpose",
        "what is the main role",
        "what does this mean",
        "why is",
        "how does",
    ]
    return any(q.startswith(x) for x in generic_starts) and len(q.split()) < 14


def _option_contains_answer_words_v1(correct_text: str, wrong_text: str) -> bool:
    c_words = set(_normalize_text_v1(correct_text).split())
    w_words = set(_normalize_text_v1(wrong_text).split())
    if not c_words or not w_words:
        return False
    overlap = len(c_words & w_words)
    return overlap >= max(3, int(0.7 * len(c_words)))


def _clean_evidence_text_v1(evidence: str) -> str:
    text = " ".join(str(evidence).strip().split())
    text = re.sub(r"\b(okay|right|you know|very straightforward)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    return text


def _quiz_user_prompt_v1(
    lecture_id: str,
    segment_id: int,
    topic: str,
    segment_text: str,
    concepts: List[Dict[str, Any]],
    num_questions: int,
    difficulty: Optional[str] = None,
) -> str:
    concept_payload = []
    for c in concepts:
        concept_payload.append(
            {
                "concept_id": c.get("concept_id", ""),
                "name": c.get("name", ""),
                "definition": c.get("definition", ""),
                "difficulty": c.get("difficulty", "medium"),
                "type": c.get("type", "definition"),
                "evidence": c.get("evidence", ""),
            }
        )

    difficulty_line = f"Requested difficulty: {difficulty}" if difficulty else "Requested difficulty: mixed"

    return f"""Lecture ID: {lecture_id}
Segment ID: {segment_id}
Topic: {topic}
{difficulty_line}
Number of questions to generate: {num_questions}

Transcript segment:
{segment_text}

Extracted concepts:
{json.dumps(concept_payload, indent=2, ensure_ascii=False)}

Generate {num_questions} MCQ question(s) for this segment.

Extra requirements:
- Use clean, student-friendly wording.
- Prefer definition, understanding, or comparison questions.
- Avoid vague "primary purpose" questions unless the answer is uniquely supported.
- Make the topic label concise and concept-based.
- Keep evidence short and directly grounded in the transcript.
Return only the JSON object.
"""


def _repair_user_prompt_v1(
    segment_text: str,
    concepts: List[Dict[str, Any]],
    bad_question: Dict[str, Any],
    reason: str,
) -> str:
    return f"""Transcript segment:
{segment_text}

Extracted concepts:
{json.dumps(concepts, indent=2, ensure_ascii=False)}

Question to repair:
{json.dumps(bad_question, indent=2, ensure_ascii=False)}

Why it needs repair:
{reason}

Rewrite this MCQ so it is clearer and less ambiguous.
Return only the JSON object.
"""


def _repair_question_v1(
    client: CodexLLMClient,
    segment_text: str,
    concepts: List[Dict[str, Any]],
    raw_question: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    try:
        raw = client.generate(_repair_system_prompt_v1(), _repair_user_prompt_v1(segment_text, concepts, raw_question, reason))
        raw = _strip_code_fence_v1(raw).strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        print(f"[REPAIR] failed: {e}")
    return raw_question


def _call_llm_for_questions_v1(
    client: CodexLLMClient,
    lecture_id: str,
    segment_id: int,
    topic: str,
    segment_text: str,
    concepts: List[Dict[str, Any]],
    num_questions: int,
    difficulty: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, str]:
    system_prompt = _quiz_system_prompt_v1()
    user_prompt = _quiz_user_prompt_v1(
        lecture_id=lecture_id,
        segment_id=segment_id,
        topic=topic,
        segment_text=segment_text,
        concepts=concepts,
        num_questions=num_questions,
        difficulty=difficulty,
    )

    try:
        raw = client.generate(system_prompt, user_prompt)
        used_provider = str(getattr(client, "provider", "")).strip().lower()
        used_model = str(getattr(client, "model", "")).strip()
    except Exception as e:
        print(f"Quiz generation failed for segment {segment_id} ({topic}): {e}")
        return [], "", ""

    return _safe_parse_questions_v1(raw), used_provider, used_model


def _pick_segment_question_count_v1(concepts_count: int, requested_total: int, total_segments: int) -> int:
    if requested_total <= total_segments:
        return 1
    if concepts_count >= 3:
        return 2
    return 1


def _clean_topic_name_v1(topic: str, concepts: List[Dict[str, Any]]) -> str:
    topic = str(topic or "").strip()
    lowered = topic.lower()

    bad_markers = [
        "word words",
        "hidden input state",
        "first token input",
        "hidden state here",
        "input decoder",
        "state english",
        "here matrix",
        "different have",
        "query key output some",
        "output some",
    ]

    looks_bad = (
        not topic
        or len(topic.split()) < 2
        or len(topic.split()) > 6
        or any(marker in lowered for marker in bad_markers)
    )

    if not looks_bad:
        return " ".join(topic.split())

    concept_names = [str(c.get("name", "")).strip() for c in concepts if str(c.get("name", "")).strip()]
    if concept_names:
        return concept_names[0][:80]
    return "Lecture Concept"


def _concept_name_to_id_map_v1(concepts: List[Dict[str, Any]]) -> Dict[str, str]:
    mapping = {}
    for c in concepts:
        name = str(c.get("name", "")).strip().lower()
        cid = str(c.get("concept_id", "")).strip()
        if name and cid:
            mapping[name] = cid
    return mapping


def _match_concept_ids_v1(concept_names: List[str], concept_name_id_map: Dict[str, str]) -> List[str]:
    matched = []
    for name in concept_names:
        key = str(name).strip().lower()
        cid = concept_name_id_map.get(key)
        if cid and cid not in matched:
            matched.append(cid)
    return matched


def _sanitize_question_v1(
    raw_q: Dict[str, Any],
    lecture_id: str,
    segment_id: int,
    local_idx: int,
    topic: str,
    concepts: List[Dict[str, Any]],
    requested_difficulty: Optional[str],
    seen_questions: set,
) -> Optional[Dict[str, Any]]:
    question = str(raw_q.get("question", "")).strip()
    explanation = str(raw_q.get("explanation", "")).strip()
    evidence = _clean_evidence_text_v1(raw_q.get("evidence", ""))
    correct_answer = str(raw_q.get("correct_answer", "")).strip().upper()
    qtype = _normalize_question_type_v1(raw_q.get("question_type", "understanding"))

    if requested_difficulty:
        difficulty = _normalize_difficulty_v1(requested_difficulty)
    else:
        difficulty = _normalize_difficulty_v1(raw_q.get("difficulty", "medium"))

    options = _normalize_options_v1(raw_q.get("options"))
    if not question or not explanation or not evidence or not options:
        return None

    if correct_answer not in {"A", "B", "C", "D"}:
        return None

    correct_text = options[correct_answer].strip()

    if _has_too_similar_options_v1(options):
        return None

    if _looks_transcripty_v1(question) or _looks_transcripty_v1(explanation):
        return None

    wrong_options = [v for k, v in options.items() if k != correct_answer]
    if any(_option_contains_answer_words_v1(correct_text, w) for w in wrong_options):
        return None

    if len(question.split()) < 6:
        return None
    if _question_is_too_generic_v1(question):
        return None
    if len(explanation.split()) < 8:
        return None

    all_option_values = [v.strip().lower() for v in options.values()]
    if len(set(all_option_values)) < 4:
        return None

    normalized_q = re.sub(r"\s+", " ", question.strip().lower())
    if normalized_q in seen_questions:
        return None
    seen_questions.add(normalized_q)

    concept_names = raw_q.get("concept_names", [])
    if not isinstance(concept_names, list):
        concept_names = []
    concept_names = [str(x).strip() for x in concept_names if isinstance(x, str) and str(x).strip()]

    concept_map = _concept_name_to_id_map_v1(concepts)
    concept_ids = _match_concept_ids_v1(concept_names, concept_map)

    if not concept_ids:
        lowered_question = question.lower()
        lowered_explanation = explanation.lower()
        for c in concepts:
            cname = str(c.get("name", "")).strip()
            cid = str(c.get("concept_id", "")).strip()
            if cname and cid and cname.lower() in (lowered_question + " " + lowered_explanation):
                concept_ids.append(cid)

    display_topic = topic
    if concept_names:
        display_topic = concept_names[0]
    elif concepts:
        first_concept_name = str(concepts[0].get("name", "")).strip()
        if first_concept_name:
            display_topic = first_concept_name

    question_id = f"{lecture_id}_s{segment_id}_q{local_idx}"

    return {
        "question_id": question_id,
        "lecture_id": lecture_id,
        "segment_id": segment_id,
        "topic": display_topic,
        "question_type": qtype,
        "difficulty": difficulty,
        "concept_ids": concept_ids,
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "correct_option_text": correct_text,
        "explanation": explanation,
        "evidence": evidence,
    }


def generate_quiz_bank_v1(
    segments: List[Dict[str, Any]],
    concepts_data: List[Dict[str, Any]],
    client: CodexLLMClient,
    lecture_id: Optional[str] = None,
    num_questions: int = 10,
    difficulty: Optional[str] = None,
    sleep_between_calls: float = 0.0,
) -> Dict[str, Any]:
    concepts_by_segment = {entry["segment_id"]: entry.get("concepts", []) for entry in concepts_data}

    if not lecture_id and segments:
        lecture_id = segments[0].get("lecture_id", "lecture_quiz")
    lecture_id = lecture_id or "lecture_quiz"

    questions: List[Dict[str, Any]] = []
    seen_questions = set()

    segments_with_concepts = [seg for seg in segments if concepts_by_segment.get(seg["segment_id"])]

    if not segments_with_concepts:
        return {
            "lecture_id": lecture_id,
            "title": lecture_id,
            "difficulty_filter": difficulty if difficulty else "mixed",
            "total_questions": 0,
            "questions": [],
        }

    for idx, seg in enumerate(segments_with_concepts, start=1):
        segment_id = seg["segment_id"]
        segment_text = seg.get("text", "")
        seg_concepts = concepts_by_segment.get(segment_id, [])
        raw_topic = seg.get("topic", f"Segment {segment_id}")
        topic = _clean_topic_name_v1(raw_topic, seg_concepts)

        print(f"[V1 SEGMENT {idx}/{len(segments_with_concepts)}] segment_id={segment_id}, topic={topic}")

        target_count = _pick_segment_question_count_v1(
            concepts_count=len(seg_concepts),
            requested_total=num_questions,
            total_segments=len(segments_with_concepts),
        )

        raw_questions, generated_provider, generated_model = _call_llm_for_questions_v1(
            client=client,
            lecture_id=lecture_id,
            segment_id=segment_id,
            topic=topic,
            segment_text=segment_text,
            concepts=seg_concepts,
            num_questions=target_count,
            difficulty=difficulty,
        )

        if not raw_questions:
            raw_questions, generated_provider, generated_model = _call_llm_for_questions_v1(
                client=client,
                lecture_id=lecture_id,
                segment_id=segment_id,
                topic=topic,
                segment_text=segment_text,
                concepts=seg_concepts,
                num_questions=1,
                difficulty=difficulty,
            )

        local_idx = 1
        for raw_q in raw_questions:
            cleaned = _sanitize_question_v1(
                raw_q=raw_q,
                lecture_id=lecture_id,
                segment_id=segment_id,
                local_idx=local_idx,
                topic=topic,
                concepts=seg_concepts,
                requested_difficulty=difficulty,
                seen_questions=seen_questions,
            )

            if not cleaned:
                repaired_raw = _repair_question_v1(
                    client=client,
                    segment_text=segment_text,
                    concepts=seg_concepts,
                    raw_question=raw_q,
                    reason="Clean student wording, one clearly correct answer, stronger grounding, better topic label.",
                )
                cleaned = _sanitize_question_v1(
                    raw_q=repaired_raw,
                    lecture_id=lecture_id,
                    segment_id=segment_id,
                    local_idx=local_idx,
                    topic=topic,
                    concepts=seg_concepts,
                    requested_difficulty=difficulty,
                    seen_questions=seen_questions,
                )

            if cleaned:
                cleaned["generation_provider"] = generated_provider
                cleaned["generation_model"] = generated_model
                questions.append(cleaned)
                local_idx += 1

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    if difficulty:
        difficulty_norm = _normalize_difficulty_v1(difficulty)
        questions = [q for q in questions if q["difficulty"] == difficulty_norm]

    if len(questions) > num_questions:
        questions = questions[:num_questions]

    return {
        "lecture_id": lecture_id,
        "title": lecture_id,
        "difficulty_filter": difficulty if difficulty else "mixed",
        "total_questions": len(questions),
        "questions": questions,
    }


def convert_quiz_bank_v1_to_v2(bank_v1: Dict[str, Any]) -> Dict[str, Any]:
    """Convert v1 questions (A/B/C/D dict + correct_answer letter) -> v2 UI schema (options list + answer_index)."""
    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}

    lecture_id = bank_v1.get("lecture_id", "lecture")
    v1_questions = bank_v1.get("questions", [])

    v2_questions = []
    for idx, q in enumerate(v1_questions, start=1):
        options = q.get("options", {})
        if not isinstance(options, dict):
            continue

        option_list = [options.get("A", ""), options.get("B", ""), options.get("C", ""), options.get("D", "")]
        if any(not isinstance(x, str) or not x.strip() for x in option_list):
            continue

        correct_letter = str(q.get("correct_answer", "")).strip().upper()
        if correct_letter not in letter_to_index:
            continue

        v2_questions.append(
            {
                "question_id": q.get("question_id", f"{lecture_id}_q{idx}"),
                "lecture_id": q.get("lecture_id", lecture_id),
                "topic": q.get("topic", "Lecture Concepts"),
                "difficulty": q.get("difficulty", "medium"),
                "question": q.get("question", ""),
                "options": option_list,
                "answer_index": letter_to_index[correct_letter],
                "explanation": q.get("explanation", ""),
                "source_concept_ids": q.get("concept_ids", []),
                # keep extras if you want later:
                "segment_id": q.get("segment_id"),
                "question_type": q.get("question_type"),
                "evidence": q.get("evidence", ""),
                "generation_provider": q.get("generation_provider", ""),
                "generation_model": q.get("generation_model", ""),
            }
        )

    return {
        "metadata": {
            "lecture_id": lecture_id,
            "question_count": len(v2_questions),
            "difficulty": bank_v1.get("difficulty_filter", "mixed"),
            "topic_filter": "All",
            "fallback_used": False,
            "llm_attempts": None,
            "source_concept_count": None,
            "generator": "v1_converted_to_v2",
        },
        "questions": v2_questions,
    }


# -------------------------
# Public entry point used by Streamlit (must return v2 schema)
# -------------------------
def generate_quiz_bank(
    lecture_id: str,
    segments: List[Dict[str, Any]],
    concept_entries: List[Dict[str, Any]],
    question_count: int,
    difficulty: str = "mixed",
    topic_filter: str = "All",
    client: Optional[LLMClientProtocol] = None,
) -> Dict[str, Any]:
    """
    Main entry point used by app.py.

    Controlled by env var:
      QUIZ_GENERATOR=v2 (default) -> existing concept-bundle MCQ generator
      QUIZ_GENERATOR=v1           -> segment-grounded generator (from Content-extraction), converted to v2 schema
    """
    generator = os.getenv("QUIZ_GENERATOR", "v2").strip().lower()

    if client is None:
        client = CodexLLMClient()

    if generator == "v1":
        # Convert summary["concepts"] to the shape v1 expects (concepts_data list with segment_id + concepts).
        # In LLMProject, concept_entries already match that: [{"segment_id":..., "concepts":[...]}]
        bank_v1 = generate_quiz_bank_v1(
            segments=segments,
            concepts_data=concept_entries,
            client=client,  # type: ignore[arg-type]
            lecture_id=lecture_id,
            num_questions=question_count,
            difficulty=None if difficulty == "mixed" else difficulty,
            sleep_between_calls=0.0,
        )
        bank_v2 = convert_quiz_bank_v1_to_v2(bank_v1)
        # Patch metadata to keep UI captions accurate-ish
        bank_v2["metadata"]["topic_filter"] = topic_filter
        bank_v2["metadata"]["difficulty"] = difficulty
        return bank_v2

    # Default: keep original v2 generator (stable with Streamlit + metadata)
    source_concepts = flatten_concepts(segments=segments, concept_entries=concept_entries, topic_filter=topic_filter)
    if not source_concepts:
        raise ValueError("No concepts are available for quiz generation. Process a lecture first.")

    prompt = build_source_bundle(
        lecture_id=lecture_id,
        source_concepts=source_concepts,
        question_count=question_count,
        difficulty=difficulty,
    )

    attempts: List[str] = []
    validated_questions: List[Dict[str, Any]] = []

    for _ in range(2):
        raw = client.generate(mcq_system_prompt(), prompt)
        generated_provider = str(getattr(client, "provider", "")).strip().lower()
        generated_model = str(getattr(client, "model", "")).strip()
        attempts.append(raw)
        validated_questions = validate_questions(
            raw_questions=safe_parse_quiz(raw),
            lecture_id=lecture_id,
            requested_count=question_count,
            requested_difficulty=difficulty,
        )
        for q in validated_questions:
            q["generation_provider"] = generated_provider
            q["generation_model"] = generated_model
        if len(validated_questions) >= question_count:
            break

    fallback_used = False
    if len(validated_questions) < question_count:
        fallback_questions = build_fallback_questions(
            lecture_id=lecture_id,
            source_concepts=source_concepts,
            requested_count=question_count - len(validated_questions),
            requested_difficulty=difficulty,
        )
        if fallback_questions:
            fallback_used = True
            for q in fallback_questions:
                q["generation_provider"] = "rule-based"
                q["generation_model"] = "rule-based"
            validated_questions.extend(fallback_questions)

    questions = validated_questions[:question_count]
    if len(questions) < question_count:
        raise RuntimeError(
            f"Quiz generation returned {len(questions)} usable questions out of the requested {question_count}."
        )

    return {
        "metadata": {
            "lecture_id": lecture_id,
            "question_count": len(questions),
            "difficulty": difficulty,
            "topic_filter": topic_filter,
            "fallback_used": fallback_used,
            "llm_attempts": len(attempts),
            "source_concept_count": len(source_concepts),
            "generator": "v2",
        },
        "questions": questions,
    }


def save_quiz_bank(quiz_bank: Dict[str, Any], output_path: str) -> Path:
    output = Path(output_path)
    output.write_text(json.dumps(quiz_bank, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


# Backward-compat helper from Content-extraction repo (not used by Streamlit, but harmless).
def load_segments_and_concepts(
    segments_path: str,
    concepts_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    concepts = json.loads(Path(concepts_path).read_text(encoding="utf-8"))
    return segments, concepts