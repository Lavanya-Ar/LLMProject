import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from content_extraction import CodexLLMClient, LLMClientProtocol


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


def generate_quiz_bank(
    lecture_id: str,
    segments: List[Dict[str, Any]],
    concept_entries: List[Dict[str, Any]],
    question_count: int,
    difficulty: str = "mixed",
    topic_filter: str = "All",
    client: Optional[LLMClientProtocol] = None,
) -> Dict[str, Any]:
    source_concepts = flatten_concepts(segments=segments, concept_entries=concept_entries, topic_filter=topic_filter)
    if not source_concepts:
        raise ValueError("No concepts are available for quiz generation. Process a lecture first.")

    if client is None:
        client = CodexLLMClient()

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
        attempts.append(raw)
        validated_questions = validate_questions(
            raw_questions=safe_parse_quiz(raw),
            lecture_id=lecture_id,
            requested_count=question_count,
            requested_difficulty=difficulty,
        )
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
        },
        "questions": questions,
    }


def save_quiz_bank(quiz_bank: Dict[str, Any], output_path: str) -> Path:
    output = Path(output_path)
    output.write_text(json.dumps(quiz_bank, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
