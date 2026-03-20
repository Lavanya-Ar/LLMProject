import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from content_extraction import CodexLLMClient


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Any, path: str) -> None:
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def _eval_system_prompt() -> str:
    return """
You are an evaluator for lecture-based multiple-choice quiz questions.

Return ONLY valid JSON.

You will be given:
- a transcript segment
- extracted concepts
- a generated MCQ

Evaluate whether the question is high quality and grounded in the provided material.

Return JSON with this exact structure:
{
  "groundedness": 1,
  "clarity": 1,
  "single_correctness": 1,
  "distractor_quality": 1,
  "difficulty_fit": 1,
  "hallucinated": false,
  "decision": "keep|revise|drop",
  "notes": "short explanation"
}

Scoring guide:
- groundedness: Is the answer supported by the transcript/concepts?
- clarity: Is the wording understandable and unambiguous?
- single_correctness: Is exactly one option clearly correct?
- distractor_quality: Are wrong options plausible but incorrect?
- difficulty_fit: Does the labeled difficulty match the question?

Rules:
- Use integers from 1 to 5 only.
- Mark hallucinated = true if the question depends on unsupported information.
- decision:
  - keep = good as-is
  - revise = usable but should be improved
  - drop = poor or unsafe to use
- Return strictly valid JSON only.
""".strip()


def _eval_user_prompt(
    segment_text: str,
    concepts: List[Dict[str, Any]],
    question: Dict[str, Any]
) -> str:
    concept_payload = []
    for c in concepts:
        concept_payload.append({
            "concept_id": c.get("concept_id", ""),
            "name": c.get("name", ""),
            "definition": c.get("definition", ""),
            "difficulty": c.get("difficulty", ""),
            "type": c.get("type", ""),
            "evidence": c.get("evidence", "")
        })

    return f"""Transcript segment:
{segment_text}

Extracted concepts:
{json.dumps(concept_payload, indent=2, ensure_ascii=False)}

Generated MCQ:
{json.dumps(question, indent=2, ensure_ascii=False)}

Evaluate this question and return only the JSON object.
"""


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _safe_parse_eval(raw: str) -> Dict[str, Any]:
    raw = _strip_code_fence(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
    except Exception:
        return {
            "groundedness": 1,
            "clarity": 1,
            "single_correctness": 1,
            "distractor_quality": 1,
            "difficulty_fit": 1,
            "hallucinated": True,
            "decision": "drop",
            "notes": "Failed to parse evaluator output."
        }

    def clamp_score(value: Any) -> int:
        try:
            n = int(value)
        except Exception:
            return 1
        return max(1, min(5, n))

    decision = str(data.get("decision", "drop")).strip().lower()
    if decision not in {"keep", "revise", "drop"}:
        decision = "drop"

    return {
        "groundedness": clamp_score(data.get("groundedness", 1)),
        "clarity": clamp_score(data.get("clarity", 1)),
        "single_correctness": clamp_score(data.get("single_correctness", 1)),
        "distractor_quality": clamp_score(data.get("distractor_quality", 1)),
        "difficulty_fit": clamp_score(data.get("difficulty_fit", 1)),
        "hallucinated": bool(data.get("hallucinated", False)),
        "decision": decision,
        "notes": str(data.get("notes", "")).strip()
    }


def evaluate_quiz_bank(
    quiz_bank: Dict[str, Any],
    segments: List[Dict[str, Any]],
    concepts_data: List[Dict[str, Any]],
    client: CodexLLMClient,
    sleep_between_calls: float = 0.0
) -> Dict[str, Any]:
    seg_by_id = {seg["segment_id"]: seg for seg in segments}
    concepts_by_segment = {
        entry["segment_id"]: entry.get("concepts", [])
        for entry in concepts_data
    }

    results = []
    kept_questions = []

    # for q in quiz_bank.get("questions", []):
    for idx, q in enumerate(quiz_bank.get("questions", []), start=1):
        segment_id = q.get("segment_id")
        if segment_id is None:
            print(f"[EVAL {idx}/{len(quiz_bank.get('questions', []))}] Missing segment_id. Skipping.")
            continue
        seg = seg_by_id.get(segment_id)
        if not seg:
            continue

        segment_text = seg.get("text", "")
        concepts = concepts_by_segment.get(segment_id, [])

        system_prompt = _eval_system_prompt()
        user_prompt = _eval_user_prompt(segment_text, concepts, q)

        try:
            raw = client.generate(system_prompt, user_prompt)
            eval_result = _safe_parse_eval(raw)
        except Exception as e:
            eval_result = {
                "groundedness": 1,
                "clarity": 1,
                "single_correctness": 1,
                "distractor_quality": 1,
                "difficulty_fit": 1,
                "hallucinated": True,
                "decision": "drop",
                "notes": f"Evaluation request failed: {e}"
            }

        row = {
            "question_id": q.get("question_id", ""),
            "segment_id": segment_id,
            "topic": q.get("topic", ""),
            "question": q.get("question", ""),
            "evaluation": eval_result
        }
        results.append(row)

        # if eval_result["decision"] == "keep" and not eval_result["hallucinated"]:
        #     kept_questions.append(q)
        # if (
        #     eval_result["decision"] == "keep"
        #     and not eval_result["hallucinated"]
        #     and eval_result["groundedness"] >= 4
        #     and eval_result["clarity"] >= 4
        #     and eval_result["single_correctness"] >= 4
        # ):
        #     kept_questions.append(q)
        if (
            eval_result["decision"] == "keep"
            and not eval_result["hallucinated"]
            and eval_result["groundedness"] >= 4
            and eval_result["clarity"] >= 4
            and eval_result["single_correctness"] >= 4
            and eval_result["distractor_quality"] >= 3
        ):
            kept_questions.append(q)

        print(f"[EVAL {idx}/{len(quiz_bank.get('questions', []))}] question_id={q.get('question_id', '')}")

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    summary = {
        "lecture_id": quiz_bank.get("lecture_id", ""),
        "original_question_count": len(quiz_bank.get("questions", [])),
        "kept_question_count": len(kept_questions),
        "results": results
    }

    filtered_quiz_bank = {
        **quiz_bank,
        "total_questions": len(kept_questions),
        "questions": kept_questions
    }
    if isinstance(filtered_quiz_bank.get("metadata"), dict):
        filtered_quiz_bank["metadata"]["question_count"] = len(kept_questions)

    return {
        "evaluation_summary": summary,
        "filtered_quiz_bank": filtered_quiz_bank
    }