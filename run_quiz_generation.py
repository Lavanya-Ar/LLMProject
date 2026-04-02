# import json
# import re
# import time
# from pathlib import Path
# from typing import List, Dict, Any, Optional

# from content_extraction import CodexLLMClient


# def load_json(path: str) -> Any:
#     return json.loads(Path(path).read_text(encoding="utf-8"))


# def save_json(data: Any, path: str) -> None:
#     Path(path).write_text(
#         json.dumps(data, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )


# def _eval_system_prompt() -> str:
#     return """
# You are an evaluator for lecture-based multiple-choice quiz questions.

# Return ONLY valid JSON.

# You will be given:
# - a transcript segment
# - extracted concepts
# - a generated MCQ

# Evaluate whether the question is high quality and grounded in the provided material.

# Return JSON with this exact structure:
# {
#   "groundedness": 1,
#   "clarity": 1,
#   "single_correctness": 1,
#   "distractor_quality": 1,
#   "difficulty_fit": 1,
#   "hallucinated": false,
#   "decision": "keep|revise|drop",
#   "notes": "short explanation"
# }

# Scoring guide:
# - groundedness: Is the answer supported by the transcript/concepts?
# - clarity: Is the wording understandable and unambiguous?
# - single_correctness: Is exactly one option clearly correct?
# - distractor_quality: Are wrong options plausible but incorrect?
# - difficulty_fit: Does the labeled difficulty match the question?

# Rules:
# - Use integers from 1 to 5 only.
# - Mark hallucinated = true if the question depends on unsupported information.
# - decision:
#   - keep = good as-is
#   - revise = usable but should be improved
#   - drop = poor or unsafe to use
# - Return strictly valid JSON only.
# """.strip()


# def _eval_user_prompt(
#     segment_text: str,
#     concepts: List[Dict[str, Any]],
#     question: Dict[str, Any]
# ) -> str:
#     concept_payload = []
#     for c in concepts:
#         concept_payload.append({
#             "concept_id": c.get("concept_id", ""),
#             "name": c.get("name", ""),
#             "definition": c.get("definition", ""),
#             "difficulty": c.get("difficulty", ""),
#             "type": c.get("type", ""),
#             "evidence": c.get("evidence", "")
#         })

#     return f"""Transcript segment:
# {segment_text}

# Extracted concepts:
# {json.dumps(concept_payload, indent=2, ensure_ascii=False)}

# Generated MCQ:
# {json.dumps(question, indent=2, ensure_ascii=False)}

# Evaluate this question and return only the JSON object.
# """


# def _strip_code_fence(raw: str) -> str:
#     raw = raw.strip()
#     if raw.startswith("```"):
#         raw = re.sub(r"^```(?:json)?", "", raw).strip()
#         raw = re.sub(r"```$", "", raw).strip()
#     return raw


# def _safe_parse_eval(raw: str) -> Dict[str, Any]:
#     raw = _strip_code_fence(raw)

#     start = raw.find("{")
#     end = raw.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         raw = raw[start:end + 1]

#     try:
#         data = json.loads(raw)
#     except Exception:
#         return {
#             "groundedness": 1,
#             "clarity": 1,
#             "single_correctness": 1,
#             "distractor_quality": 1,
#             "difficulty_fit": 1,
#             "hallucinated": True,
#             "decision": "drop",
#             "notes": "Failed to parse evaluator output."
#         }

#     def clamp_score(value: Any) -> int:
#         try:
#             n = int(value)
#         except Exception:
#             return 1
#         return max(1, min(5, n))

#     decision = str(data.get("decision", "drop")).strip().lower()
#     if decision not in {"keep", "revise", "drop"}:
#         decision = "drop"

#     return {
#         "groundedness": clamp_score(data.get("groundedness", 1)),
#         "clarity": clamp_score(data.get("clarity", 1)),
#         "single_correctness": clamp_score(data.get("single_correctness", 1)),
#         "distractor_quality": clamp_score(data.get("distractor_quality", 1)),
#         "difficulty_fit": clamp_score(data.get("difficulty_fit", 1)),
#         "hallucinated": bool(data.get("hallucinated", False)),
#         "decision": decision,
#         "notes": str(data.get("notes", "")).strip()
#     }


# def evaluate_quiz_bank(
#     quiz_bank: Dict[str, Any],
#     segments: List[Dict[str, Any]],
#     concepts_data: List[Dict[str, Any]],
#     client: CodexLLMClient,
#     sleep_between_calls: float = 0.0
# ) -> Dict[str, Any]:
#     seg_by_id = {seg["segment_id"]: seg for seg in segments}
#     concepts_by_segment = {
#         entry["segment_id"]: entry.get("concepts", [])
#         for entry in concepts_data
#     }

#     results = []
#     kept_questions = []

#     for q in quiz_bank.get("questions", []):
#         segment_id = q["segment_id"]
#         seg = seg_by_id.get(segment_id)
#         if not seg:
#             continue

#         segment_text = seg.get("text", "")
#         concepts = concepts_by_segment.get(segment_id, [])

#         system_prompt = _eval_system_prompt()
#         user_prompt = _eval_user_prompt(segment_text, concepts, q)

#         try:
#             raw = client.generate(system_prompt, user_prompt)
#             eval_result = _safe_parse_eval(raw)
#         except Exception as e:
#             eval_result = {
#                 "groundedness": 1,
#                 "clarity": 1,
#                 "single_correctness": 1,
#                 "distractor_quality": 1,
#                 "difficulty_fit": 1,
#                 "hallucinated": True,
#                 "decision": "drop",
#                 "notes": f"Evaluation request failed: {e}"
#             }

#         row = {
#             "question_id": q.get("question_id", ""),
#             "segment_id": segment_id,
#             "topic": q.get("topic", ""),
#             "question": q.get("question", ""),
#             "evaluation": eval_result
#         }
#         results.append(row)

#         if eval_result["decision"] == "keep" and not eval_result["hallucinated"]:
#             kept_questions.append(q)

#         if sleep_between_calls > 0:
#             time.sleep(sleep_between_calls)

#     summary = {
#         "lecture_id": quiz_bank.get("lecture_id", ""),
#         "original_question_count": len(quiz_bank.get("questions", [])),
#         "kept_question_count": len(kept_questions),
#         "results": results
#     }

#     filtered_quiz_bank = {
#         **quiz_bank,
#         "total_questions": len(kept_questions),
#         "questions": kept_questions
#     }

#     return {
#         "evaluation_summary": summary,
#         "filtered_quiz_bank": filtered_quiz_bank
#     }

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from content_extraction import CodexLLMClient
from quiz_generation import (
    load_segments_and_concepts,
    generate_quiz_bank,
    save_quiz_bank,
)
from eval_quiz import evaluate_quiz_bank


def backup_existing(path_str: str) -> None:
    path = Path(path_str)
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
    path.rename(backup_path)


def save_json(data: object, path_str: str) -> None:
    path = Path(path_str)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    print("[START] run_quiz_generation.py")

    parser = argparse.ArgumentParser(description="Generate quiz questions from existing segments and concepts JSON.")
    parser.add_argument("--segments-input", default="segments_lec4_transformers.json")
    parser.add_argument("--concepts-input", default="concepts_lec4_transformers.json")
    parser.add_argument("--lecture-id", default="lec4_transformers")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--difficulty", default="")
    parser.add_argument("--topic-filter", default="All")
    parser.add_argument("--generator", choices=["v1", "v2"], default="v2")
    parser.add_argument("--quiz-output", default="quiz_bank_lec4_transformers.json")
    parser.add_argument("--eval-output", default="quiz_eval_lec4_transformers.json")
    parser.add_argument("--filtered-quiz-output", default="quiz_bank_lec4_transformers_filtered.json")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--sleep-between-calls", type=float, default=0.0)
    args = parser.parse_args()

    difficulty = args.difficulty.strip().lower() if args.difficulty else ""
    if difficulty == "":
        difficulty = "mixed"

    print(f"[INFO] Segments file: {args.segments_input}")
    print(f"[INFO] Concepts file: {args.concepts_input}")
    print(f"[INFO] Lecture ID: {args.lecture_id}")
    print(f"[INFO] Num questions: {args.num_questions}")
    print(f"[INFO] Difficulty: {difficulty}")
    print(f"[INFO] Topic filter: {args.topic_filter}")
    print(f"[INFO] Generator: {args.generator}")

    print("[STEP] Loading segments and concepts...")
    segments, concepts = load_segments_and_concepts(
        segments_path=args.segments_input,
        concepts_path=args.concepts_input
    )
    print(f"[OK] Loaded {len(segments)} segment entries.")
    print(f"[OK] Loaded {len(concepts)} concept entries.")

    print("[STEP] Initializing LLM client...")
    client = CodexLLMClient()
    print("[OK] LLM client initialized.")

    backup_existing(args.quiz_output)

    print("[STEP] Generating quiz bank...")
    os.environ["QUIZ_GENERATOR"] = args.generator
    quiz_bank = generate_quiz_bank(
        lecture_id=args.lecture_id,
        segments=segments,
        concept_entries=concepts,
        question_count=args.num_questions,
        difficulty=difficulty,
        topic_filter=args.topic_filter,
        client=client,
    )
    save_quiz_bank(quiz_bank, args.quiz_output)

    question_count = quiz_bank.get("metadata", {}).get("question_count", len(quiz_bank.get("questions", [])))
    print(f"[OK] Generated quiz bank with {question_count} question(s).")
    print(f"[OK] Saved quiz bank to: {args.quiz_output}")

    if args.skip_eval:
        print("[DONE] Skipped evaluation.")
        return

    if args.generator != "v1":
        print("[WARN] Evaluation requires v1 (segment-grounded) questions. Skipping evaluation.")
        return

    backup_existing(args.eval_output)
    backup_existing(args.filtered_quiz_output)

    print("[STEP] Evaluating quiz bank...")
    evaluated = evaluate_quiz_bank(
        quiz_bank=quiz_bank,
        segments=segments,
        concepts_data=concepts,
        client=client,
        sleep_between_calls=args.sleep_between_calls
    )

    save_json(evaluated["evaluation_summary"], args.eval_output)
    save_json(evaluated["filtered_quiz_bank"], args.filtered_quiz_output)

    kept_count = evaluated["filtered_quiz_bank"]["total_questions"]
    print(f"[OK] Evaluation complete. Kept {kept_count} question(s).")
    print(f"[OK] Saved evaluation report to: {args.eval_output}")
    print(f"[OK] Saved filtered quiz bank to: {args.filtered_quiz_output}")
    print("[DONE] All finished.")


if __name__ == "__main__":
    main()