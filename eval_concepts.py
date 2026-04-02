import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from content_extraction import CodexLLMClient

load_dotenv(override=True)


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _eval_system_prompt() -> str:
#     return """
# You are a strict evaluator of extracted concepts from transcript segments.
# Rate each concept for factual support by the provided segment text only.
# Return ONLY valid JSON.
# """.strip()
    return """
    You are a strict evaluator of extracted concepts from transcript segments.

    Return ONLY a valid JSON array.
    Do not return markdown.
    Do not return explanations.
    Do not wrap the array in an object.

    Each array item must have:
    - concept_id
    - support
    - clarity
    - hallucinated
    - notes
    """.strip()


def _eval_user_prompt(segment_text: str, concepts: List[Dict[str, str]]) -> str:
    payload = json.dumps(concepts, ensure_ascii=False, indent=2)
    return f"""Segment text:
{segment_text}

Concepts JSON:
{payload}

Return a JSON array where each item has:
- concept_id
- support (1-5, 5 means fully supported by the segment, 1 means not supported)
- clarity (1-5, 5 means concise and clear)
- hallucinated (true if not supported by the segment)
- notes (short reason)
""".strip()


# def _safe_parse(raw: str) -> List[Dict[str, Any]]:
#     raw = raw.strip()
#     if raw.startswith("```"):
#         raw = raw.strip("`")
#         raw = raw.replace("json\n", "", 1).strip()
#     try:
#         data = json.loads(raw)
#         if isinstance(data, list):
#             return data
#         return []
#     except Exception:
#         return []

def _safe_parse(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()

    # Try to isolate a JSON array first
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # Fallback: try full JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("evaluations"), list):
                return data["evaluations"]
            if isinstance(data.get("results"), list):
                return data["results"]
        return []
    except Exception:
        return []


# def evaluate_concepts(
#     segments: List[Dict[str, Any]],
#     concepts: List[Dict[str, Any]],
#     model: str,
#     api_key: str
# ) -> Dict[str, Any]:
#     client = Groq(api_key=api_key)

#     concepts_by_segment: Dict[int, List[Dict[str, str]]] = {}
#     for entry in concepts:
#         seg_id = entry["segment_id"]
#         items = []
#         for concept in entry.get("concepts", []):
#             items.append({
#                 "concept_id": concept.get("concept_id", ""),
#                 "name": concept.get("name", ""),
#                 "definition": concept.get("definition", ""),
#             })
#         concepts_by_segment[seg_id] = items

#     results = []
#     support_scores = []
#     clarity_scores = []
#     hallucinated_count = 0
#     total_concepts = 0

#     for seg in segments:
#         seg_id = seg["segment_id"]
#         seg_concepts = concepts_by_segment.get(seg_id, [])
#         if not seg_concepts:
#             continue

#         user_prompt = _eval_user_prompt(seg["text"], seg_concepts)
#         # response = client.chat.completions.create(
#         #     model=model,
#         #     temperature=0,
#         #     messages=[
#         #         {"role": "system", "content": _eval_system_prompt()},
#         #         {"role": "user", "content": user_prompt},
#         #     ],
#         # )

#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 temperature=0,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#             )
#         except Exception as e:
#             print(f"Skipping one concept due to API error: {e}")
#             continue

#         raw = response.choices[0].message.content.strip()
#         eval_items = _safe_parse(raw)

#         for item in eval_items:
#             support = int(item.get("support", 0))
#             clarity = int(item.get("clarity", 0))
#             hallucinated = bool(item.get("hallucinated", False))
#             if support:
#                 support_scores.append(support)
#             if clarity:
#                 clarity_scores.append(clarity)
#             if hallucinated:
#                 hallucinated_count += 1
#             total_concepts += 1

#         results.append({
#             "segment_id": seg_id,
#             "evaluations": eval_items,
#         })

#     summary = {
#         "total_concepts": total_concepts,
#         "avg_support": sum(support_scores) / max(len(support_scores), 1),
#         "avg_clarity": sum(clarity_scores) / max(len(clarity_scores), 1),
#         "hallucination_rate": hallucinated_count / max(total_concepts, 1),
#     }

#     return {
#         "summary": summary,
#         "results": results,
#     }
# def evaluate_concepts(
#     segments: List[Dict[str, Any]],
#     concepts: List[Dict[str, Any]],
#     model: str,
#     api_key: str
# ) -> Dict[str, Any]:
#     # client = Groq(api_key=api_key)
#     provider = os.getenv("LLM_PROVIDER", "groq").lower()

#     if provider == "groq":
#         client = Groq(api_key=api_key)
#     elif provider == "openrouter":
#         client = OpenAI(
#             api_key=api_key,
#             base_url="https://openrouter.ai/api/v1",
#         )
#     elif provider == "mistral":
#         client = OpenAI(
#             api_key=api_key,
#             base_url="https://api.mistral.ai/v1",
#         )
#     elif provider == "nim":
#         client = OpenAI(
#             api_key=api_key,
#             base_url="https://integrate.api.nvidia.com/v1",
#         )
#     else:
#         raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

#     concepts_by_segment: Dict[int, List[Dict[str, str]]] = {}
#     print("DEBUG concept entries loaded:", len(concepts))
#     for entry in concepts:
#         print("DEBUG segment entry:", entry.get("segment_id"), "concept count:", len(entry.get("concepts", [])))
#         seg_id = entry["segment_id"]
#         items = []
#         for concept in entry.get("concepts", []):
#             items.append({
#                 "concept_id": concept.get("concept_id", ""),
#                 "name": concept.get("name", ""),
#                 "definition": concept.get("definition", ""),
#             })
#         concepts_by_segment[seg_id] = items

#     results = []
#     support_scores = []
#     clarity_scores = []
#     hallucinated_count = 0
#     total_concepts = 0

#     system_prompt = """
# You are a strict evaluator of extracted concepts from transcript segments.
# Rate each concept for factual support by the provided segment text only.
# Return ONLY valid JSON.
# """.strip()

#     for seg in segments:
#         seg_id = seg["segment_id"]
#         seg_concepts = concepts_by_segment.get(seg_id, [])
#         if not seg_concepts:
#             continue

#         user_prompt = _eval_user_prompt(seg["text"], seg_concepts)

#         # try:
#         #     response = client.chat.completions.create(
#         #         model=model,
#         #         temperature=0,
#         #         messages=[
#         #             {"role": "system", "content": system_prompt},
#         #             {"role": "user", "content": user_prompt},
#         #         ],
#         #     )
#         # except Exception as e:
#         #     print(f"Skipping segment {seg_id} due to API error: {e}")
#         #     continue

#         # raw = response.choices[0].message.content.strip()
#         # eval_items = _safe_parse(raw)

#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 temperature=0,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#             )
#             raw = response.choices[0].message.content.strip()
#             print(f"DEBUG eval raw for segment {seg_id}: {raw}")
#         except Exception as e:
#             print(f"Skipping segment {seg_id} due to API error: {e}")
#             continue

#         eval_items = _safe_parse(raw)
#         print(f"DEBUG parsed eval items for segment {seg_id}: {eval_items}")

#         for item in eval_items:
#             support = int(item.get("support", 0))
#             clarity = int(item.get("clarity", 0))
#             hallucinated = bool(item.get("hallucinated", False))

#             if support:
#                 support_scores.append(support)
#             if clarity:
#                 clarity_scores.append(clarity)
#             if hallucinated:
#                 hallucinated_count += 1
#             total_concepts += 1

#         results.append({
#             "segment_id": seg_id,
#             "evaluations": eval_items,
#         })

#     summary = {
#         "total_concepts": total_concepts,
#         "avg_support": sum(support_scores) / max(len(support_scores), 1),
#         "avg_clarity": sum(clarity_scores) / max(len(clarity_scores), 1),
#         "hallucination_rate": hallucinated_count / max(total_concepts, 1),
#     }

#     return {
#         "summary": summary,
#         "results": results,
#     }

# def evaluate_concepts(
#     segments: List[Dict[str, Any]],
#     concepts: List[Dict[str, Any]],
#     model: str,
#     api_key: str
# ) -> Dict[str, Any]:
#     client = CodexLLMClient(model=model)

#     concepts_by_segment: Dict[int, List[Dict[str, str]]] = {}
#     print("DEBUG concept entries loaded:", len(concepts))
#     for entry in concepts:
#         print("DEBUG segment entry:", entry.get("segment_id"), "concept count:", len(entry.get("concepts", [])))
#         seg_id = entry["segment_id"]
#         items = []
#         for concept in entry.get("concepts", []):
#             items.append({
#                 "concept_id": concept.get("concept_id", ""),
#                 "name": concept.get("name", ""),
#                 "definition": concept.get("definition", ""),
#             })
#         concepts_by_segment[seg_id] = items

#     results = []
#     support_scores = []
#     clarity_scores = []
#     hallucinated_count = 0
#     total_concepts = 0

#     system_prompt = """
# You are a strict evaluator of extracted concepts from transcript segments.
# Rate each concept for factual support by the provided segment text only.
# Return ONLY valid JSON.
# """.strip()

#     for seg in segments:
#         seg_id = seg["segment_id"]
#         seg_concepts = concepts_by_segment.get(seg_id, [])
#         if not seg_concepts:
#             continue

#         user_prompt = _eval_user_prompt(seg["text"], seg_concepts)

#         try:
#             raw = client.generate(system_prompt, user_prompt)
#             print(f"DEBUG eval raw for segment {seg_id}: {raw}")
#         except Exception as e:
#             print(f"Skipping segment {seg_id} due to API error: {e}")
#             continue

#         eval_items = _safe_parse(raw)
#         print(f"DEBUG parsed eval items for segment {seg_id}: {eval_items}")

#         for item in eval_items:
#             support = int(item.get("support", 0))
#             clarity = int(item.get("clarity", 0))
#             hallucinated = bool(item.get("hallucinated", False))

#             if support:
#                 support_scores.append(support)
#             if clarity:
#                 clarity_scores.append(clarity)
#             if hallucinated:
#                 hallucinated_count += 1
#             total_concepts += 1

#         results.append({
#             "segment_id": seg_id,
#             "evaluations": eval_items,
#         })

#     summary = {
#         "total_concepts": total_concepts,
#         "avg_support": sum(support_scores) / max(len(support_scores), 1),
#         "avg_clarity": sum(clarity_scores) / max(len(clarity_scores), 1),
#         "hallucination_rate": hallucinated_count / max(total_concepts, 1),
#     }

#     return {
#         "summary": summary,
#         "results": results,
#     }
def evaluate_concepts(
    segments: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    model: str,
    api_key: str
) -> Dict[str, Any]:
    client = CodexLLMClient(model=model)

    concepts_by_segment: Dict[int, List[Dict[str, str]]] = {}
    print("DEBUG concept entries loaded:", len(concepts))
    for entry in concepts:
        print("DEBUG segment entry:", entry.get("segment_id"), "concept count:", len(entry.get("concepts", [])))
        seg_id = entry["segment_id"]
        items = []
        for concept in entry.get("concepts", []):
            items.append({
                "concept_id": concept.get("concept_id", ""),
                "name": concept.get("name", ""),
                "definition": concept.get("definition", ""),
            })
        concepts_by_segment[seg_id] = items

    results = []
    support_scores = []
    clarity_scores = []
    hallucinated_count = 0
    total_concepts = 0

    system_prompt = """
You are a strict evaluator of extracted concepts from transcript segments.
Rate each concept for factual support by the provided segment text only.
Return ONLY valid JSON.
""".strip()

    for seg in segments:
        seg_id = seg["segment_id"]
        seg_concepts = concepts_by_segment.get(seg_id, [])
        if not seg_concepts:
            continue

        user_prompt = _eval_user_prompt(seg["text"], seg_concepts)

        try:
            raw = client.generate(system_prompt, user_prompt)
            print(f"DEBUG eval raw for segment {seg_id}: {raw}")
        except Exception as e:
            print(f"Skipping segment {seg_id} due to API error: {e}")
            continue

        eval_items = _safe_parse(raw)
        print(f"DEBUG parsed eval items for segment {seg_id}: {eval_items}")

        for item in eval_items:
            support = int(item.get("support", 0))
            clarity = int(item.get("clarity", 0))
            hallucinated = bool(item.get("hallucinated", False))

            if support:
                support_scores.append(support)
            if clarity:
                clarity_scores.append(clarity)
            if hallucinated:
                hallucinated_count += 1
            total_concepts += 1

        results.append({
            "segment_id": seg_id,
            "evaluations": eval_items,
        })

    summary = {
        "total_concepts": total_concepts,
        "avg_support": sum(support_scores) / max(len(support_scores), 1),
        "avg_clarity": sum(clarity_scores) / max(len(clarity_scores), 1),
        "hallucination_rate": hallucinated_count / max(total_concepts, 1),
    }

    return {
        "summary": summary,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extracted concepts for segment support.")
    parser.add_argument("--segments", default="segments_lec4_transformers.json")
    parser.add_argument("--concepts", default="concepts_lec4_transformers.json")
    parser.add_argument("--output", default="concept_eval.json")
    # parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    # parser.add_argument("--model", default=os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant"))
    # parser.add_argument("--model", default=os.getenv("OPENROUTER_EVAL_MODEL", "openrouter/free"))
    parser.add_argument("--max-segments", type=int, default=10)

    # provider = os.getenv("LLM_PROVIDER", "groq").lower()

    # if provider == "groq":
    #     default_model = os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant")
    # elif provider == "openrouter":
    #     default_model = os.getenv("OPENROUTER_EVAL_MODEL", "openrouter/free")
    # elif provider == "mistral":
    #     default_model = os.getenv("MISTRAL_EVAL_MODEL", "mistral-small-latest")
    # elif provider == "nim":
    #     default_model = os.getenv("NIM_EVAL_MODEL", "meta/llama-3.1-8b-instruct")
    # else:
    #     raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    # parser.add_argument("--model", default=default_model)

    default_model = os.getenv("NIM_EVAL_MODEL", os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct"))
    parser.add_argument("--model", default=default_model)

    args = parser.parse_args()

    segments = load_json(args.segments)
    concepts = load_json(args.concepts)

    if args.max_segments and len(concepts) > args.max_segments:
        concepts = concepts[:args.max_segments]

    # api_key = os.getenv("GROQ_API_KEY")
    # if not api_key:
    #     raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    # if provider == "groq":
    #     api_key = os.getenv("GROQ_API_KEY")
    # elif provider == "openrouter":
    #     api_key = os.getenv("OPENROUTER_API_KEY")
    # elif provider == "mistral":
    #     api_key = os.getenv("MISTRAL_API_KEY")
    # elif provider == "nim":
    #     api_key = os.getenv("NIM_API_KEY")
    # else:
    #     raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    # if not api_key:
    #     raise ValueError(f"Missing API key for provider: {provider}")
    api_key = ""
    
    report = evaluate_concepts(
        segments=segments,
        concepts=concepts,
        model=args.model,
        api_key=api_key,
    )

    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation report to: {args.output}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()
