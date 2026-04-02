import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Protocol, Optional

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv(override=True)

GENERIC_CONCEPTS = {
    "example", "input", "output", "value", "values", "word", "words",
    "item", "thing", "process", "component"
}

class LLMClientProtocol(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


# class CodexLLMClient:
#     """
#         Uses Groq chat completion API.
#         Assumes API key is provided through environment variable:
#             GROQ_API_KEY
#         Optional environment variable:
#             GROQ_MODEL (default: llama-3.3-70b-versatile)
#     """

#     def __init__(self, model: Optional[str] = None):
#                 # self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#                 # self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#                 api_key = os.getenv("GROQ_API_KEY")
#                 if not api_key:
#                     raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

#                 self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#                 self.client = Groq(api_key=api_key)

#     def generate(self, system_prompt: str, user_prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#         )
#         return response.choices[0].message.content.strip()
# class CodexLLMClient:
#     """
#     Uses either Groq or OpenRouter based on LLM_PROVIDER.

#     Supported providers:
#       - groq
#       - openrouter
#     """

#     def __init__(self, model: Optional[str] = None):
#         self.provider = os.getenv("LLM_PROVIDER", "groq").lower()

#         if self.provider == "groq":
#             api_key = os.getenv("GROQ_API_KEY")
#             if not api_key:
#                 raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

#             self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#             self.client = Groq(api_key=api_key)

#         elif self.provider == "openrouter":
#             api_key = os.getenv("OPENROUTER_API_KEY")
#             if not api_key:
#                 raise ValueError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")

#             self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
#             self.client = OpenAI(
#                 api_key=api_key,
#                 base_url="https://openrouter.ai/api/v1",
#             )

#         else:
#             raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

#     def generate(self, system_prompt: str, user_prompt: str) -> str:
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 temperature=0,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             raise RuntimeError(f"{self.provider} request failed: {e}")

# class CodexLLMClient:
#     """
#     Primary provider: NVIDIA NIM
#     Fallback provider: Mistral

#     You can still override through .env if needed:
#     - PRIMARY_LLM_PROVIDER (default: nim)
#     - FALLBACK_LLM_PROVIDER (default: mistral)

#     Supported providers:
#     - groq
#     - openrouter
#     - mistral
#     - nim
#     """

#     def __init__(self, model: Optional[str] = None):
#         self.primary_provider = os.getenv("PRIMARY_LLM_PROVIDER", "nim").lower()
#         self.fallback_provider = os.getenv("FALLBACK_LLM_PROVIDER", "mistral").lower()

#         # keep this mainly for debug compatibility with your current prints
#         self.provider = self.primary_provider

#         self.primary_model, self.primary_client = self._build_provider_client(
#             provider=self.primary_provider,
#             model_override=model
#         )
#         self.fallback_model, self.fallback_client = self._build_provider_client(
#             provider=self.fallback_provider,
#             model_override=None
#         )

#     def _build_provider_client(self, provider: str, model_override: Optional[str] = None):
#         if provider == "groq":
#             api_key = os.getenv("GROQ_API_KEY")
#             if not api_key:
#                 raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
#             model = model_override or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#             client = Groq(api_key=api_key)
#             return model, client

#         elif provider == "openrouter":
#             api_key = os.getenv("OPENROUTER_API_KEY")
#             if not api_key:
#                 raise ValueError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")
#             model = model_override or os.getenv("OPENROUTER_MODEL", "openrouter/free")
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url="https://openrouter.ai/api/v1",
#             )
#             return model, client

#         elif provider == "mistral":
#             api_key = os.getenv("MISTRAL_API_KEY")
#             if not api_key:
#                 raise ValueError("MISTRAL_API_KEY is not set. Please add it to your .env file.")
#             model = model_override or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url="https://api.mistral.ai/v1",
#             )
#             return model, client

#         elif provider == "nim":
#             api_key = os.getenv("NIM_API_KEY")
#             if not api_key:
#                 raise ValueError("NIM_API_KEY is not set. Please add it to your .env file.")
#             model = model_override or os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
#             client = OpenAI(
#                 api_key=api_key,
#                 base_url="https://integrate.api.nvidia.com/v1",
#             )
#             return model, client

#         else:
#             raise ValueError(f"Unsupported provider: {provider}")

#     def _extract_content(self, response) -> str:
#         if not getattr(response, "choices", None):
#             raise RuntimeError(f"{self.provider} returned no choices: {response}")

#         message = response.choices[0].message
#         content = getattr(message, "content", None)

#         if isinstance(content, str):
#             return content.strip()

#         if isinstance(content, list):
#             parts = []
#             for item in content:
#                 if isinstance(item, dict) and item.get("type") == "text":
#                     parts.append(item.get("text", ""))
#                 else:
#                     text = getattr(item, "text", None)
#                     if text:
#                         parts.append(text)
#             joined = "".join(parts).strip()
#             if joined:
#                 return joined

#         raise RuntimeError(f"{self.provider} returned empty message content: {response}")

#     def _request(self, client, model: str, provider: str, system_prompt: str, user_prompt: str) -> str:
#         self.provider = provider
#         print(f"[LLM] Provider: {provider}")
#         print(f"[LLM] Model: {model}")
#         print("[LLM] Sending request...")

#         response = client.chat.completions.create(
#             model=model,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             timeout=60,
#         )

#         print("[LLM] Response received.")
#         return self._extract_content(response)

#     def generate(self, system_prompt: str, user_prompt: str) -> str:
#         primary_error = None

#         try:
#             return self._request(
#                 client=self.primary_client,
#                 model=self.primary_model,
#                 provider=self.primary_provider,
#                 system_prompt=system_prompt,
#                 user_prompt=user_prompt,
#             )
#         except Exception as e:
#             primary_error = e
#             print(f"[LLM] Primary provider failed ({self.primary_provider}): {e}")
#             print(f"[LLM] Falling back to {self.fallback_provider}...")

#         try:
#             return self._request(
#                 client=self.fallback_client,
#                 model=self.fallback_model,
#                 provider=self.fallback_provider,
#                 system_prompt=system_prompt,
#                 user_prompt=user_prompt,
#             )
#         except Exception as fallback_error:
#             print(f"[LLM] Fallback provider failed ({self.fallback_provider}): {fallback_error}")
#             raise RuntimeError(
#                 f"Both providers failed. "
#                 f"Primary ({self.primary_provider}): {primary_error} | "
#                 f"Fallback ({self.fallback_provider}): {fallback_error}"
#             )
class CodexLLMClient:
    """
    Primary provider: NVIDIA NIM
    Fallback provider: Mistral
    """

    def __init__(
        self,
        model: Optional[str] = None,
        primary_provider: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        enable_fallback: bool = True,
    ):
        self.primary_provider = (primary_provider or os.getenv("PRIMARY_LLM_PROVIDER", "nim")).lower()
        resolved_fallback = fallback_provider if fallback_provider is not None else os.getenv("FALLBACK_LLM_PROVIDER", "mistral")
        self.fallback_provider = (resolved_fallback or "").lower()
        self.enable_fallback = bool(enable_fallback and self.fallback_provider)

        self.primary_model, self.primary_client = self._build_provider_client(
            provider=self.primary_provider,
            model_override=model
        )
        self.fallback_model = None
        self.fallback_client = None
        if self.enable_fallback:
            self.fallback_model, self.fallback_client = self._build_provider_client(
                provider=self.fallback_provider,
                model_override=None
            )

        # keep these so old code that expects self.provider/self.model/self.client will not break
        self.provider = self.primary_provider
        self.model = self.primary_model
        self.client = self.primary_client

    def _build_provider_client(self, provider: str, model_override: Optional[str] = None):
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
            model = model_override or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            client = Groq(api_key=api_key)
            return model, client

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")
            model = model_override or os.getenv("OPENROUTER_MODEL", "openrouter/free")
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            return model, client

        elif provider == "mistral":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY is not set. Please add it to your .env file.")
            model = model_override or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
            )
            return model, client

        elif provider == "nim":
            api_key = os.getenv("NIM_API_KEY")
            if not api_key:
                raise ValueError("NIM_API_KEY is not set. Please add it to your .env file.")
            model = model_override or os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
            client = OpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
            return model, client

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _extract_content(self, response) -> str:
        if not getattr(response, "choices", None):
            raise RuntimeError(f"{self.provider} returned no choices: {response}")

        message = response.choices[0].message
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
            joined = "".join(parts).strip()
            if joined:
                return joined

        raise RuntimeError(f"{self.provider} returned empty message content: {response}")

    def _request(self, client, model: str, provider: str, system_prompt: str, user_prompt: str) -> str:
        self.provider = provider
        self.model = model
        self.client = client

        print(f"[LLM] Provider: {provider}")
        print(f"[LLM] Model: {model}")
        print("[LLM] Sending request...")

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=60,
        )

        print("[LLM] Response received.")
        return self._extract_content(response)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        primary_error = None

        try:
            return self._request(
                client=self.primary_client,
                model=self.primary_model,
                provider=self.primary_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            primary_error = e
            print(f"[LLM] Primary provider failed ({self.primary_provider}): {e}")
            if not self.enable_fallback:
                raise RuntimeError(
                    f"Primary provider failed ({self.primary_provider}) and fallback is disabled: {e}"
                )

            print(f"[LLM] Falling back to {self.fallback_provider}...")

        try:
            return self._request(
                client=self.fallback_client,
                model=self.fallback_model,
                provider=self.fallback_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as fallback_error:
            print(f"[LLM] Fallback provider failed ({self.fallback_provider}): {fallback_error}")
            raise RuntimeError(
                f"Both providers failed. "
                f"Primary ({self.primary_provider}): {primary_error} | "
                f"Fallback ({self.fallback_provider}): {fallback_error}"
            )

    # def generate(self, system_prompt: str, user_prompt: str) -> str:
    #     # response = self.client.chat.completions.create(
    #     #     model=self.model,
    #     #     temperature=0,
    #     #     messages=[
    #     #         {"role": "system", "content": system_prompt},
    #     #         {"role": "user", "content": user_prompt},
    #     #     ],
    #     # )
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             temperature=0,
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt},
    #             ],
    #         )
    #     except Exception as e:
    #         raise RuntimeError(f"{self.provider} request failed: {e}")


    #     # print("DEBUG raw response:", response)

    #     if not getattr(response, "choices", None):
    #         raise RuntimeError(f"{self.provider} returned no choices: {response}")

    #     message = response.choices[0].message
    #     content = getattr(message, "content", None)

    #     if isinstance(content, str):
    #         return content.strip()

    #     if isinstance(content, list):
    #         parts = []
    #         for item in content:
    #             if isinstance(item, dict) and item.get("type") == "text":
    #                 parts.append(item.get("text", ""))
    #             else:
    #                 text = getattr(item, "text", None)
    #                 if text:
    #                     parts.append(text)
    #         joined = "".join(parts).strip()
    #         if joined:
    #             return joined

    #     raise RuntimeError(f"{self.provider} returned empty message content: {response}")
    
def _concept_system_prompt() -> str:
#     return """
# You are an assistant that extracts key learning concepts from lecture transcript segments.

# For each input segment, identify 2 to 3 important technical concepts that would be useful for downstream multiple-choice question generation.

# Return ONLY valid JSON with this exact structure:
# {
#   "concepts": [
#     {
#       "name": "short concept name",
#       "definition": "Exactly one clear sentence definition suitable for students.",
#       "difficulty": "easy|medium|hard",
#       "type": "definition|formula|example|procedure"
#     }
#   ]
# }

# Guidelines:
# - "name" must be a short phrase, ideally 2 to 6 words.
# - "definition" must be exactly one sentence and concise.
# - "difficulty" should reflect how hard the concept is for a typical student.
# - "type":
#   - "definition" for key ideas or terms
#   - "formula" for equations or quantitative relations
#   - "example" for illustrative worked ideas or examples
#   - "procedure" for multi-step methods or algorithms
# - Do not include extra commentary, markdown, or explanations.
# - Concepts must be grounded in the segment text; do not add outside facts or corrections.
# - Prefer terms and phrasing that appear in the segment; avoid inventing new terminology.
# - Skip generic concepts unless central to the segment.
# - If the segment is weak or repetitive, still extract the 1 to 3 most meaningful concepts.
# """.strip()
    return """
You are an assistant that extracts key learning concepts from lecture transcript segments.

For each input segment, identify 2 to 3 important technical concepts that would be useful for downstream multiple-choice question generation.

Return ONLY valid JSON with this exact structure:
{
  "concepts": [
    {
      "name": "short concept name",
      "definition": "Exactly one clear sentence definition suitable for students.",
      "difficulty": "easy|medium|hard",
      "type": "definition|formula|example|procedure",
      "evidence": "short phrase from the segment supporting the concept"
    }
  ]
}

Guidelines:
- "name" must be a short, specific technical phrase, ideally 2 to 6 words.
- "name" should match the lecture wording closely, but clean up obvious ASR mistakes.
- Do not invent awkward names such as partial phrases or unnatural terms.
- Avoid generic names like "process", "item", "thing", "value", "example", or "input" unless absolutely central.
- "definition" must be exactly one clear sentence written simply for students.
- "definition" must explain what the concept is in the context of the lecture, not just what it helps do.
- Avoid vague wording such as "a component", "a technique", or "something that helps" unless followed by a precise explanation.
- Prefer concrete and specific definitions over broad textbook-style definitions.
- If a concept is too broad for the segment, do not include it.
- "difficulty" should reflect how hard the concept is for a typical student.
- "type":
  - "definition" for key ideas or terms
  - "formula" for equations or quantitative relations
  - "example" for illustrative worked ideas or examples
  - "procedure" for multi-step methods or algorithms
- "evidence" must be a short supporting phrase directly grounded in the segment.
- "evidence" should support the concept clearly, not just be a vague nearby phrase.
- Do not include extra commentary, markdown, or explanations.
- Concepts must be grounded in the segment text; do not add outside facts or corrections.
- Prefer terms and phrasing that appear in the segment; avoid inventing new terminology.
- Skip generic or repetitive concepts unless they are central to the segment.
- If the segment is weak or repetitive, extract only the 1 to 3 most meaningful concepts.
- Return strictly valid JSON.
- Every field value must be a single valid JSON string.
- Do not join two quoted phrases with words like "and" inside one JSON value.
- If evidence has multiple candidate phrases, choose only one short phrase.
""".strip()


# def _concept_user_prompt(segment_text: str, topic: str = "") -> str:
#     if topic:
#         return f"""Segment topic hint: {topic}

# Segment text:
# {segment_text}

# Extract 2 to 3 key concepts and return only the JSON object."""
#     return f"""Segment text:
# {segment_text}

# Extract 2 to 3 key concepts and return only the JSON object."""

def _concept_user_prompt(segment_text: str, topic: str = "") -> str:
    if topic:
        return f"""Segment topic hint: {topic}

Segment text:
{segment_text}

Extract 2 to 3 key concepts from this segment.

Choose concepts that are:
- specific to this segment
- useful for quiz generation
- clearly supported by the segment text

Avoid:
- concepts that are too broad for the segment
- vague or generic concept names
- awkward names caused by transcript errors

Return only the JSON object."""
    return f"""Segment text:
{segment_text}

Extract 2 to 3 key concepts from this segment.

Choose concepts that are:
- specific to this segment
- useful for quiz generation
- clearly supported by the segment text

Avoid:
- concepts that are too broad for the segment
- vague or generic concept names
- awkward names caused by transcript errors

Return only the JSON object."""


# def _safe_parse_concepts(raw: str) -> List[Dict[str, Any]]:
#     """
#     Robustly parse LLM JSON output.
#     """
#     raw = raw.strip()

#     # Remove fenced code blocks if the model adds them accidentally
#     if raw.startswith("```"):
#         raw = raw.strip("`")
#         raw = raw.replace("json\n", "", 1).strip()

#     try:
#         data = json.loads(raw)
#         concepts = data.get("concepts", [])
#         if isinstance(concepts, list):
#             return concepts
#         return []
#     except Exception:
#         return []

def _safe_parse_concepts(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
        concepts = data.get("concepts", [])
        if isinstance(concepts, list):
            return concepts
        return []
    except Exception:
        return []


# def _call_llm_for_concepts(
#     client: LLMClientProtocol,
#     text: str,
#     topic: str = ""
# ) -> List[Dict[str, Any]]:
#     system_prompt = _concept_system_prompt()
#     user_prompt = _concept_user_prompt(text, topic)
#     raw = client.generate(system_prompt, user_prompt)
#     return _safe_parse_concepts(raw)

def _call_llm_for_concepts(
    client: LLMClientProtocol,
    text: str,
    topic: str = ""
) -> List[Dict[str, Any]]:
    system_prompt = _concept_system_prompt()
    user_prompt = _concept_user_prompt(text, topic)
    try:
        raw = client.generate(system_prompt, user_prompt)
    except Exception as e:
        print(f"Concept extraction failed for topic '{topic}': {e}")
        return []
    return _safe_parse_concepts(raw)


def extract_key_concepts(
    segments: List[Dict[str, Any]],
    client: LLMClientProtocol,
    sleep_between_calls: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Input schema for each segment:
    {
      "lecture_id": "lec4_transformers",
      "segment_id": 1,
      "text": "...",
      "topic": "..."
    }

    Output schema:
    [
      {
        "lecture_id": "...",
        "segment_id": ...,
        "concepts": [
          {
            "concept_id": "...",
            "name": "...",
            "definition": "...",
            "topic": "...",
            "difficulty": "...",
            "type": "...",
            "evidence": "..."
          }
        ]
      }
    ]
    """
    results: List[Dict[str, Any]] = []

    for seg in segments:
        lecture_id = seg["lecture_id"]
        segment_id = seg["segment_id"]
        text = seg["text"]
        topic = seg.get("topic", "")

        raw_concepts = _call_llm_for_concepts(client, text, topic)
        generation_provider = str(getattr(client, "provider", "")).strip().lower()
        generation_model = str(getattr(client, "model", "")).strip()

        concepts = []

        seen_names = set()

        for idx, concept in enumerate(raw_concepts, start=1):
            name = str(concept.get("name", "")).strip()
            definition = str(concept.get("definition", "")).strip()
            difficulty = str(concept.get("difficulty", "medium")).strip().lower()
            ctype = str(concept.get("type", "definition")).strip().lower()
            evidence = str(concept.get("evidence", "")).strip()

            if not name or not definition or not evidence:
                continue

            if difficulty not in {"easy", "medium", "hard"}:
                difficulty = "medium"

            if ctype not in {"definition", "formula", "example", "procedure"}:
                ctype = "definition"

            if name.lower() in GENERIC_CONCEPTS:
                continue
            
            normalized_name = " ".join(name.lower().split())
            bad_words = {"item", "thing"}
            if any(word in normalized_name.split() for word in bad_words):
                continue
            if normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)

            concepts.append({
                "concept_id": f"{lecture_id}_s{segment_id}_c{idx}",
                "name": name,
                "definition": definition,
                "topic": topic,
                "difficulty": difficulty,
                "type": ctype,
                "evidence": evidence
            })

        results.append({
            "lecture_id": lecture_id,
            "segment_id": segment_id,
            "generation_provider": generation_provider,
            "generation_model": generation_model,
            "concepts": concepts
        })

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return results


def load_segments(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_concepts(data: List[Dict[str, Any]], output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
