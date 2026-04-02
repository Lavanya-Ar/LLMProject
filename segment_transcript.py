import json
import re
import os
import html
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from content_extraction import CodexLLMClient

load_dotenv(override=True)


def _clean_processed_txt_export(raw_text: str) -> str:
    """
    Clean page-based enriched TXT exports from processed_data.
    Removes page separators and repetitive diagram-description boilerplate.
    """
    cleaned_lines: List[str] = []

    for raw_line in raw_text.splitlines():
        line = html.unescape(raw_line).strip()

        if not line:
            cleaned_lines.append("")
            continue

        if re.fullmatch(r"=+", line):
            continue
        if re.fullmatch(r"📄\s*PAGE\s*\d+", line, flags=re.IGNORECASE):
            continue
        if line == "---":
            continue
        if re.match(r"^>\s*\*\*Diagram Description:\s*", line, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    # Collapse excessive blank lines while preserving paragraph boundaries.
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_transcript(path: str) -> str:
    transcript_path = Path(path)
    text = transcript_path.read_text(encoding="utf-8").strip()

    # Support processed_data enriched TXT exports without requiring JSON conversion.
    if transcript_path.suffix.lower() == ".txt" and "📄 PAGE" in text:
        return _clean_processed_txt_export(text)

    return text


def split_paragraphs(text: str) -> List[str]:
    """
    Split transcript into rough blocks using blank lines first.
    Falls back to sentence grouping if transcript is one large block.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) > 1:
        return blocks

    # Fallback: split by sentence boundaries and group into chunks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    grouped = []
    chunk = []
    word_count = 0

    for sent in sentences:
        chunk.append(sent)
        word_count += len(sent.split())
        # if word_count >= 120:
        if word_count >= 220:
            grouped.append(" ".join(chunk).strip())
            chunk = []
            word_count = 0

    if chunk:
        grouped.append(" ".join(chunk).strip())

    return grouped


# def infer_topic(segment_text: str, segment_id: int) -> str:
#     """
#     Lightweight heuristic topic label from the first few informative words.
#     """
#     text = re.sub(r"\s+", " ", segment_text).strip()
#     text = re.sub(r"[^A-Za-z0-9\s\-]", "", text)
#     words = text.split()

#     stopwords = {
#         "the", "a", "an", "and", "or", "but", "so", "because", "of", "to", "in",
#         "on", "for", "with", "this", "that", "these", "those", "is", "are", "was",
#         "were", "be", "by", "as", "at", "it", "we", "you", "they", "he", "she",
#         "i", "from", "our", "their", "his", "her"
#     }

#     filtered = [w for w in words if w.lower() not in stopwords]
#     if not filtered:
#         return f"Segment {segment_id}"

#     topic = " ".join(filtered[:4])
#     return topic[:80]

def infer_topic_heuristic(segment_text: str, segment_id: int) -> str:
    """
    Fallback heuristic topic label if LLM topic generation fails.
    Uses frequent meaningful words instead of the first few words.
    """
    text = re.sub(r"\s+", " ", segment_text).strip().lower()
    text = re.sub(r"[^a-z\s\-]", "", text)

    stopwords = {
        "the", "a", "an", "and", "or", "but", "so", "because", "of", "to", "in",
        "on", "for", "with", "this", "that", "these", "those", "is", "are", "was",
        "were", "be", "by", "as", "at", "it", "we", "you", "they", "he", "she",
        "i", "from", "our", "their", "his", "her", "all", "right", "okay", "ok",
        "now", "then", "just", "like", "uh", "um", "yeah", "yes", "no", "if",
        "start", "one", "two", "three"
    }

    words = [w for w in text.split() if w not in stopwords and len(w) > 2]

    if not words:
        return f"Segment {segment_id}"

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    top_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:4]
    topic = " ".join(w for w, _ in top_words).title()

    return topic if topic else f"Segment {segment_id}"

# class TopicLabeler:
#     # def __init__(self, model: Optional[str] = None):
#         # api_key = os.getenv("GROQ_API_KEY")
#         # if not api_key:
#         #     raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

#         # self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#         # self.client = Groq(api_key=api_key)
#         # self.provider = os.getenv("LLM_PROVIDER", "groq").lower()

#         # if self.provider == "groq":
#         #     api_key = os.getenv("GROQ_API_KEY")
#         #     if not api_key:
#         #         raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
#         #     self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
#         #     self.client = Groq(api_key=api_key)

#         # elif self.provider == "openrouter":
#         #     api_key = os.getenv("OPENROUTER_API_KEY")
#         #     if not api_key:
#         #         raise ValueError("OPENROUTER_API_KEY is not set. Please add it to your .env file.")
#         #     self.model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
#         #     self.client = OpenAI(
#         #         api_key=api_key,
#         #         base_url="https://openrouter.ai/api/v1",
#         #     )
#         # elif self.provider == "mistral":
#         #     api_key = os.getenv("MISTRAL_API_KEY")
#         #     if not api_key:
#         #         raise ValueError("MISTRAL_API_KEY is not set. Please add it to your .env file.")
#         #     self.model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
#         #     self.client = OpenAI(
#         #         api_key=api_key,
#         #         base_url="https://api.mistral.ai/v1",
#         #     )
#         # elif self.provider == "nim":
#         #     api_key = os.getenv("NIM_API_KEY")
#         #     if not api_key:
#         #         raise ValueError("NIM_API_KEY is not set. Please add it to your .env file.")
#         #     self.model = model or os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
#         #     self.client = OpenAI(
#         #         api_key=api_key,
#         #         base_url="https://integrate.api.nvidia.com/v1",
#         #     )
#         # else:
#         #     raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")
#     def __init__(self, model: Optional[str] = None):
#             self.llm = CodexLLMClient(model=model)

# #     def generate_topic(self, segment_text: str, segment_id: int) -> str:
# #         """
# #         Generate a short lecture-style topic title for one segment.
# #         """
# #         truncated_text = segment_text[:1800]

# #         system_prompt = """
# # You label lecture transcript segments with short topic titles.

# # Return ONLY the topic title as plain text.
# # Do not return JSON.
# # Do not return quotes.
# # Do not return explanations.

# # Rules:
# # - 2 to 6 words only
# # - Use lecture-style wording
# # - Focus on the main technical concept
# # - Avoid filler words, numbers, and sentence fragments
# # - Avoid generic titles like "Introduction" or "Example" unless truly appropriate
# # """.strip()

# #         user_prompt = f"""Transcript segment:
# # {truncated_text}

# # Write one short topic title only."""

# #         try:
# #         #     response = self.client.chat.completions.create(
# #         #         model=self.model,
# #         #         temperature=0,
# #         #         messages=[
# #         #             {"role": "system", "content": system_prompt},
# #         #             {"role": "user", "content": user_prompt},
# #         #         ],
# #         #     )

# #         #     topic = response.choices[0].message.content.strip()
# #         #     topic = topic.replace('"', "").replace("'", "").strip()
# #         #     topic = re.sub(r"\s+", " ", topic)

# #         #     if not topic or len(topic.split()) > 8:
# #         #         return infer_topic_heuristic(segment_text, segment_id)

# #         #     return topic

# #         # except Exception:
# #         #     return infer_topic_heuristic(segment_text, segment_id)
# #             if self.provider in {"groq", "openrouter"}:
# #                 response = self.client.chat.completions.create(
# #                     model=self.model,
# #                     temperature=0,
# #                     messages=[
# #                         {"role": "system", "content": system_prompt},
# #                         {"role": "user", "content": user_prompt},
# #                     ],
# #                 )
# #                 topic = response.choices[0].message.content.strip()

# #             elif self.provider == "gemini":
# #                 prompt = f"""System instruction:
# #             {system_prompt}

# #             User request:
# #             {user_prompt}
# #             """
# #                 response = self.client.models.generate_content(
# #                     model=self.model,
# #                     contents=prompt,
# #                 )
# #                 topic = response.text.strip()

# #             else:
# #                 return infer_topic_heuristic(segment_text, segment_id)
# #         except Exception:
# #             return infer_topic_heuristic(segment_text, segment_id)

#     # def generate_topic(self, segment_text: str, segment_id: int) -> str:
#     #     """
#     #     Generate a short lecture-style topic title for one segment.
#     #     """
#     #     truncated_text = segment_text[:1800]

#     #     system_prompt = """
#     # You label lecture transcript segments with short topic titles.

#     # Return ONLY the topic title as plain text.
#     # Do not return JSON.
#     # Do not return quotes.
#     # Do not return explanations.

#     # Rules:
#     # - 2 to 6 words only
#     # - Use lecture-style wording
#     # - Focus on the main technical concept
#     # - Avoid filler words, numbers, and sentence fragments
#     # - Avoid generic titles like "Introduction" or "Example" unless truly appropriate
#     # """.strip()

#     #     user_prompt = f"""Transcript segment:
#     # {truncated_text}

#     # Write one short topic title only."""

#     #     try:
#     #         response = self.client.chat.completions.create(
#     #             model=self.model,
#     #             temperature=0,
#     #             messages=[
#     #                 {"role": "system", "content": system_prompt},
#     #                 {"role": "user", "content": user_prompt},
#     #             ],
#     #         )
#     #         topic = response.choices[0].message.content

#     #         if not isinstance(topic, str):
#     #             return infer_topic_heuristic(segment_text, segment_id)

#     #         topic = topic.strip()
#     #         topic = topic.replace('"', "").replace("'", "").strip()
#     #         topic = re.sub(r"\s+", " ", topic)

#     #         if not topic or len(topic.split()) > 8:
#     #             return infer_topic_heuristic(segment_text, segment_id)

#     #         return topic

#     #     except Exception as e:
#     #         print(f"Topic generation failed for segment {segment_id}: {e}")
#     #         return infer_topic_heuristic(segment_text, segment_id)

#     def generate_topic(self, segment_text: str, segment_id: int) -> str:
#         """
#         Generate a short lecture-style topic title for one segment.
#         """
#         truncated_text = segment_text[:1800]

#         system_prompt = """
#     You label lecture transcript segments with short topic titles.

#     Return ONLY the topic title as plain text.
#     Do not return JSON.
#     Do not return quotes.
#     Do not return explanations.

#     Rules:
#     - 2 to 6 words only
#     - Use lecture-style wording
#     - Focus on the main technical concept
#     - Avoid filler words, numbers, and sentence fragments
#     - Avoid generic titles like "Introduction" or "Example" unless truly appropriate
#     """.strip()

#         user_prompt = f"""Transcript segment:
#     {truncated_text}

#     Write one short topic title only."""

#         try:
#         #     response = self.client.chat.completions.create(
#         #         model=self.model,
#         #         temperature=0,
#         #         messages=[
#         #             {"role": "system", "content": system_prompt},
#         #             {"role": "user", "content": user_prompt},
#         #         ],
#         #     )

#         #     if not getattr(response, "choices", None):
#         #         return infer_topic_heuristic(segment_text, segment_id)

#         #     message = response.choices[0].message
#         #     content = getattr(message, "content", None)

#         #     if isinstance(content, str):
#         #         topic = content.strip()
#         #     elif isinstance(content, list):
#         #         parts = []
#         #         for item in content:
#         #             if isinstance(item, dict) and item.get("type") == "text":
#         #                 parts.append(item.get("text", ""))
#         #             else:
#         #                 text = getattr(item, "text", None)
#         #                 if text:
#         #                     parts.append(text)
#         #         topic = "".join(parts).strip()
#         #     else:
#         #         return infer_topic_heuristic(segment_text, segment_id)

#         #     topic = topic.replace('"', "").replace("'", "").strip()
#         #     topic = re.sub(r"\s+", " ", topic)

#         #     if not topic or len(topic.split()) > 8:
#         #         return infer_topic_heuristic(segment_text, segment_id)

#         #     return topic

#         # except Exception as e:
#         #     print(f"Topic generation failed for segment {segment_id}: {e}")
#         #     return infer_topic_heuristic(segment_text, segment_id)
#             topic = self.llm.generate(system_prompt, user_prompt)
#             topic = topic.replace('"', "").replace("'", "").strip()
#             topic = re.sub(r"\s+", " ", topic)

#             if not topic or len(topic.split()) > 8:
#                 return infer_topic_heuristic(segment_text, segment_id)

#             return topic

#         except Exception as e:
#             print(f"Topic generation failed for segment {segment_id}: {e}")
#             return infer_topic_heuristic(segment_text, segment_id)

class TopicLabeler:
    def __init__(self, model: Optional[str] = None):
        self.llm = CodexLLMClient(model=model)

    def generate_topic(self, segment_text: str, segment_id: int) -> str:
        """
        Generate a short lecture-style topic title for one segment.
        """
        truncated_text = segment_text[:1800]

        system_prompt = """
You label lecture transcript segments with short topic titles.

Return ONLY the topic title as plain text.
Do not return JSON.
Do not return quotes.
Do not return explanations.

Rules:
- 2 to 6 words only
- Use lecture-style wording
- Focus on the main technical concept
- Avoid filler words, numbers, and sentence fragments
- Avoid generic titles like "Introduction" or "Example" unless truly appropriate
""".strip()

        user_prompt = f"""Transcript segment:
{truncated_text}

Write one short topic title only."""

        try:
            topic = self.llm.generate(system_prompt, user_prompt)
            topic = topic.replace('"', "").replace("'", "").strip()
            topic = re.sub(r"\s+", " ", topic)

            if not topic or len(topic.split()) > 8:
                return infer_topic_heuristic(segment_text, segment_id)

            return topic

        except Exception as e:
            print(f"Topic generation failed for segment {segment_id}: {e}")
            return infer_topic_heuristic(segment_text, segment_id)

def segment_transcript(
    transcript_path: str,
    lecture_id: str = "lec4_transformers",
    min_segments: int = 10,
    max_segments: int = 20,
) -> List[Dict]:
    """
    Convert raw transcript text into a list of segment dictionaries.
    Heuristic:
    - Split by paragraphs if available
    - Otherwise split by sentences and group into chunks
    - Merge/split lightly to aim for 10-20 segments where possible
    """
    raw_text = read_transcript(transcript_path)
    blocks = split_paragraphs(raw_text)

    # If there are too few blocks, split larger blocks further
    refined_blocks: List[str] = []
    for block in blocks:
        words = block.split()
        # if len(words) > 220:
        if len(words) > 320:
            mid = len(words) // 2
            refined_blocks.append(" ".join(words[:mid]))
            refined_blocks.append(" ".join(words[mid:]))
        else:
            refined_blocks.append(block)

    # If too many blocks, merge nearby ones
    # if len(refined_blocks) > max_segments:
    #     merged = []
    #     temp = []
    #     for block in refined_blocks:
    #         temp.append(block)
    #         if len(temp) == 2:
    #             merged.append("\n".join(temp))
    #             temp = []
    #     if temp:
    #         merged.append("\n".join(temp))
    #     refined_blocks = merged
    if len(refined_blocks) > max_segments:
        group_size = 3 if len(refined_blocks) > 30 else 2
        merged = []
        temp = []
        for block in refined_blocks:
            temp.append(block)
            if len(temp) == group_size:
                merged.append("\n".join(temp))
                temp = []
        if temp:
            merged.append("\n".join(temp))
        refined_blocks = merged

    # segments = []
    # for idx, block in enumerate(refined_blocks, start=1):
    #     clean_text = re.sub(r"\s+", " ", block).strip()
    #     if not clean_text:
    #         continue

    #     segments.append({
    #         "lecture_id": lecture_id,
    #         "segment_id": idx,
    #         "text": clean_text,
    #         "topic": infer_topic(clean_text, idx),
    #     })

    topic_labeler = None
    try:
        topic_labeler = TopicLabeler()
    except Exception as e:
        print(f"TopicLabeler init failed: {e}")
        topic_labeler = None

    segments = []
    for idx, block in enumerate(refined_blocks, start=1):
        clean_text = re.sub(r"\s+", " ", block).strip()
        if not clean_text:
            continue

        if topic_labeler:
            topic = topic_labeler.generate_topic(clean_text, idx)
        else:
            topic = infer_topic_heuristic(clean_text, idx)

        print(f"Segment {idx} topic: {topic}")

        segments.append({
            "lecture_id": lecture_id,
            "segment_id": idx,
            "text": clean_text,
            "topic": topic,
        })

    return segments


def save_segments(segments: List[Dict], output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(segments, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment transcript into JSON segments.")
    parser.add_argument(
        "--input",
        default="lec4_transformers_transcript.txt",
        help="Path to raw transcript .txt file"
    )
    parser.add_argument(
        "--output",
        default="segments_lec4_transformers.json",
        help="Path to write segment JSON"
    )
    parser.add_argument(
        "--lecture-id",
        default="lec4_transformers",
        help="Lecture identifier"
    )
    args = parser.parse_args()

    segments = segment_transcript(args.input, lecture_id=args.lecture_id)
    save_segments(segments, args.output)

    print(f"Saved {len(segments)} segments to {args.output}")
