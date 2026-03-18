import argparse
from datetime import datetime
from pathlib import Path

from segment_transcript import segment_transcript, save_segments
from content_extraction import (
    CodexLLMClient,
    extract_key_concepts,
    save_concepts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transcript segmentation and concept extraction pipeline.")
    parser.add_argument(
        "--input",
        default="lec4_transformers_transcript.txt",
        help="Path to raw transcript text file"
    )
    parser.add_argument(
        "--lecture-id",
        default="lec4_transformers",
        help="Lecture identifier"
    )
    parser.add_argument(
        "--segments-output",
        default="segments_lec4_transformers.json",
        help="Where to save intermediate segmented transcript JSON"
    )
    parser.add_argument(
        "--concepts-output",
        default="concepts_lec4_transformers.json",
        help="Where to save final concepts JSON"
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Optional delay between LLM calls"
    )
    args = parser.parse_args()

    Path("content_extraction").mkdir(parents=True, exist_ok=True)

    def backup_existing(path_str: str) -> None:
        path = Path(path_str)
        if not path.exists():
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
        path.rename(backup_path)

    # Step 1: segment transcript
    backup_existing(args.segments_output)
    segments = segment_transcript(
        transcript_path=args.input,
        lecture_id=args.lecture_id,
    )
    save_segments(segments, args.segments_output)

    # Step 2: concept extraction
    backup_existing(args.concepts_output)
    client = CodexLLMClient()
    concepts = extract_key_concepts(
        segments=segments,
        client=client,
        sleep_between_calls=args.sleep_between_calls,
    )
    save_concepts(concepts, args.concepts_output)

    total_concepts = sum(len(entry["concepts"]) for entry in concepts)

    print(f"Segments created: {len(segments)}")
    print(f"Total concepts extracted: {total_concepts}")
    print(f"Saved segments to: {args.segments_output}")
    print(f"Saved concepts to: {args.concepts_output}")


if __name__ == "__main__":
    main()