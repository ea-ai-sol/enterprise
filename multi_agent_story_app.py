#!/usr/bin/env python3
"""Two-agent story application using OpenAI Responses API.

Agent 1: Story writer
Agent 2: Proofreader/editor
"""

import argparse
import os
import sys
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class StoryResult:
    draft: str
    polished: str


WRITER_SYSTEM_PROMPT = (
    "You are a creative fiction writer. Produce vivid, coherent stories with a clear "
    "beginning, middle, and end."
)

PROOFREADER_SYSTEM_PROMPT = (
    "You are an expert proofreader and editor. Correct grammar, spelling, punctuation, "
    "and clarity while preserving tone and intent. Return only the improved story."
)


def call_agent(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


def write_and_proofread(
    client: OpenAI, model: str, topic: str, style: str, target_words: int
) -> StoryResult:
    draft_prompt = (
        f"Write a {style} short story about: {topic}\n"
        f"Target length: about {target_words} words."
    )
    draft = call_agent(client, model, WRITER_SYSTEM_PROMPT, draft_prompt)

    proof_prompt = (
        "Proofread and improve the story below for grammar, flow, and readability.\n\n"
        f"{draft}"
    )
    polished = call_agent(client, model, PROOFREADER_SYSTEM_PROMPT, proof_prompt)
    return StoryResult(draft=draft, polished=polished)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-agent story app: writer agent + proofreader agent."
    )
    parser.add_argument("--topic", required=True, help="Story topic or premise.")
    parser.add_argument("--style", default="mysterious", help="Story style or tone.")
    parser.add_argument(
        "--words", type=int, default=300, help="Approximate story length in words."
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model to use for both agents (default: gpt-4.1-mini).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    client = OpenAI()
    result = write_and_proofread(
        client=client,
        model=args.model,
        topic=args.topic,
        style=args.style,
        target_words=args.words,
    )

    print("\n=== Writer Draft ===\n")
    print(result.draft)
    print("\n=== Proofread Version ===\n")
    print(result.polished)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
