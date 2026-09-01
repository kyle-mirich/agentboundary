from __future__ import annotations

import json
import random

from openai import OpenAI

from .config import settings
from .models import ExampleInput, ExampleSource, Label

_SEED_PROMPT = """\
You are generating labeled training data for a text classifier.

Domain:
{description}

Generate exactly {total} examples as a JSON array only. Do not include markdown, comments, or explanation.

Label balance:
- {per_label} "in_scope": clearly inside the domain and unambiguous
- {per_label} "out_of_scope": clearly outside the domain and unambiguous
- {per_label} "ambiguous": plausible boundary cases that could reasonably be confused

Quality rules:
- Write realistic end-user messages, not synthetic label descriptions.
- Vary length, tone, wording, and intent.
- Keep each example self-contained and specific.
- Avoid duplicates, near-duplicates, and template-like phrasing.
- Make ambiguous examples genuinely borderline, not obviously mixed-label spam.
- Do not mention the labels inside the text.

Return this schema only:
[{{"text": "...", "label": "in_scope"}}, ...]\
"""

_REQUEST_TIMEOUT_SECONDS = 120.0
_LUCKY_PROMPTS = [
    "The chatbot should handle questions about Star Wars lore, characters, timelines, and canon debates.",
    "The chatbot should handle feedback on startup pitches, business models, and target customers.",
    "The chatbot should handle movie reviews, genre discussions, and spoiler-heavy film analysis.",
    "The chatbot should handle fitness questions about training, nutrition, recovery, and injury risk.",
    "The chatbot should handle personal budgeting, saving habits, and monthly spending plans.",
    "The chatbot should handle skincare routines, ingredients, and product layering.",
    "The chatbot should handle meal prep, grocery planning, and beginner home cooking.",
    "The chatbot should handle marathon training, pacing, hydration, and race prep.",
    "The chatbot should handle college essay feedback, school fit, and application strategy.",
    "The chatbot should handle interior design ideas for small apartments, furniture layout, and decor choices.",
    "The chatbot should handle coffee brewing methods, bean selection, and espresso technique.",
    "The chatbot should handle gardening questions about herbs, vegetables, soil, and seasonal planting.",
    "The chatbot should handle guitar gear, pedals, amps, and tone settings.",
    "The chatbot should handle chess openings, tactics, and endgame study plans.",
    "The chatbot should handle parenting questions about toddler routines, sleep schedules, and meal ideas.",
    "The chatbot should handle fashion advice about outfits, wardrobe basics, and seasonal styling.",
    "The chatbot should handle language learning plans, study routines, and pronunciation practice.",
    "The chatbot should handle board game rules, strategy, and player count recommendations.",
    "The chatbot should handle hiking trip prep, trail essentials, and gear packing.",
    "The chatbot should handle wedding planning timelines, vendor checklists, and guest logistics.",
    "The chatbot should handle nonfiction book recommendations, reading order, and note-taking methods.",
]
def _fallback_lucky_description() -> str:
    return random.choice(_LUCKY_PROMPTS)


def generate_seeds(description: str) -> list[ExampleInput]:
    """Generate the configured labeled seed examples for the description.

    Retries once on failure. Raises RuntimeError if both attempts fail.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate seeds")

    per_label = settings.seed_examples_per_label
    total = per_label * 3

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    def _attempt() -> list[ExampleInput]:
        response = client.chat.completions.create(
            # Seed generation is structured example generation, so it uses the
            # dedicated generation model rather than the orchestration model.
            model=settings.responses_generation_model,
            messages=[
                {
                    "role": "user",
                    "content": _SEED_PROMPT.format(
                        description=description,
                        per_label=per_label,
                        total=total,
                    ),
                }
            ],
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Seed response was not a JSON array")

        by_label: dict[Label, list[ExampleInput]] = {label: [] for label in Label}
        for item in data:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            label = Label(item["label"])
            # Truncate rather than trust the model: an oversized response would
            # otherwise insert an unbounded number of rows.
            if len(by_label[label]) >= per_label:
                continue
            by_label[label].append(
                ExampleInput(text=text, label=label, source=ExampleSource.HUMAN_SEED)
            )

        required = {Label.IN_SCOPE, Label.OUT_OF_SCOPE, Label.AMBIGUOUS}
        short = {
            label: per_label - len(by_label[label])
            for label in required
            if len(by_label[label]) < per_label
        }
        if short:
            raise ValueError(f"Insufficient examples per label: {short}")

        return by_label[Label.IN_SCOPE] + by_label[Label.OUT_OF_SCOPE] + by_label[Label.AMBIGUOUS]

    last_exc: Exception | None = None
    for _ in range(2):
        try:
            return _attempt()
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Seed generation failed: {last_exc}") from last_exc


def generate_lucky_description() -> str:
    """Return a single curated chatbot scope description for the homepage lucky path."""
    return _fallback_lucky_description()
