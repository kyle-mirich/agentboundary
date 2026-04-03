from __future__ import annotations

import json
import random

from openai import APIConnectionError, APITimeoutError, OpenAI

from .config import settings
from .models import ExampleInput, ExampleSource, Label

_SEED_PROMPT = """\
You are generating labeled training data for a text classifier.

Domain:
{description}

Generate exactly 90 examples as a JSON array only. Do not include markdown, comments, or explanation.

Label balance:
- 30 "in_scope": clearly inside the domain and unambiguous
- 30 "out_of_scope": clearly outside the domain and unambiguous
- 30 "ambiguous": plausible boundary cases that could reasonably be confused

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

_REQUEST_TIMEOUT_SECONDS = 20.0
_SEEDS_PER_LABEL = 30
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
    """Call GPT-5 to produce the configured labeled seed examples for the description.

    Retries once on parse failure. Raises RuntimeError if both attempts fail.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate seeds")

    client = OpenAI(
        api_key=settings.openai_api_key,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    def _attempt() -> list[ExampleInput]:
        response = client.chat.completions.create(
            model=settings.default_agent_model,
            messages=[
                {
                    "role": "user",
                    "content": _SEED_PROMPT.format(description=description),
                }
            ],
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        examples = [
            ExampleInput(
                text=item["text"],
                label=Label(item["label"]),
                source=ExampleSource.HUMAN_SEED,
            )
            for item in data
        ]
        labels_present = {e.label for e in examples}
        required = {Label.IN_SCOPE, Label.OUT_OF_SCOPE, Label.AMBIGUOUS}
        if not required.issubset(labels_present):
            raise ValueError(f"Missing labels: {required - labels_present}")
        return examples

    try:
        return _attempt()
    except (APITimeoutError, APIConnectionError):
        pass
    except Exception:
        try:
            return _attempt()
        except (APITimeoutError, APIConnectionError):
            pass
        except Exception as exc:
            raise RuntimeError("Seed generation failed") from exc
    raise RuntimeError("Seed generation failed")


def generate_lucky_description() -> str:
    """Return a single curated chatbot scope description for the homepage lucky path."""
    return _fallback_lucky_description()
