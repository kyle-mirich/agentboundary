from __future__ import annotations

import json
import random
import re

from openai import APIConnectionError, APITimeoutError, OpenAI

from .config import settings
from .models import ExampleInput, ExampleSource, Label

_SEED_PROMPT = """\
You are generating training data for a text classifier.

Domain: {description}

Generate exactly 90 labeled examples in JSON format:
- 30 labeled "in_scope": messages clearly within the domain
- 30 labeled "out_of_scope": messages clearly outside the domain
- 30 labeled "ambiguous": messages that mix in-scope and out-of-scope intent

Return a JSON array only, no explanation:
[{{"text": "...", "label": "in_scope"}}, ...]

Make examples realistic, varied in length, and challenging enough to require a real classifier.\
"""

_REQUEST_TIMEOUT_SECONDS = 20.0
_SEEDS_PER_LABEL = 30
_FALLBACK_TOPIC_LIMIT = 4
_LUCKY_PROMPTS = [
    "The chatbot should only handle questions about Star Wars lore, characters, timelines, and canon debates. It should not handle coding help, travel planning, or general trivia.",
    "The chatbot should only handle feedback on startup pitches, business models, and target customers. It should not handle debugging code, entertainment picks, or personal finance advice.",
    "The chatbot should only handle movie reviews, genre discussions, and spoiler-heavy film analysis. It should not handle billing issues, workout planning, or programming questions.",
    "The chatbot should only handle fitness questions about training, nutrition, recovery, and injury risk. It should not handle legal advice, software troubleshooting, or vacation planning.",
    "The chatbot should only handle questions about personal budgeting, saving habits, and monthly spending plans. It should not handle relationship advice, coding help, or travel itineraries.",
    "The chatbot should only handle skincare routines, ingredients, and product layering. It should not handle medical diagnosis, tax questions, or laptop troubleshooting.",
    "The chatbot should only handle meal prep, grocery planning, and beginner home cooking. It should not handle stock picks, fantasy sports, or legal forms.",
    "The chatbot should only handle marathon training, pacing, hydration, and race prep. It should not handle visa questions, programming help, or movie recommendations.",
    "The chatbot should only handle college essay feedback, school fit, and application strategy. It should not handle therapy, coding bugs, or restaurant recommendations.",
    "The chatbot should only handle interior design ideas for small apartments, furniture layout, and decor choices. It should not handle contract law, debugging, or car repair advice.",
    "The chatbot should only handle coffee brewing methods, bean selection, and espresso technique. It should not handle travel visas, resume reviews, or fantasy novels.",
    "The chatbot should only handle gardening questions about herbs, vegetables, soil, and seasonal planting. It should not handle coding interviews, credit disputes, or pet training.",
    "The chatbot should only handle guitar gear, pedals, amps, and tone settings. It should not handle plumbing repairs, investment advice, or airline bookings.",
    "The chatbot should only handle chess openings, tactics, and endgame study plans. It should not handle medication guidance, app development, or celebrity gossip.",
    "The chatbot should only handle parenting questions about toddler routines, sleep schedules, and meal ideas. It should not handle divorce law, Python code, or hotel planning.",
    "The chatbot should only handle fashion advice about outfits, wardrobe basics, and seasonal styling. It should not handle crypto trading, database tuning, or immigration paperwork.",
    "The chatbot should only handle language learning plans, study routines, and pronunciation practice. It should not handle product returns, legal contracts, or software setup.",
    "The chatbot should only handle board game rules, strategy, and player count recommendations. It should not handle tax filing, backend debugging, or workout injuries.",
    "The chatbot should only handle hiking trip prep, trail essentials, and gear packing. It should not handle coding tasks, airline refunds, or skincare routines.",
    "The chatbot should only handle wedding planning timelines, vendor checklists, and guest logistics. It should not handle legal disputes, machine learning code, or fantasy football picks.",
    "The chatbot should only handle nonfiction book recommendations, reading order, and note-taking methods. It should not handle tax optimization, coding errors, or recipe substitutions.",
]
_DEFAULT_TOPICS = ["billing", "refunds", "account access", "login issues"]
_OUT_OF_SCOPE_TOPICS = [
    "Python debugging",
    "travel planning",
    "creative writing",
    "general trivia",
]
_AMBIGUOUS_EXTRAS = [
    "also help me write code",
    "and recommend a vacation spot",
    "plus answer a trivia question",
    "and draft a marketing slogan",
]


def _fallback_lucky_description() -> str:
    return random.choice(_LUCKY_PROMPTS)


def _extract_topics(description: str) -> list[str]:
    fragments = re.split(r"[,\n]| and | or ", description.lower())
    cleaned: list[str] = []
    for fragment in fragments:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", fragment).strip()
        if len(normalized) < 4:
            continue
        if normalized in cleaned:
            continue
        cleaned.append(normalized)
    return (cleaned[:_FALLBACK_TOPIC_LIMIT] or _DEFAULT_TOPICS).copy()


def _fallback_examples(description: str) -> list[ExampleInput]:
    topics = _extract_topics(description)
    in_scope = [
        ExampleInput(
            text=f"I need help with {topics[i % len(topics)]} for request #{i + 1}.",
            label=Label.IN_SCOPE,
            source=ExampleSource.HUMAN_SEED,
        )
        for i in range(_SEEDS_PER_LABEL)
    ]
    out_of_scope = [
        ExampleInput(
            text=f"Can you help with {_OUT_OF_SCOPE_TOPICS[i % len(_OUT_OF_SCOPE_TOPICS)]} instead of support issue #{i + 1}?",
            label=Label.OUT_OF_SCOPE,
            source=ExampleSource.HUMAN_SEED,
        )
        for i in range(_SEEDS_PER_LABEL)
    ]
    ambiguous = [
        ExampleInput(
            text=(
                f"I have a {topics[i % len(topics)]} problem, "
                f"{_AMBIGUOUS_EXTRAS[i % len(_AMBIGUOUS_EXTRAS)]} for request #{i + 1}."
            ),
            label=Label.AMBIGUOUS,
            source=ExampleSource.HUMAN_SEED,
        )
        for i in range(_SEEDS_PER_LABEL)
    ]
    return in_scope + out_of_scope + ambiguous


def generate_seeds(description: str) -> list[ExampleInput]:
    """Call GPT-5 to produce the configured labeled seed examples for the description.

    Retries once on parse failure. Raises RuntimeError if both attempts fail.
    """
    if not settings.openai_api_key:
        return _fallback_examples(description)

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
        return _fallback_examples(description)
    except Exception:
        try:
            return _attempt()
        except (APITimeoutError, APIConnectionError):
            return _fallback_examples(description)
        except Exception as exc:
            raise RuntimeError("Seed generation failed") from exc


def generate_lucky_description() -> str:
    """Return a single curated chatbot scope description for the homepage lucky path."""
    return _fallback_lucky_description()
