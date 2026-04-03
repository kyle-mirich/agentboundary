import json
from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError

from app import config as config_module
from app.models import ExampleSource, Label
from app.seed_generator import generate_seeds

pytestmark = pytest.mark.no_db


def _make_openai_response(data: list[dict]) -> MagicMock:
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


VALID_SEEDS = (
    [{"text": f"billing issue {i}", "label": "in_scope"} for i in range(30)]
    + [{"text": f"off topic {i}", "label": "out_of_scope"} for i in range(30)]
    + [{"text": f"mixed {i}", "label": "ambiguous"} for i in range(30)]
)


def test_generate_seeds_returns_90_examples():
    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("classify customer support tickets")

    assert len(results) == 90


def test_generate_seeds_label_distribution():
    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("classify customer support tickets")

    labels = {r.label for r in results}
    assert labels == {Label.IN_SCOPE, Label.OUT_OF_SCOPE, Label.AMBIGUOUS}


def test_generate_seeds_source_is_human_seed():
    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.return_value = (
            _make_openai_response(VALID_SEEDS)
        )
        results = generate_seeds("any domain")

    assert all(r.source == ExampleSource.HUMAN_SEED for r in results)


def test_generate_seeds_retries_on_bad_json():
    bad_response = _make_openai_response([])
    bad_response.choices[0].message.content = "not json"
    good_response = _make_openai_response(VALID_SEEDS)

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.side_effect = [
            bad_response,
            good_response,
        ]
        results = generate_seeds("any domain")

    assert len(results) == 90


def test_generate_seeds_raises_after_two_failures():
    bad = MagicMock()
    bad.choices[0].message.content = "not json"

    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.return_value = bad
        with pytest.raises(RuntimeError, match="Seed generation failed"):
            generate_seeds("any domain")


def test_generate_seeds_raises_on_timeout():
    with (
        patch.object(config_module.settings, "openai_api_key", "test-key"),
        patch("app.seed_generator.OpenAI") as mock_cls,
    ):
        mock_cls.return_value.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(RuntimeError, match="Seed generation failed"):
            generate_seeds("classify customer support tickets")


def test_generate_seeds_raises_without_openai_key():
    with patch.object(config_module.settings, "openai_api_key", ""):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required to generate seeds"):
            generate_seeds("classify customer support tickets")
