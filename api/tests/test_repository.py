from app.models import ExampleInput, ExampleSource, Label, ProjectCreate, Split
from app.repository import Repository


def test_seed_validation_and_split_assignment():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support",
            support_domain_description="Domain",
            allowed_topics=["billing"],
            disallowed_topics=["coding"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    examples = (
        [ExampleInput(text=f"support {index}", label=Label.IN_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"random {index}", label=Label.OUT_OF_SCOPE) for index in range(5)]
        + [ExampleInput(text=f"mixed {index}", label=Label.AMBIGUOUS) for index in range(3)]
    )
    repository.add_examples(project.id, examples)
    repository.ensure_seed_minimums(project.id)
    repository.assign_locked_eval_split(project.id)
    eval_examples = repository.get_examples_for_split(project.id, Split.EVAL)
    train_examples = repository.get_examples_for_split(project.id, Split.TRAIN)
    assert len(eval_examples) == 3
    assert len(train_examples) == 10


def test_holdout_examples_are_persisted_separately():
    repository = Repository()
    project = repository.create_project(
        ProjectCreate(
            name="Support Holdout",
            support_domain_description="Domain",
            allowed_topics=["billing"],
            disallowed_topics=["coding"],
            routing_notes="",
            sandbox_profile="isolated_fs",
        ),
        session_id="test-session",
    )
    repository.add_examples(
        project.id,
        [
            ExampleInput(
                text="Holdout billing example",
                label=Label.IN_SCOPE,
                source=ExampleSource.SYNTHETIC_HOLDOUT,
            )
        ],
        split=Split.HOLDOUT,
    )
    holdout_examples = repository.get_examples_for_split(project.id, Split.HOLDOUT)
    assert len(holdout_examples) == 1
    assert holdout_examples[0].source == ExampleSource.SYNTHETIC_HOLDOUT
