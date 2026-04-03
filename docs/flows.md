# Application Flows

## 1. Quick Start

This is the fastest way to experience the project.

1. A user describes the classifier boundary on the landing page.
2. The frontend calls `POST /quick-start`.
3. The backend generates 216 labeled human-seed examples.
4. A project is created and the seeds are stored.
5. A run is queued with a three-round budget.
6. The frontend transitions into the live run experience and subscribes to the event stream.

This flow is optimized for a reviewer who wants to understand the product value immediately.

## 2. Manual Project Authoring

The project workspace supports a slower, more explicit setup path.

1. A user creates a project.
2. They review or add seed examples by label.
3. Once the minimum seed counts are satisfied, they start a run.
4. The API creates a run record and launches the Deep Agent orchestration loop.

This flow demonstrates that the app is not only a one-click experience; it also supports manual iteration.

## 3. Experiment Run

Each run follows a consistent sequence:

1. Plan the strategy.
2. Generate or refine candidate examples.
3. Prepare train/eval datasets.
4. Train the model.
5. Evaluate the round.
6. Generate holdout evaluation when available.
7. Review results and choose the best round.

The UI renders this through:

- a stage tracker
- a terminal-style event feed
- round metrics and summaries

## 4. Promotion and Classification

After a run completes:

1. The best round is selected.
2. The run can be promoted for the project.
3. The user enters free-form text in the playground.
4. The backend loads the promoted checkpoint and returns:
   - predicted label
   - confidence
   - per-label probabilities
   - explanation

This closes the loop from project creation to live inference.

## 5. Session Model

The app is intentionally unauthenticated for simplicity. Instead:

1. The browser creates a stable local session id.
2. The frontend sends that id on every API request.
3. The backend scopes projects and runs by session id.

This keeps the open-source workflow low-friction while still preventing accidental cross-user state mixing.
