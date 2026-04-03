# Interview Demo UX — Design Spec
**Date:** 2026-03-26
**Status:** Approved

## Overview

Replace the current dashboard-style UI with a single-flow, Apple-minimal experience purpose-built for live interview demos. A visitor types one sentence describing their classifier, and the app handles everything — seed generation, training, and live testing — without requiring any manual configuration.

---

## User Flow (3 Screens, 1 Page)

All transitions happen in-place via CSS crossfades. No page navigation occurs until the user explicitly visits history.

### Screen 1 — The Prompt

Pure white, vertically centered. Shown on initial load.

**Elements (top to bottom, centered):**
- Small uppercase label: `CLASSIFIER STUDIO`
- Large heading: `What kind of classifier do you want to build?`
- Textarea (single field, autofocused, ~3 rows)
- Button: `Build it →`

**Behavior:**
- On submit: validate textarea is non-empty, call `POST /quick-start`, then transition to Screen 2
- Button shows a subtle spinner while the request is in flight (project creation is fast, <1s)
- No other UI on this page — no nav, no sidebar, no stats

---

### Screen 2 — Training Progress

Crossfades in after `POST /quick-start` responds. Connects to the SSE stream immediately.

**Elements:**
- Small label at top: the user's description, truncated to ~60 chars, greyed out
- 4-item checklist (updates in real time as SSE events arrive):

  | # | Label | Done when event received |
  |---|-------|--------------------------|
  | 1 | Generating seeds with GPT-5 | `seeds_ready` |
  | 2 | Training Round 1 | `round_complete` (index 1) |
  | 3 | Training Round 2 | `round_complete` (index 2) |
  | 4 | Evaluating & finishing | `run_reviewed` / `run_done` |

- Active item shows a spinning indicator; completed items show a filled checkmark
- Live log box below the checklist: scrolling monospace feed of `event.message` strings from the SSE stream, auto-scrolls to bottom
- If `max_rounds > 2`, extra round items are generated dynamically from the project config

**Behavior:**
- On `run_done` SSE event: wait 800ms (let user see the final checkmark), then crossfade to Screen 3
- On run `status === "failed"`: show an inline error message with a "Try again" button that resets to Screen 1

---

### Screen 3 — Test Your Classifier

Full-screen crossfade. This is the payoff.

**Elements (centered vertically):**
- Heading: `Your classifier is live.`
- Subline: `F1 {score} · {n} examples · {duration}`
- Large input field: placeholder `Type anything to test it…`
- On Enter → result appears below input:
  - Label badge (`IN SCOPE` / `OUT OF SCOPE` / `AMBIGUOUS`) in bold
  - Confidence percentage: `96.2% confident`
  - User can keep typing and pressing Enter; previous results stack below as a small feed (newest on top, max 5 visible)
- Bottom-right corner: subtle link `View training details ↗`

**View Training Details Drawer:**
- Slides up from the bottom of the screen (not a full modal — sits in lower ~50% of viewport)
- Contents:
  - F1 score per round (table: Round | Train F1 | Holdout F1)
  - Total examples generated
  - Full scrollable event log (same messages as Screen 2 log)
- Dismiss by clicking backdrop or pressing Escape

---

## Backend Changes

### New Endpoint: `POST /quick-start`

**Purpose:** Single call that does everything — create project, generate seeds with GPT-5, add examples, start run.

**Request body:**
```json
{
  "description": "Classify customer support tickets for a SaaS billing product"
}
```

**Response** (returns immediately after run is queued, ~1–2s):
```json
{
  "project_id": "uuid",
  "run_id": "uuid"
}
```

**Server-side steps (synchronous before returning):**
1. Call GPT-5 with a prompt asking it to generate ~90 seed examples from the description:
   - 30 in-scope examples
   - 30 out-of-scope examples
   - 30 ambiguous examples
2. Parse GPT-5 response into `[{ text, label }]` format
3. `POST /projects` — create project using the description as `support_domain_description`, sane defaults for all other fields (model: `gpt-5.4-mini`, max_rounds: 3, target_f1: 0.9)
4. `POST /projects/{id}/examples` — add the 90 generated seeds
5. `POST /projects/{id}/runs` — start the training run
6. Return `{ project_id, run_id }`

**GPT-5 seed generation prompt (template):**
```
You are generating training data for a text classifier.

Domain: {description}

Generate exactly 90 labeled examples in JSON format:
- 30 labeled "in_scope": messages clearly within the domain
- 30 labeled "out_of_scope": messages clearly outside the domain
- 30 labeled "ambiguous": messages that mix in-scope and out-of-scope intent

Return a JSON array only, no explanation:
[{"text": "...", "label": "in_scope"}, ...]

Make examples realistic, varied in length, and challenging enough to require a real classifier.
```

**Error handling:**
- If GPT-5 returns malformed JSON: retry once, then return 500 with `"detail": "Seed generation failed"`
- Validate that all 3 labels are represented before accepting

---

## Frontend Changes

### Pages / Files

| File | Change |
|------|--------|
| `web/app/page.tsx` | Full rewrite — new 3-screen single-flow component |
| `web/lib/api.ts` | Add `quickStart(description)` → `POST /quick-start` |
| `web/app/globals.css` | Add transition, drawer, and badge styles; keep existing design tokens |
| `web/app/project/[id]/page.tsx` | No change — still accessible directly via URL for debugging |

### State Machine (in `page.tsx`)

```
"prompt" → "training" → "testing"
```

- `phase: "prompt" | "training" | "testing"`
- Transitions driven by: form submit → SSE `run_done` event
- All phase state lives in a single top-level component; no router pushes

### CSS Transitions

- Screen-to-screen: `opacity` fade over 400ms (`ease-in-out`)
- Checklist items: check icon fades in, spinner fades out (`200ms`)
- Drawer: `transform: translateY(100%)` → `translateY(0)` over `300ms`
- Result feed items: fade + slide in from below (`150ms`)

---

## What Stays the Same

- `/project/[id]` detail page — unchanged, accessible for debugging
- All existing API endpoints — unchanged
- SSE streaming infrastructure — reused as-is
- Supabase / PostgreSQL backend — unchanged
- Session ID system — unchanged (quick-start still sends `X-Session-Id`)

---

## Out of Scope

- Multi-project management UI (dashboard removed from main flow)
- Manual seed entry
- Model / round configuration (uses hardcoded sane defaults)
- User accounts / auth

---

## Success Criteria

1. An interviewer can go from blank page to a live working classifier in under 5 minutes, typing only one sentence
2. The UI contains zero forms, zero configuration, zero jargon on the happy path
3. Every state transition (prompt → training → test) feels intentional and smooth
4. The "View training details" drawer satisfies any technical questions about what the agent actually did
