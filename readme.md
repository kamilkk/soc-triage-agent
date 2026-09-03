# SOC Next-Best-Action Triage Agent

Use-case-specific triage for a Security Operations Center. The agent (1) generates
realistic incoming items with an LLM, (2) independently triages each one into a
priority level with a required action and a confidence, and (3) maintains a live
in-memory priority queue you can ask "what's next?" at any moment — even as more
urgent items arrive mid-stream.

Orchestration is **LangGraph**. The model is a **local GLM-5.2 served via vLLM**
(OpenAI-compatible endpoint), reached through a swappable factory. Everything
external is mocked, so the whole thing runs offline and deterministically.

---

## How to run

Requires [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync                                          # create venv + install from uv.lock
uv run python -m soc_triage.demo --mock          # full demo, offline & deterministic
uv run python -m soc_triage.eval --mock          # triage evaluation harness
uv run pytest                                    # test suite (13 tests)
```

The `--mock` flag uses a deterministic, network-free model — recommended for
review. To run against a real GLM-5.2:

```bash
# 1. serve the model (separate machine/GPU; the weights are large — see note below)
vllm serve zai-org/GLM-5.2-FP8 --port 8000

# 2. point the agent at it and drop --mock
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=zai-org/GLM-5.2-FP8
uv run python -m soc_triage.demo
```

> **Model footprint.** GLM-5.2 is a ~744B-parameter MoE model; even 2-bit quants
> are hundreds of GB. The code only ever talks to an OpenAI-compatible URL, so
> the endpoint can live anywhere (local GPU box or a shared/remote vLLM). For
> review, `--mock` reproduces the full behaviour with no model required.

---

## Running in Docker

The image (`Dockerfile`) contains only the agent — never the model. `docker-compose.yml`
exposes two independent profiles.

**Demo (offline, deterministic — no GPU, no network, no model):**

```bash
docker compose --profile demo up --build
```

This runs `python -m soc_triage.demo --mock` inside the container and prints the
same walkthrough as the local run. Tweak the args via the `demo` service's
`command` (e.g. `--strategy urgency`, `--n-per-level 3`).

**Real run (vLLM serving GLM-5.2-FP8 + the agent against it):**

```bash
cp .env.example .env        # set HUGGING_FACE_HUB_TOKEN and GPU/model knobs
docker compose --profile vllm up --build
```

This starts a `vllm/vllm-openai` container serving `LLM_MODEL` on an
OpenAI-compatible endpoint, waits for its `/health` check, then runs the `agent`
container pointed at `http://vllm:8000/v1`. Configure the model id, GPU count,
tensor-parallel size, and context length in `.env` (see `.env.example`).

> **Requirements for the `vllm` profile.** NVIDIA GPU(s) with enough combined
> memory for GLM-5.2-FP8 (hundreds of GB — see the footprint note above), the
> [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
> and a Hugging Face token. On a machine without adequate GPUs the `vllm`
> service will not boot; use the `demo` profile to see the full agent behaviour.

The demo prints: synthetic items → per-item triage decisions → the live queue
and "what's next?" → an explicit emergency-jumps-the-queue scenario → the
failure path → the dead-letter summary. Try `--strategy urgency` to see the
pluggable ordering in action.

---

## Project layout

```
soc_triage/
  schema.py     priority schema (Sev1–Sev4) + helpers
  models.py     Pydantic contracts: IncomingItem, TriageResult, LabeledItem
  llm.py        make_chat_model(): real vLLM client + deterministic MockChatModel
  generate.py   synthetic item generation (forced coverage + edge cases)
  triage.py     layered triage: structured-output → JSON fallback → heuristic
  queue.py      TriageQueue: heapq with a pluggable ordering strategy
  graph.py      LangGraph pipeline: validate → triage → route → enqueue
  demo.py       runnable CLI
  eval.py       evaluation harness
tests/          queue, triage routing, graph integration (13 tests)
```

---

## Key decisions and trade-offs

**LangGraph over plain LangChain chains.** The flow branches (valid vs malformed;
confident vs review) and carries long-lived state (the queue). LangGraph's
conditional edges model that branching explicitly, which is the point of showing
the flow. A linear chain would hide it.

**Two halves: the brain and the state.** The LangGraph pipeline is the *cognitive*
part — it turns one raw item into a `TriageResult` (or routes it aside). The
`TriageQueue` is the *operational* state, a long-lived object the orchestrator
owns and the graph pushes into via injected config (`configurable.queue`). This
separation is what makes "ask what's next at any moment, even as new things
arrive" clean: new items flow through the graph and land in the same queue, so
`peek()` always reflects current truth. Trade-off: the queue isn't part of the
graph's checkpointed state, so durable resume is future work (below).

**Local GLM-5.2 behind one factory.** `make_chat_model()` returns a LangChain
`ChatOpenAI` pointed at the vLLM endpoint, or a mock. The rest of the code is
provider-agnostic — swapping models, or adding a cheap/expensive cascade, is a
factory change.

**Structured output is hardened, not trusted.** A quantized local model is less
reliable at tool-call-based structured output than a hosted one, so triage is
layered (see Safeguards). The mock deliberately refuses tool calling, so the test
suite actually exercises the JSON-fallback path.

**`uv` for packaging.** One fast tool for the venv, resolution, lockfile, and
running. The committed `uv.lock` makes installs reproducible, and `uv run` means
a reviewer never has to activate a venv. It's an application (no build backend),
so pytest gets the package via `pythonpath = ["."]`.

---

## How synthetic items are generated (and why reliable inputs matter)

`generate.py` asks the LLM for items, but under three controls:

- **Forced coverage** — N items per severity level, so the queue and the eval
  actually exercise the full range and the re-ordering behaviour. Uncontrolled
  generation tends to collapse to one tone or one severity.
- **Deliberate edge injection** — explicitly requests ambiguous and off-topic
  items; the demo also hand-injects a malformed record. The safeguards are
  *exercised*, not hoped for.
- **No label leakage** — the generator is told the target severity (a weak label
  used only by eval) but is forbidden from naming it in the text. Otherwise triage
  degrades to string-matching and the evaluation becomes meaningless. A test
  asserts no severity labels leak into the generated text.

Why this matters for *this* workflow: the queue's behaviour and the entire triage
evaluation are only as trustworthy as the inputs. Skewed or unrealistic inputs
mean you can't demonstrate re-ordering (you need a spread of severities arriving
out of order), can't measure triage across the range, and get a falsely rosy
score from leaked labels. Controlled, diverse, label-clean generation is the
foundation the demo and the eval both stand on.

---

## How the queue prioritises, breaks ties, and re-orders

A `heapq` of `(sort_key, seq, item)` tuples. The `sort_key` comes from a
**pluggable strategy** the queue is constructed with; `seq` is a monotonic
counter that (a) breaks ties and (b) guarantees two items are never compared by
their payload.

- **v1 default — `fifo_key`:** `(priority_level,)`. Sev1 (level 1) sorts first;
  ties within a level fall through to `seq`, i.e. **FIFO**.
- **v2 drop-in — `urgency_key`:** `(priority_level, -urgency)`. Same cross-tier
  order, but higher intra-tier urgency surfaces first. `TriageQueue(key_fn=
  urgency_key)` switches the whole system with no change to the graph, the nodes,
  or the data model — the `urgency` field is already populated by triage. Try
  `--strategy urgency` in the demo.

**Re-ordering / emergency jumps the queue.** The heap invariant *is* the
re-ordering. With three Sev3s queued, a new Sev1's key `(1,)` sorts above every
`(3,)`, so the very next `peek()`/`pop()` returns it — no manual resort, no scan.
The demo shows this explicitly, and a test asserts it.

**Why FIFO for v1 (be ready to defend):** determinism and auditability — "we
worked them in arrival order within the tier" is explainable and reproducible,
and it keeps a noisy LLM score off the critical ordering path until that score
has been validated. The deliberate sequencing is: ship FIFO → validate the
model's `urgency` against the eval set → *then* flip the strategy. Switching is
config; trusting it is earned. Known trade-off: pure FIFO can starve Sev4 under
constant high-sev load — an aging strategy (`(level − wait_bonus, …)`) slots into
the same `key_fn` seam and is listed as future work.

---

## Safeguards

| Situation | Handling |
|---|---|
| Fits no category / ambiguous | Triage emits `category="UNKNOWN"` + low confidence → routed to the **review lane** at a conservative **Sev3**, flagged `needs_human_review`. Surfaced above routine noise without false Sev1 escalation. |
| Ambiguous between two levels | Prompt picks the more severe level and lowers confidence; if below the 0.6 threshold → review. |
| Low confidence (< 0.6) | Flagged and routed to review regardless of category. |
| Malformed item | Pydantic validation fails in the `validate` node → **dead-letter**, run continues. Never enters the queue, never crashes. |
| Failed / timed-out LLM call | tenacity retry → JSON-mode reprompt → deterministic **keyword heuristic** with `needs_human_review=True`. Triage always returns a routable result. |

So a flaky model degrades to *flagged-for-human*, never to a crash or a silently
wrong answer. Each safeguard has a test.

---

## How to know the triage is any good (evaluation)

`eval.py` scores triage's independent decision against the intended severity from
generation. The metrics emphasise the **asymmetric cost** in a SOC — missing a
real Sev1 is far worse than over-prioritising routine noise:

- exact-level accuracy and within-±1-level accuracy;
- **Sev1 recall**, tracked on its own (the costly miss);
- under- vs over-triage counts and a **cost-weighted error** (under-triage
  penalised 3×);
- % routed to review (too high → schema too narrow or model weak; near-zero with
  errors → overconfident);
- a confusion matrix across Sev1–Sev4.

Run against the mock, every score is perfect by construction — that validates the
*harness mechanics*, not a model. Real numbers come from running without `--mock`.
The intended labels are a weak baseline, not gold; see future work.

---

## What was faked / skipped, and what I'd do with more time

**Faked / skipped (intentionally, for the time-box):**
- The LLM is mocked for offline runs; only the mock path is exercised here (the
  real vLLM path is wired but needs the model served).
- Queue is in-memory only — no persistence/checkpointer.
- Single-process and synchronous; items are fed in a loop, not truly concurrently.
- Aging/anti-starvation is not implemented (the `key_fn` seam for it exists).
- Eval uses generation's weak labels; no hand-labeled gold set, no LLM-as-judge.

**With more time:**
- **Aging** `key_fn` to prevent Sev4 starvation.
- LangGraph **checkpointer** for durable queue state + resume.
- A **hand-labeled gold set** + an **LLM-as-judge** scoring action quality, not
  just the level.
- **Async/streaming** ingestion so items genuinely arrive concurrently.
- **Validate the model's `urgency`** against the eval set, then flip the queue to
  `urgency_key` — the seam is already in place; this is the validation that earns
  trusting it.
- A cheap/expensive **model cascade** (fast local triage, escalate low-confidence
  items to a stronger model) — trivial given the `make_chat_model()` factory.
