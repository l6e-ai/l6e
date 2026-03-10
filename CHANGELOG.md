# Changelog

All notable changes to l6e are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-10

Initial release. Pipeline-scoped budget enforcement for AI agents.

### Core enforcement runtime

- **`PipelineContext`** — the central object for a single pipeline run. Wires together the constraint gate, run store, cost estimator, prompt classifier, local router, and run log. Constructed via the `pipeline()` factory.
- **`pipeline(run_id, policy, log_path=None, router=None)`** — context manager factory. Automatically writes a `RunSummary` to `.l6e/runs.jsonl` on exit, whether the run completes or raises. `log_path` and `router` can be overridden for testing.
- **`ctx.call(fn, model, messages, stage)`** — the primary integration point. Runs advise → execute → record in one call. On allow, calls `fn(model, messages)`. On reroute, calls `fn(local_model, messages)` and marks `rerouted=True` in the record. On halt, behaviour is determined by `policy.on_budget_exceeded`.
- **`ctx.advise(model, prompts, stage)`** — gate check only, without executing. Returns a `GateDecision`.
- **`ctx.record(...)`** — manually record a completed call for cases where `ctx.call()` isn't used.
- **`ctx.budget_status()`** — returns a `BudgetStatus` snapshot (zero tokens, pure arithmetic). Available at any point mid-run.
- **`ctx.run_summary()`** — returns the full `RunSummary` for the current run.

### Policy and types

- **`PipelinePolicy`** — frozen dataclass declaring a run's budget, mode, routing hints, stage overrides, and reroute threshold. Supports `from_toml(path)` for file-based config.
- **`BudgetMode`** — `HALT`, `REROUTE`, `WARN`. Controls behaviour when budget pressure hits the reroute threshold.
- **`OnBudgetExceeded`** — `RAISE`, `PARTIAL`, `EMPTY`, `FALLBACK`. Controls what `ctx.call()` returns (or raises) when the gate halts.
- **`StageRoutingHint`** — `LOCAL`, `CLOUD_STANDARD`, `CLOUD_FRONTIER`, `INHERIT`. Per-stage tier declarations in `stage_routing`.
- **`stage_overrides`** — per-stage hard `BudgetMode` that takes priority over all other routing logic. Use `"halt"` on a stage to ensure it never degrades. `WARN` is informational only in v0.1 and falls through to allow.
- **`BudgetStatus`** — zero-token economics snapshot: `spent_usd`, `remaining_usd`, `budget_usd`, `pct_used`, `budget_pressure` (`low` / `moderate` / `high` / `critical`), `reroutes`, `calls_made`.
- **`CallRecord`** — per-call telemetry: `model_requested`, `model_used`, `stage`, `prompt_complexity`, token counts, `cost_usd`, `rerouted`, `elapsed_ms`, `is_multi_turn`.
- **`RunSummary`** — full run record: `run_id`, `policy`, `total_cost`, `calls_made`, `reroutes`, `savings_usd`, all `CallRecord`s.

### Constraint gate (`gate.py`)

Pure decision logic with no side effects. Priority order for each call:

1. `stage_overrides` — explicit per-stage `BudgetMode`, wins over everything. `WARN` falls through to allow in v0.1.
2. `stage_routing` — tier hints: `LOCAL` triggers immediate reroute attempt, `CLOUD_STANDARD` and `CLOUD_FRONTIER` allow without pressure check, `INHERIT` falls through to budget pressure.
3. Over-budget guard — `store.spent() + estimated_cost > policy.budget` → halt.
4. Budget pressure threshold — `spent / budget >= reroute_threshold` → applies `budget_mode` (halt, reroute, or warn/allow).
5. Default allow.

On reroute: calls `router.best_local_model()`. If no local model is available, falls back to halt.

### Local router (`router.py`)

- Hardware-aware Ollama detection via `l6e_forge` (optional dependency).
- On first call to `best_local_model()`, imports `l6e_forge.models.auto`, calls `get_system_profile()`, then `suggest_models()`. Returns `None` immediately if `l6e_forge` is not installed, if the import fails, or if `profile.has_ollama` is false.
- Filters suggestions for `fits_local=True` and `provider="ollama"`, returns `"ollama/{provider_tag}"` for the first match.
- Result is cached after first call — hardware doesn't change mid-run.
- When `best_local_model()` returns `None`, the gate falls back to halt regardless of the reroute intent.

### Prompt complexity classifier (`_classify.py`)

- Classifies prompt complexity as `LOW`, `MEDIUM`, or `HIGH` using structural heuristics only — no model, no network call, ~2ms.
- Stage name short-circuit (checked before content): LOW stages: `formatting`, `format`, `extraction`, `extract`, `retrieval`, `retrieve`, `translation`, `translate`, `listing`. HIGH stages: `reasoning`, `final_reasoning`, `synthesis`, `synthesize`, `analysis`, `analyze`, `evaluation`, `evaluate`.
- Content score: low-complexity first word (`summarize`, `extract`, `format`, `translate`, `list`, `transcribe`, `convert`, `reformat`, `enumerate`) → -2; high placeholder/template density → -1; short prompt (<200 chars) → -1; long prompt (>2000 chars) → +1; reasoning keywords (`analyze`, `compare`, `evaluate`, `synthesize`, `critique`, `assess`, `contrast`, `argue`, `reason`) → up to +2; multi-step pattern (`Step 1:`, `firstly…then`, numbered list) → +2; more than 2 question marks → +1. Score ≥ 2 → HIGH, ≤ -2 → LOW, otherwise MEDIUM.
- Used by `L6eCallbackHandler` to auto-infer stage when no `l6e_stage:` tag is present (LOW → `"retrieval"`, MEDIUM → `"formatting"`, HIGH → `"reasoning"`).
- Populates `CallRecord.prompt_complexity` for future Pro profiling.

### Run log (`_log.py`)

- Appends every `RunSummary` to `.l6e/runs.jsonl` on `PipelineContext.__exit__` — always, even if the run raised an exception.
- Creates `.l6e/` directory if it doesn't exist.
- `read_recent(n)` reads back the last `n` summaries from the tail of the file.
- The JSONL file is the data source for the Pro profiling layer: every run since install is immediately available when a developer upgrades.

### Adapters

- **`L6eCallbackHandler`** (`l6e.adapters.langchain`) — `BaseCallbackHandler` for LangChain. Attach via `config={"callbacks": [handler]}`. In `on_llm_start`: extracts model from `invocation_params`, resolves stage from `l6e_stage:<name>` tag or auto-infers from prompt complexity, calls `ctx.advise()`, raises `BudgetExceeded` on halt. In `on_llm_end`: calls `ctx.record()`. **Reroute is advisory** — the decision is recorded but LangChain executes the call with the original model; the `model_used` in the record reflects the decision target, not a network-level substitution. Hard model substitution requires `ctx.call()` directly. Install extra: `pip install 'l6e[langchain]'`.
- **`L6eStepCallback`** (`l6e.adapters.crewai`) — plain callable for CrewAI's `step_callback`. No `crewai` package import required at runtime. Calls `ctx.advise()` between each agent step; raises `BudgetExceeded` on halt. Allow and reroute are both advisory in v0.1 — the step proceeds in both cases. Per-step model injection is a v0.2 target.
- **Universal adapter** (`l6e.adapters.universal`) — wraps any callable; used internally by `ctx.call()`.

### Exceptions

- **`BudgetExceeded(spent, budget, reason)`** — raised when `on_budget_exceeded=RAISE`. Attributes: `spent`, `budget`, `reason`.
- **`LatencySLAExceeded(elapsed_ms, sla_ms)`** — defined; not raised at runtime in v0.1. `latency_sla` on `PipelinePolicy` is accepted but not enforced until v0.2.

### Known limitations in v0.1

- **LangChain reroute is advisory.** `L6eCallbackHandler` records reroute decisions but cannot intercept the actual network call — LangChain's callback API does not expose model substitution at `on_llm_start`. Use `ctx.call()` directly for hard model substitution.
- **CrewAI reroute is advisory.** Allow and reroute both let the step proceed; enforcement is budget halt only.
- **Latency SLA not enforced.** `LatencySLAExceeded` is defined, `latency_sla` is accepted in `PipelinePolicy`, but the exception is never raised in v0.1.
- **Local router requires `l6e_forge`.** Without it, `best_local_model()` returns `None` and any reroute attempt falls back to halt.
- **`quality_safe_to_reroute` is always `None`** in the OSS runtime. Populated by the Pro Task Classification Tuner.

---

## Upcoming

### v0.2 (planned)

- Latency SLA enforcement (`LatencySLAExceeded` raised when wall time exceeds `policy.latency_sla`)
- CrewAI per-step model injection (hard reroute, not advisory)
- MCP server interface: `l6e_pipeline_start`, `l6e_checkpoint`, `l6e_pipeline_end`, `l6e_spend` tools for any MCP-compatible client (Claude Code, Cursor, raw agent loops)

### Pro layer (waitlist)

- Pipeline profiler — reads `.l6e/runs.jsonl` history to generate optimized `PipelinePolicy` configs
- Task Classification Tuner — calibrates stage routing thresholds to your actual workloads; populates `quality_safe_to_reroute`
- Budget Envelope Recommender — suggests per-pipeline budget limits from real cost distributions
- Spend analytics across all runs and engineers
- Anomaly detection and drift alerts

→ [Join the waitlist](mailto:hello@l6e.ai?subject=l6e%20Pro%20waitlist)
