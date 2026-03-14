# Changelog

All notable changes to l6e are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.6] — 2026-03-13

Initial release. Per-run budget enforcement and model routing for AI agent pipelines.

### Core runtime

- **`PipelineContext`** — central object for a single pipeline run. Constructed via the `pipeline()` factory.
- **`pipeline(policy, run_id=None, log_path=None, router=None)`** — context manager factory. `run_id` defaults to a generated UUID when omitted. Automatically writes a `RunSummary` to `.l6e/runs.jsonl` on exit, whether the run completes or raises.
- **`ctx.call(fn, model, messages, stage)`** — primary integration point. Runs advise → execute → record in one call. `fn` is called as `fn(model=..., messages=...)` with keyword arguments. On allow, calls `fn` with the original model. On reroute, calls `fn` with the local model and marks `rerouted=True` in the record. On halt, behaviour is determined by `policy.on_budget_exceeded`.
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

### Constraint gate

Pure decision logic with no side effects. Each call is evaluated in priority order: explicit stage overrides win first, then the hard budget ceiling, then stage routing tier hints, then the budget pressure threshold, then default allow.

On reroute: calls `router.best_local_model()`. If no local model is available, falls back to halt.

### Local router

- Hardware-aware Ollama detection via `l6e_forge` (optional dependency).
- On first call to `best_local_model()`, queries the local hardware profile and returns the best fitting Ollama model, or `None` if `l6e_forge` is not installed, Ollama is not running, or no suitable model is found.
- Result is cached after first call.
- When `best_local_model()` returns `None`, the gate falls back to halt regardless of the reroute intent.

### Prompt complexity classifier

- Classifies prompt complexity as `LOW`, `MEDIUM`, or `HIGH` using structural heuristics only — no model, no network call, ~2ms.
- Stage name takes priority: well-known stage names (e.g. `retrieval`, `formatting`, `reasoning`) short-circuit content analysis.
- Used by `L6eCallbackHandler` to auto-infer stage when no `l6e_stage:` tag is present.
- Populates `CallRecord.prompt_complexity`.

### Run log

- Appends every `RunSummary` to `.l6e/runs.jsonl` on `PipelineContext.__exit__` — always, even if the run raised an exception.
- Creates `.l6e/` directory if it doesn't exist.
- `read_recent(n)` reads back the last `n` summaries from the tail of the file.

### Adapters

- **`L6eCallbackHandler`** (`l6e.adapters.langchain`) — `BaseCallbackHandler` for LangChain. Attach via `config={"callbacks": [handler]}`. In `on_llm_start`: extracts model from `invocation_params`, resolves stage from `l6e_stage:<name>` tag or auto-infers from prompt complexity, calls `ctx.advise()`, raises `BudgetExceeded` on halt. In `on_llm_end`: calls `ctx.record()`. **Reroute is advisory** — the decision is recorded but LangChain executes the call with the original model; `model_used` in the record reflects the original model (not the reroute target), and `rerouted` is always `False`. Hard model substitution requires `ctx.call()` directly. Install extra: `pip install 'l6e[langchain]'`.
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
- **`quality_safe_to_reroute` is always `None`** in v0.1.

---

## Upcoming

### v0.2 (planned)

- Latency SLA enforcement (`LatencySLAExceeded` raised when wall time exceeds `policy.latency_sla`)
- CrewAI per-step model injection (hard reroute, not advisory)

→ [Join the Pro waitlist](mailto:hello@l6e.ai?subject=l6e%20Pro%20waitlist)
