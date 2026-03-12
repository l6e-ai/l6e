# l6e

[![pytest](https://github.com/l6e-ai/l6e/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/l6e-ai/l6e/actions/workflows/pytest.yml)
[![coverage](https://raw.githubusercontent.com/l6e-ai/l6e/python-coverage-comment-action-data/badge.svg)](https://github.com/l6e-ai/l6e/actions/workflows/pytest.yml)
[![mypy](https://github.com/l6e-ai/l6e/actions/workflows/mypy.yml/badge.svg?branch=main)](https://github.com/l6e-ai/l6e/actions/workflows/mypy.yml)
[![ruff](https://github.com/l6e-ai/l6e/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/l6e-ai/l6e/actions/workflows/ruff.yml)

**Pipeline-scoped model choice enforcement for AI agents.**

Your agent pipeline doesn't know how much it costs. LiteLLM and Portkey enforce budgets per API key or per user — not per pipeline run. There's no way to say "this CrewAI crew gets $0.50 for this run, reroute to local models when it's running low."

l6e fixes that. It sits between your orchestrator and your router, enforces a budget across the whole run, and automatically downgrades to cheaper model tiers before you overspend.

Looking for the MCP integration for Claude Code, Cursor, or other use? Check out [l6e-mcp (Apache 2.0)](https://github.com/l6e-ai/l6e-mcp?branch=main).

---

## Install

```bash
pip install l6e
```

With LangChain support:

```bash
pip install 'l6e[langchain]'
```

---

## Quickstart: Universal wrapper

Works with any LLM client — LiteLLM, raw OpenAI SDK, anything callable.

```python
import l6e
import litellm

policy = l6e.PipelinePolicy(
    budget=0.50,
    budget_mode=l6e.BudgetMode.REROUTE,
)

with l6e.pipeline("my-run", policy=policy) as ctx:
    response = ctx.call(
        fn=litellm.completion,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summarize this document."}],
        stage="summarization",
    )

print(ctx.budget_status())
# BudgetStatus(spent_usd=0.00203, remaining_usd=0.49797, reroutes=0, budget_pressure='low', ...)
```

`ctx.call()` wraps advise → execute → record in one call. When budget pressure hits your reroute threshold, l6e substitutes the locally-available model automatically. Your code doesn't change.

---

## LangChain: zero pipeline code changes

Attach `L6eCallbackHandler` to any existing chain. Annotate stages with a tag.

```python
import l6e
from l6e.adapters.langchain import L6eCallbackHandler

policy = l6e.PipelinePolicy(
    budget=0.50,
    budget_mode=l6e.BudgetMode.REROUTE,
    stage_routing={
        "retrieval":  l6e.StageRoutingHint.LOCAL,           # reroute to Ollama
        "reasoning":  l6e.StageRoutingHint.CLOUD_FRONTIER,  # always gpt-4o
        "formatting": l6e.StageRoutingHint.CLOUD_STANDARD,  # gpt-4o-mini sufficient
    },
)

with l6e.pipeline("run-001", policy=policy) as ctx:
    handler = L6eCallbackHandler(ctx)

    summary_out = (
        summary_chain
        .with_config(tags=["l6e_stage:retrieval"])
        .invoke({"input": docs}, config={"callbacks": [handler]})
    )
    reasoning_out = (
        reasoning_chain
        .with_config(tags=["l6e_stage:reasoning"])
        .invoke({"input": summary_out}, config={"callbacks": [handler]})
    )
```

Before each LLM call, l6e checks the stage routing hint and budget pressure, and either allows, reroutes to a cheaper model tier, or halts with `BudgetExceeded`.

See [`examples/langchain_demo.ipynb`](examples/langchain_demo.ipynb) for a complete runnable demo showing per-stage routing decisions and cost savings.

---

## Agents can read budget state and adapt

`ctx.budget_status()` returns a snapshot of the current run's economics — `spent_usd`, `remaining_usd`, `budget_pressure`, `reroutes`, `calls_made`. Your agent can call it at any point mid-run and branch on the result:

```python
with l6e.pipeline("run", policy) as ctx:
    retrieval_result = ctx.call(fn=litellm.completion, model="gpt-4o",
                                messages=[...], stage="retrieval")

    status = ctx.budget_status()
    if status.budget_pressure in ("high", "critical"):
        # Skip the expensive next step, return what we have
        return f"Partial result: {retrieval_result}"

    return ctx.call(fn=litellm.completion, model="gpt-4o",
                    messages=[...], stage="reasoning")
```

`budget_status()` makes no LLM call — it's just arithmetic over the calls recorded so far. `budget_pressure` is one of `low`, `moderate`, `high`, or `critical`.

---

## Declare your policy in TOML

```toml
# l6e-policy.toml

[policy]
budget = 0.50
budget_mode = "reroute"
on_budget_exceeded = "partial"

[stage_routing]
retrieval     = "local"           # Qwen-32B on local hardware
summarization = "cloud_standard"  # gpt-4o-mini sufficient
reasoning     = "cloud_frontier"  # gpt-4o required
formatting    = "local"

[stage_overrides]
final_reasoning = "halt"          # never degrade, even under budget pressure
```

```python
from pathlib import Path
import l6e

policy = l6e.PipelinePolicy.from_toml(Path("l6e-policy.toml"))
with l6e.pipeline("run-001", policy=policy) as ctx:
    ...
```

---

## How it fits in your stack

```
Your stack today:
  LangChain / CrewAI / AutoGen   ← orchestrates agents
          ↓
  LiteLLM / OpenAI SDK           ← routes calls to models
          ↓
  GPT-4o / Claude / Ollama       ← executes inference

Where l6e sits:
  LangChain / CrewAI / AutoGen
    │       ↓
    │   [l6e — knows pipeline budget, stage, quality constraints]
    │       ↓  advises model tier
    │   LiteLLM / OpenAI SDK     ← executes the call l6e selected
    │       ↓
    │   GPT-4o-mini / Ollama / GPT-4o
    │
    └── ctx.budget_status()      ← zero-token economics snapshot
```

l6e does not replace LiteLLM or your existing router. It adds pipeline-run context — the budget envelope around the whole run, and the per-stage routing decisions within it.

---

## Local model rerouting

When `stage_routing` declares a stage as `"local"` and budget pressure triggers a reroute, l6e detects your hardware and picks the best available Ollama model automatically — no configuration required.

```python
# Stage declared as LOCAL + Ollama available:
# model_requested = "gpt-4o"
# model_used      = "ollama/qwen2.5:7b"   ← l6e substituted this
# rerouted        = True
# savings_usd     = 0.00333               ← what gpt-4o would have cost
```

On machines without Ollama, `LOCAL` stages fall back to the global `budget_mode` behaviour.

---

## Run log

Every `RunSummary` is appended to `.l6e/runs.jsonl` on context exit — automatically, no extra code required.

```
.l6e/runs.jsonl
{"run_id": "run-001", "total_cost": 0.0074, "reroutes": 1, "savings_usd": 0.0033, "records": [...]}
{"run_id": "run-002", "total_cost": 0.0081, "reroutes": 2, "savings_usd": 0.0041, "records": [...]}
```

Each record includes `model_requested`, `model_used`, `stage`, `prompt_complexity`, and token counts. The file grows with every run.

---

## License

Apache 2.0
