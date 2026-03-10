"""Response token extraction — structural inspection, no vendor isinstance checks."""
from __future__ import annotations


def extract_token_usage(response: object) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from any LLM response object.

    Inspection order:
    1. response.usage.prompt_tokens / completion_tokens  (OpenAI SDK, LiteLLM)
    2. response.usage.input_tokens / output_tokens       (Anthropic)
    2.5. response.llm_output["token_usage"]["..."]       (LangChain LLMResult)
    3. response["usage"]["prompt_tokens"]                (dict response)
    4. response["usage"]["input_tokens"]                 (dict Anthropic)
    5. tiktoken fallback: encode str(response), estimate split 80/20 prompt/completion
    6. (0, 0) if all else fails
    """
    # --- 1 & 2: attribute-based (OpenAI, LiteLLM, Anthropic SDK objects) ---
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion

        inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "output_tokens", None)
        if isinstance(inp, int) and isinstance(out, int):
            return inp, out

    # --- 2.5: LangChain LLMResult — response.llm_output["token_usage"]["prompt_tokens"] ---
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        tu = llm_output.get("token_usage")
        if tu is None:
            tu = llm_output.get("usage")
        if isinstance(tu, dict):
            p = tu.get("prompt_tokens")
            if p is None:
                p = tu.get("input_tokens")
            c = tu.get("completion_tokens")
            if c is None:
                c = tu.get("output_tokens")
            if isinstance(p, int) and isinstance(c, int):
                return p, c

    # --- 3 & 4: dict-based ---
    if isinstance(response, dict):
        usage_dict = response.get("usage")
        if isinstance(usage_dict, dict):
            prompt = usage_dict.get("prompt_tokens")
            completion = usage_dict.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt, completion

            inp = usage_dict.get("input_tokens")
            out = usage_dict.get("output_tokens")
            if isinstance(inp, int) and isinstance(out, int):
                return inp, out

    # --- 5: tiktoken fallback for plain strings ---
    if isinstance(response, str):
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            total = len(enc.encode(response))
            prompt_est = int(total * 0.8)
            completion_est = total - prompt_est
            return prompt_est, completion_est
        except Exception:
            return 0, 0

    # --- 6: unknown ---
    return 0, 0
