import time

import ollama

from monitoring_config import (
    MODEL_NAME,
    CONTEXT_WINDOW_SIZE,
    RESERVED_OUTPUT_TOKENS,
    INPUT_COST_PER_1K_TOKENS,
    OUTPUT_COST_PER_1K_TOKENS,
)


def generate_response_with_metrics(prompt):
    # This is a rough estimate used only for the context pre-check.
    estimated_input_tokens = max(1, len(prompt) // 4)

    available_input_tokens = (
        CONTEXT_WINDOW_SIZE - RESERVED_OUTPUT_TOKENS
    )

    if estimated_input_tokens > available_input_tokens:
        raise ValueError(
            "Estimated prompt is too long for the configured context. "
            "Please shorten it."
        )

    start_time = time.perf_counter()

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "num_ctx": CONTEXT_WINDOW_SIZE,
            "num_predict": RESERVED_OUTPUT_TOKENS,
        },
    )

    application_latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    input_tokens = response.prompt_eval_count or 0
    output_tokens = response.eval_count or 0
    total_tokens = input_tokens + output_tokens

    context_used = total_tokens
    context_usage_percentage = (
        context_used / CONTEXT_WINDOW_SIZE
    ) * 100

    input_cost = (
        input_tokens / 1000
    ) * INPUT_COST_PER_1K_TOKENS

    output_cost = (
        output_tokens / 1000
    ) * OUTPUT_COST_PER_1K_TOKENS

    generation_duration_ns = response.eval_duration or 0
    tokens_per_second = 0

    if generation_duration_ns > 0:
        tokens_per_second = round(
            output_tokens / (generation_duration_ns / 1_000_000_000),
            2,
        )

    metrics = {
        "model": MODEL_NAME,
        "estimated_input_tokens": estimated_input_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "context_window_size": CONTEXT_WINDOW_SIZE,
        "context_used": context_used,
        "context_usage_percentage": round(
            context_usage_percentage, 2
        ),
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
        "application_latency_ms": round(application_latency_ms, 2),
        "ollama_total_duration_ms": round(
            (response.total_duration or 0) / 1_000_000, 2
        ),
        "prompt_eval_duration_ms": round(
            (response.prompt_eval_duration or 0) / 1_000_000, 2
        ),
        "generation_duration_ms": round(
            generation_duration_ns / 1_000_000, 2
        ),
        "tokens_per_second": tokens_per_second,
    }

    return {
        "response": response.response,
        "metrics": metrics,
    }