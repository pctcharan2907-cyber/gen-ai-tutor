import ollama


def generate_response_with_metrics(prompt, model_name="llama3.2"):
    result = ollama.generate(
        model=model_name,
        prompt=prompt,
    )

    input_tokens = result.prompt_eval_count or 0
    output_tokens = result.eval_count or 0
    generation_time_ns = result.eval_duration or 0

    tokens_per_second = 0
    if generation_time_ns > 0:
        tokens_per_second = round(
            output_tokens / (generation_time_ns / 1_000_000_000),
            2,
        )

    metrics = {
        "model": result.model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "total_duration_seconds": round(
            (result.total_duration or 0) / 1_000_000_000,
            3,
        ),
        "generation_duration_seconds": round(
            generation_time_ns / 1_000_000_000,
            3,
        ),
        "tokens_per_second": tokens_per_second,
    }

    return {
        "response": result.response,
        "metrics": metrics,
    }