from pymongo import MongoClient

from monitoring_config import (
    INPUT_COST_PER_1K_TOKENS,
    OUTPUT_COST_PER_1K_TOKENS,
)


NUMBER_OF_USERS = 1000
REQUESTS_PER_USER_PER_DAY = 2
DAYS_PER_MONTH = 30


client = MongoClient(
    "mongodb://localhost:27017",
    serverSelectionTimeoutMS=5000,
)

try:
    client.admin.command("ping")
    collection = client["GENAI"]["LLM_LOGS"]

    records = list(
        collection.find(
            {
                "provider": "ollama",
                "metrics.input_tokens": {"$exists": True},
                "metrics.output_tokens": {"$exists": True},
            }
        )
    )

    if not records:
        print("No Ollama usage records found.")
    else:
        total_input_tokens = sum(
            record["metrics"]["input_tokens"] for record in records
        )
        total_output_tokens = sum(
            record["metrics"]["output_tokens"] for record in records
        )
        total_requests = len(records)

        average_input_tokens = total_input_tokens / total_requests
        average_output_tokens = total_output_tokens / total_requests

        daily_requests = NUMBER_OF_USERS * REQUESTS_PER_USER_PER_DAY
        daily_input_tokens = daily_requests * average_input_tokens
        daily_output_tokens = daily_requests * average_output_tokens

        daily_input_cost = (
            daily_input_tokens / 1000
        ) * INPUT_COST_PER_1K_TOKENS
        daily_output_cost = (
            daily_output_tokens / 1000
        ) * OUTPUT_COST_PER_1K_TOKENS
        daily_total_cost = daily_input_cost + daily_output_cost

        monthly_requests = daily_requests * DAYS_PER_MONTH
        monthly_total_cost = daily_total_cost * DAYS_PER_MONTH

        print("\n========== OLLAMA BUDGET ESTIMATE ==========")
        print("MongoDB usage records:", total_requests)
        print("Average input tokens:", round(average_input_tokens, 2))
        print("Average output tokens:", round(average_output_tokens, 2))
        print("Estimated daily requests:", daily_requests)
        print("Estimated monthly requests:", monthly_requests)
        print("Estimated daily cost: Rs.", round(daily_total_cost, 2))
        print("Estimated monthly cost: Rs.", round(monthly_total_cost, 2))
        print("These costs use the demo rates in monitoring_config.py.")
        print("============================================")

finally:
    client.close()