from pymongo import DESCENDING, MongoClient


def main():
    client = MongoClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=5000,
    )

    try:
        client.admin.command("ping")

        collection = client["GENAI"]["LLM_LOGS"]
        logs = collection.find().sort("timestamp", DESCENDING).limit(10)

        found_logs = False

        for log in logs:
            found_logs = True

            print("\n--- Tutor Request ---")
            print("Timestamp:", log.get("timestamp"))
            print("Provider:", log.get("provider"))
            print("Prompt:", log.get("prompt"))
            print("Response:", log.get("response"))

            metrics = log.get("metrics", {})
            if metrics:
                print("Metrics:")
                for name, value in metrics.items():
                    print(f"  {name}: {value}")

        if not found_logs:
            print("No logs found in GENAI.LLM_LOGS.")

    finally:
        client.close()


if __name__ == "__main__":
    main()