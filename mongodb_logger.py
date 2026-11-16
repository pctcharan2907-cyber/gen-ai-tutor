from datetime import datetime, timezone

from pymongo import MongoClient


class MongoDBLogger:
    def __init__(
        self,
        uri="mongodb://localhost:27017",
        database_name="GENAI",
        collection_name="LLM_LOGS",
    ):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Check that MongoDB is reachable.
        self.client.admin.command("ping")

        self.collection = self.client[database_name][collection_name]

    def log_request(self, provider, prompt, response, metrics=None):
        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "provider": provider,
            "prompt": prompt,
            "response": response,
            "metrics": metrics or {},
        }

        result = self.collection.insert_one(log_entry)
        return result.inserted_id

    def close(self):
        self.client.close()