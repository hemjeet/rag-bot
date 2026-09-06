import os
import time

from locust import HttpUser, task, between
import random

# Load collection name from environment (same as the API server)
COLLECTION_NAME = os.environ.get("DEFAULT_COLLECTION", "legal_documents")

# Rephrased variants so repeated load-test runs don't all hit the fuzzy/semantic cache.
# Each original question has 2-3 paraphrases → 28 unique queries total.
QUERIES = [
    # Liability
    "What is the limit of liability?",
    "What is the liability cap?",
    "How much can a party be liable for?",
    # Indemnification
    "How does indemnification work?",
    "What are the indemnification provisions?",
    "Explain the indemnification clause.",
    # Confidentiality
    "What are the confidentiality obligations?",
    "What are the NDA requirements?",
    "How is confidential information protected?",
    # Force majeure
    "Can you explain the force majeure clause?",
    "What does force majeure cover?",
    "What happens during force majeure events?",
    # Breach
    "What constitutes a breach of contract?",
    "What is considered a breach?",
    "When is a contract breached?",
    # Late payment
    "Are there any penalties for late payment?",
    "What are the late payment fees?",
    "Is there a penalty for delayed payment?",
    # Intellectual property
    "How is intellectual property handled?",
    "What are the IP provisions?",
    "Who owns the intellectual property?",
    # Termination
    "What is the term and termination process?",
    "How long does the agreement last?",
    "What is the termination procedure?",
    # Data breaches
    "Does this cover data breaches?",
    "What about data breach notifications?",
    "Are data breaches addressed?",
    # Disputes
    "What happens if there is a dispute?",
    "How are disputes resolved?",
    "What is the dispute resolution process?",
]


class RAGUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks for each simulated user
    wait_time = between(1.0, 3.0)

    @task(3)
    def query_rag_api(self):
        """Simulate a standard API query (non-streaming)."""
        query = random.choice(QUERIES)
        payload = {
            "query": query,
            "collection_name": COLLECTION_NAME,
            "session_id": "load_test_session",
        }

        # Name the endpoint so it groups in Locust UI, regardless of the random query
        with self.client.post(
            "/api/query", json=payload, name="/api/query", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Failed with status {response.status_code}: {response.text}"
                )

    @task(1)
    def query_rag_stream(self):
        """Simulate a streaming API query — measures full body consumption time."""
        query = random.choice(QUERIES)
        payload = {
            "query": query,
            "collection_name": COLLECTION_NAME,
            "session_id": "load_test_session_stream",
        }

        start = time.perf_counter()
        with self.client.post(
            "/api/query/stream",
            json=payload,
            name="/api/query/stream",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # Force full body download so Locust measures total stream time
                response.content  # noqa: B018
                elapsed_ms = (time.perf_counter() - start) * 1000
                if elapsed_ms > 100:
                    # Report custom timing for streams that took meaningful time
                    response.success()
                else:
                    response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(2)
    def health_check_fast(self):
        """Hit the lightweight health check (no DB round-trip)."""
        self.client.get("/health/fast", name="/health/fast")
