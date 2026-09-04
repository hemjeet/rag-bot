from locust import HttpUser, task, between
import random

# A diverse set of queries to test both cache hits and cache misses
QUERIES = [
    "What is the limit of liability?",
    "How does indemnification work?",
    "What are the confidentiality obligations?",
    "Can you explain the force majeure clause?",
    "What constitutes a breach of contract?",
    "Are there any penalties for late payment?",
    "How is intellectual property handled?",
    "What is the term and termination process?",
    "Does this cover data breaches?",
    "What happens if there is a dispute?"
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
            "collection_name": "legal_documents",
            "session_id": "load_test_session"
        }
        
        # Name the endpoint so it groups in Locust UI, regardless of the random query
        with self.client.post("/api/query", json=payload, name="/api/query", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text}")

    @task(1)
    def query_rag_stream(self):
        """Simulate a streaming API query."""
        query = random.choice(QUERIES)
        payload = {
            "query": query,
            "collection_name": "legal_documents",
            "session_id": "load_test_session_stream"
        }
        
        # For streaming, we need to read the chunks to measure total time
        with self.client.post("/api/query/stream", json=payload, stream=True, name="/api/query/stream", catch_response=True) as response:
            if response.status_code == 200:
                # Read the stream
                for line in response.iter_lines():
                    pass # Just consume it to measure full latency
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(2)
    def health_check(self):
        """Hit the health check endpoint to monitor system load capability."""
        self.client.get("/health", name="/health")
