import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads directly from os.environ (e.g. on Vercel), with optional fallback
    # to a local .env file if present during local development.
    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(...)
    openai_api_key: str = Field(...)
    deepseek_api_key: Optional[str] = Field(None)
    deepseek_base_url: str = Field("https://api.deepseek.com")

    embedding_model: str = Field("text-embedding-3-small")
    embedding_dimension: int = Field(1536)

    dense_top_k: int = Field(10)
    sparse_top_k: int = Field(10)
    rrf_k: int = Field(60)
    hybrid_top_n: int = Field(5)

    # Connection pool sizing. Keep this small when using Supabase's
    # session-mode pooler, which caps total connections (default 15).
    db_pool_min_size: int = Field(1)
    db_pool_max_size: int = Field(3)

    # When unset, the generator picks a sensible default based on which
    # provider key is configured (deepseek-chat vs gpt-4o-mini).
    llm_model: Optional[str] = Field(None)
    router_model: Optional[str] = Field(None)
    llm_temperature: float = Field(0.0)

    # Logging.
    log_level: str = Field("INFO")
    log_file: str = Field("agent.log")

    # --- Resiliency Settings ---
    # Tenacity (Retries)
    llm_max_retries: int = Field(3, description="Max retries for transient LLM errors")
    db_max_retries: int = Field(3, description="Max retries for transient DB errors")
    embed_max_retries: int = Field(3, description="Max retries for transient embedding errors")
    
    # Circuit Breaker (aiobreaker)
    llm_cb_failures: int = Field(5, description="Failures before LLM circuit trips")
    llm_cb_timeout: int = Field(30, description="Seconds to wait in Half-Open state")
    db_cb_failures: int = Field(5, description="Failures before DB circuit trips")
    db_cb_timeout: int = Field(15, description="Seconds to wait in Half-Open state")
    embed_cb_failures: int = Field(5, description="Failures before embedding circuit trips")
    embed_cb_timeout: int = Field(15, description="Seconds to wait in Half-Open state")


settings = Settings()
