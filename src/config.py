from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # pydantic-settings reads environment variables case-insensitively by
    # default, so field names like ``database_url`` map to ``DATABASE_URL``.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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


settings = Settings()
