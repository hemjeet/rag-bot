import logging
from datetime import timedelta
from aiobreaker import CircuitBreaker
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async
from src.config import settings

logger = logging.getLogger(__name__)

# DB Circuit Breaker
db_breaker = CircuitBreaker(
    fail_max=settings.db_cb_failures,
    timeout_duration=timedelta(seconds=settings.db_cb_timeout),
)

pool: AsyncConnectionPool | None = None


async def init_pool():
    global pool
    logger.info(
        "Initializing PostgreSQL connection pool (min_size=%s, max_size=%s).",
        settings.db_pool_min_size,
        settings.db_pool_max_size,
    )
    # The ``configure`` callback registers the pgvector type adapters on every
    # new connection, so Python lists can be bound to VECTOR columns.
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        configure=register_vector_async,
        # Ping connections before handing them out to detect server-side closures
        check=AsyncConnectionPool.check_connection,
        # Discard connections idle for more than 5 minutes (Supabase / cloud PG
        # aggressively kills idle connections, often after ~60-300s)
        max_idle=300,
    )
    await pool.open()
    logger.info("PostgreSQL connection pool opened successfully.")


async def close_pool():
    global pool
    if pool is None:
        return
    try:
        # close(timeout=0) immediately drops all connections instead of waiting
        # for in-use connections to finish — critical during uvicorn reload
        # so the old process does not hold Supabase connections hostage.
        await pool.close(timeout=0)
        logger.info("PostgreSQL connection pool closed.")
    except Exception:
        logger.exception("Error while closing the connection pool.")
    finally:
        pool = None


def get_pool():
    if pool is None:
        raise RuntimeError("Pool not initialized. Call init_pool() first.")
    return pool


async def check_db_health() -> bool:
    """Run a simple SELECT 1 to verify the database is reachable."""
    if pool is None:
        return False
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        return True
    except Exception:
        logger.exception("[DB] health check failed")
        return False
