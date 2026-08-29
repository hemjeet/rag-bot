import logging

from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async
from src.config import settings

logger = logging.getLogger(__name__)

pool: AsyncConnectionPool | None = None


async def init_pool():
    global pool
    logger.info(
        "Initializing PostgreSQL connection pool (min_size=%s, max_size=%s).",
        settings.db_pool_min_size, settings.db_pool_max_size,
    )
    # The ``configure`` callback registers the pgvector type adapters on every
    # new connection, so Python lists can be bound to VECTOR columns.
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        open=False,
        configure=register_vector_async,
    )
    await pool.open()
    logger.info("PostgreSQL connection pool opened successfully.")


async def close_pool():
    global pool
    if pool is None:
        return
    try:
        # ``terminate`` immediately drops all connections instead of waiting
        # for in-use connections to finish — critical during uvicorn reload
        # so the old process does not hold Supabase connections hostage.
        await pool.terminate()
        logger.info("PostgreSQL connection pool terminated.")
    except Exception:
        logger.exception("Error while closing the connection pool.")
    finally:
        pool = None


def get_pool():
    if pool is None:
        raise RuntimeError("Pool not initialized. Call init_pool() first.")
    return pool
