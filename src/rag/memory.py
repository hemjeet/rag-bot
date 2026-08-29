import asyncio
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self._sessions: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=max_history * 2))
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        async with self._lock:
            history = list(self._sessions[session_id])
        logger.debug("Retrieved %d history messages for session '%s'.", len(history), session_id)
        return history

    async def add_message(self, session_id: str, role: str, content: str):
        async with self._lock:
            self._sessions[session_id].append({"role": role, "content": content})
        logger.debug("Added %s message to session '%s'.", role, session_id)

    async def clear_session(self, session_id: str):
        async with self._lock:
            self._sessions.pop(session_id, None)
        logger.debug("Cleared session '%s'.", session_id)


            