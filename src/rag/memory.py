import asyncio
import logging
from collections import OrderedDict
from typing import Dict, List

logger = logging.getLogger(__name__)

# Default max number of active sessions to prevent memory leaks
_MAX_SESSIONS = 1000


class MemoryManager:
    def __init__(self, max_history: int = 5, max_sessions: int = _MAX_SESSIONS):
        self.max_history = max_history
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        async with self._lock:
            history = self._sessions.get(session_id, [])
        return history

    async def add_message(self, session_id: str, role: str, content: str):
        async with self._lock:
            if session_id not in self._sessions:
                # Evict oldest session if at capacity
                if len(self._sessions) >= self.max_sessions:
                    evicted_id, _ = self._sessions.popitem(last=False)
                    logger.info(
                        "[MEMORY] evicted session '%s' (max_sessions=%d)",
                        evicted_id,
                        self.max_sessions,
                    )
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": role, "content": content})
            # Trim to max_history turns (each turn = user + assistant)
            max_messages = self.max_history * 2
            if len(self._sessions[session_id]) > max_messages:
                self._sessions[session_id] = self._sessions[session_id][-max_messages:]
            # Move to end (most recently used)
            self._sessions.move_to_end(session_id)

    async def clear_session(self, session_id: str):
        async with self._lock:
            self._sessions.pop(session_id, None)
        logger.debug("Cleared session '%s'.", session_id)

    async def session_count(self) -> int:
        async with self._lock:
            return len(self._sessions)
