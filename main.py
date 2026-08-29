"""Application entry point.

Run the API with::

    python main.py

This wrapper exists primarily for Windows, where ``psycopg`` async requires a
``SelectorEventLoop``. Calling ``uvicorn app:app`` directly from the command
line creates the event loop *before* importing ``app``, so the policy set in
``app.py`` is too late. Setting the policy here (before ``uvicorn.run``) keeps
both direct invocation and the reloader working on Windows.
"""

import asyncio
import sys

import uvicorn

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
