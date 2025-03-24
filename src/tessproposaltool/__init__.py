from __future__ import absolute_import, division, print_function, unicode_literals

import asyncio
import logging  # noqa: E401
import os
import sys
import time
from threading import Event, Thread

from rich.console import Console
from rich.logging import RichHandler  # noqa: E402

PACKAGEDIR = os.path.dirname(os.path.abspath(__file__))

__version__ = "0.1.3"


def get_logger():
    """Configure and return a logger with RichHandler."""
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    # # Add RichHandler
    # rich_handler = RichHandler(show_time=False, show_level=False, show_path=False, rich_tracebacks=True)
    # rich_handler.setFormatter(
    #     logging.Formatter("%(message)s")
    # )

    # logger.addHandler(rich_handler)
    #   return logger
    return TPTLogger("TPT")


# Custom Logger with Rich
class TPTLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        console = Console()
        self.handler = RichHandler(
            show_time=False, show_level=False, show_path=False, console=console
        )
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.addHandler(self.handler)
        self.spinner_thread = None
        self.spinner_event = None

    def start_spinner(self, message="Searching..."):
        if self.spinner_thread is None:
            self.spinner_event = Event()
            self.spinner_thread = Thread(target=self._spinner, args=(message,))
            self.spinner_thread.start()

    def stop_spinner(self):
        if self.spinner_thread is not None:
            self.spinner_event.set()
            self.spinner_thread.join()
            self.spinner_thread = None
            self.spinner_event = None

    def _spinner(self, message):
        while not self.spinner_event.is_set():
            time.sleep(0.1)


logger = get_logger()
_nest_asyncio_applied = False


def _sync_call(func, *args, **kwargs):
    global _nest_asyncio_applied
    # Check if we're in a Jupyter notebook environment
    if "ipykernel" in sys.modules and not _nest_asyncio_applied:
        # We are in Jupyter, check for nest_asyncio
        try:
            import nest_asyncio

            nest_asyncio.apply()
            _nest_asyncio_applied = True  # Set the flag so we don't apply it again
        except ImportError:
            logger.warn(
                "nest_asyncio is required in a Jupyter environment. Please install with `!pip install nest_asyncio`."
            )
            return None
        # Run the async function with the current event loop
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    else:
        # We are not in Jupyter or nest_asyncio has already been applied, use asyncio.run()
        return asyncio.run(func(*args, **kwargs))


from .tpt import *  # noqa: E402, F401, F403
