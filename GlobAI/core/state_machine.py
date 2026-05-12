"""
core/state_machine.py
---------------------
Strict query-cycle state machine.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class State(Enum):
    INIT = auto()
    LOAD = auto()
    RETRIEVE = auto()
    FILTER = auto()
    THINK = auto()
    ANSWER = auto()
    CLEANUP = auto()
    EXIT = auto()


class StateMachineError(RuntimeError):
    pass


class QueryStateMachine:
    SEQUENCE = [
        State.INIT,
        State.LOAD,
        State.RETRIEVE,
        State.FILTER,
        State.THINK,
        State.ANSWER,
        State.CLEANUP,
        State.EXIT,
    ]

    def __init__(self, handlers: Dict[State, Callable[[Dict[str, Any]], Dict[str, Any]]]):
        missing = [state.name for state in self.SEQUENCE if state not in handlers]
        if missing:
            raise StateMachineError(f"Missing handler(s): {', '.join(missing)}")
        self._handlers = handlers

    def run(self, query: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "query": query,
            "chunks": [],
            "answer": "",
            "error": None,
            "start_time": time.time(),
        }

        current_state_idx = 0
        cleanup_idx = self.SEQUENCE.index(State.CLEANUP)

        while current_state_idx < len(self.SEQUENCE):
            state = self.SEQUENCE[current_state_idx]
            logger.info("[SM] -> %s", state.name)
            try:
                next_context = self._handlers[state](context)
                if not isinstance(next_context, dict):
                    raise StateMachineError(f"{state.name} handler did not return context.")
                context = next_context
            except Exception as exc:
                logger.exception("[SM] Error in state %s: %s", state.name, exc)
                context["error"] = str(exc)
                if state not in (State.CLEANUP, State.EXIT):
                    current_state_idx = cleanup_idx
                    continue
                break
            current_state_idx += 1

        elapsed = time.time() - context.get("start_time", time.time())
        logger.info("[SM] Cycle complete in %.2fs.", elapsed)
        return context
