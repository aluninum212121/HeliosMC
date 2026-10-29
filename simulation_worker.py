"""One cooperative background job, polled by the GUI thread."""
from enum import Enum
from queue import Queue, Empty
from threading import Event, Thread
from physics import SimulationCancelled


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SimulationWorker:
    def __init__(self):
        self.state = ExecutionState.IDLE
        self._messages = Queue(maxsize=1)
        self.cancel_event = Event()
        self.thread = None

    def start(self, function, **config):
        if self.state == ExecutionState.RUNNING or (self.thread and self.thread.is_alive()):
            raise RuntimeError("A simulation is already running")
        self.cancel_event.clear()
        self.state = ExecutionState.RUNNING

        def run():
            try:
                result = function(cancel_event=self.cancel_event, **config)
                message = (ExecutionState.CANCELLED, None) if self.cancel_event.is_set() else (ExecutionState.COMPLETED, result)
            except SimulationCancelled:
                message = (ExecutionState.CANCELLED, None)
            except Exception as exc:
                message = (ExecutionState.FAILED, f"{type(exc).__name__}: {exc}")
            self._messages.put(message)

        self.thread = Thread(target=run, name="HeliosMC simulation", daemon=False)
        try:
            self.thread.start()
        except Exception:
            self.state = ExecutionState.FAILED
            raise

    def cancel(self):
        if self.state == ExecutionState.RUNNING:
            self.cancel_event.set()

    def poll(self):
        # Wait until the thread has exited before permitting another start.
        if self.thread and self.thread.is_alive():
            return None
        try:
            state, payload = self._messages.get_nowait()
        except Empty:
            return None
        if self.cancel_event.is_set() and state == ExecutionState.COMPLETED:
            state, payload = ExecutionState.CANCELLED, None
        self.state = state
        self.thread = None
        return state, payload
