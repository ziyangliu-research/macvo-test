from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Optional, TypeVar

from .contracts import StopSignal, WorkerFailure

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class QueuePolicy:
    maxsize: int = 2
    put_timeout_sec: float = 1.0
    get_timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        if self.maxsize <= 0:
            raise ValueError("queue maxsize must be positive")
        if self.put_timeout_sec <= 0 or self.get_timeout_sec <= 0:
            raise ValueError("queue timeouts must be positive")


class ThreadWorker(Generic[TIn, TOut]):
    """Bounded, fail-fast worker with an explicit initialization barrier.

    Each GPU component is initialized exactly once inside its owning thread. The
    coordinator waits for all workers to become ready before reading the first
    frame, avoiding concurrent setup races and excluding model loading from the
    streaming throughput measurement.
    """

    def __init__(
        self,
        name: str,
        process: Callable[[TIn], TOut | Iterable[TOut] | None],
        *,
        policy: QueuePolicy = QueuePolicy(),
        setup: Optional[Callable[[], None]] = None,
        teardown: Optional[Callable[[], Iterable[TOut] | TOut | None]] = None,
    ) -> None:
        self.name = name
        self.process = process
        self.setup = setup
        self.teardown = teardown
        self.policy = policy
        self.input: queue.Queue[TIn | StopSignal] = queue.Queue(maxsize=policy.maxsize)
        # Input queues enforce backpressure. Outputs are unbounded to prevent a
        # producer/close deadlock; outstanding frames remain bounded by upstream
        # input queues and the ordered joiner.
        self.output: queue.Queue[TOut | StopSignal | WorkerFailure] = queue.Queue()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=False)

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: Optional[float] = None) -> None:
        if not self._ready_event.wait(timeout):
            raise TimeoutError(f"worker {self.name} did not initialize in time")
        self.raise_if_failed()
        if not self._thread.is_alive():
            self.raise_if_failed()
            raise RuntimeError(f"worker {self.name} stopped during initialization")

    def submit(self, item: TIn) -> None:
        while True:
            self.raise_if_failed()
            try:
                self.input.put(item, timeout=self.policy.put_timeout_sec)
                return
            except queue.Full:
                if not self._thread.is_alive():
                    self.raise_if_failed()
                    raise RuntimeError(f"worker {self.name} stopped while submitting input")

    def close_input(self) -> None:
        while True:
            self.raise_if_failed()
            try:
                self.input.put(
                    StopSignal(source="producer"),
                    timeout=self.policy.put_timeout_sec,
                )
                return
            except queue.Full:
                if not self._thread.is_alive():
                    self.raise_if_failed()
                    raise RuntimeError(f"worker {self.name} stopped while closing input")

    def join(self) -> None:
        self._thread.join()
        self.raise_if_failed()

    def request_stop(self) -> None:
        self._stop_event.set()

    def raise_if_failed(self) -> None:
        with self.output.mutex:
            items = list(self.output.queue)
        for item in items:
            if isinstance(item, WorkerFailure):
                raise RuntimeError(
                    f"worker {item.worker} failed: {item.message}\n{item.traceback_text}"
                )

    def _emit(self, value: TOut | Iterable[TOut] | None) -> None:
        if value is None:
            return
        if isinstance(value, (str, bytes, dict)):
            self.output.put(value)  # type: ignore[arg-type]
            return
        try:
            iterator = iter(value)  # type: ignore[arg-type]
        except TypeError:
            self.output.put(value)  # type: ignore[arg-type]
            return
        for item in iterator:
            self.output.put(item)

    def _run(self) -> None:
        try:
            if self.setup is not None:
                self.setup()
            self._ready_event.set()
            while not self._stop_event.is_set():
                try:
                    item = self.input.get(timeout=self.policy.get_timeout_sec)
                except queue.Empty:
                    continue
                if isinstance(item, StopSignal):
                    break
                self._emit(self.process(item))
            if self.teardown is not None:
                self._emit(self.teardown())
            self.output.put(StopSignal(source=self.name))
        except BaseException as exc:
            self.output.put(
                WorkerFailure(
                    worker=self.name,
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                )
            )
            self._stop_event.set()
            self._ready_event.set()
