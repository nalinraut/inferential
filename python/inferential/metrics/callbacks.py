from __future__ import annotations

from typing import Callable

MetricCallback = Callable[[str, float, dict[str, str]], None]


class CallbackRegistry:
    def __init__(self) -> None:
        self._callbacks: list[MetricCallback] = []

    def register(self, callback: MetricCallback) -> None:
        self._callbacks.append(callback)

    def fire(self, name: str, value: float, labels: dict[str, str]) -> None:
        for cb in self._callbacks:
            try:
                cb(name, value, labels)
            except Exception:
                pass  # Callbacks must not crash the server
