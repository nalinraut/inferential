from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Callable

from inferential.config.schema import InferentialConfig

logger = logging.getLogger("inferential.config")

ReloadCallback = Callable[[InferentialConfig], None]


class ConfigWatcher:
    def __init__(
        self,
        config_path: str | Path | None = None,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._config_path = Path(config_path) if config_path else None
        self._poll_interval_s = poll_interval_s
        self._callbacks: list[ReloadCallback] = []
        self._last_mtime: float = 0.0
        self._sighup_received = False

    def on_reload(self, callback: ReloadCallback) -> ReloadCallback:
        self._callbacks.append(callback)
        return callback

    def _install_sighup_handler(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGHUP, self._handle_sighup)
        except (NotImplementedError, OSError):
            pass  # Windows or restricted environment

    def _handle_sighup(self) -> None:
        self._sighup_received = True
        logger.info("SIGHUP received, will reload config")

    def _load_config(self) -> InferentialConfig | None:
        if self._config_path is None or not self._config_path.exists():
            return None
        try:
            data = json.loads(self._config_path.read_text())
            return InferentialConfig(**data)
        except Exception as e:
            logger.error("Failed to reload config from %s: %s", self._config_path, e)
            return None

    def _fire(self, config: InferentialConfig) -> None:
        for cb in self._callbacks:
            try:
                cb(config)
            except Exception:
                logger.exception("Error in config reload callback")

    async def watch(self) -> None:
        self._install_sighup_handler()

        while True:
            await asyncio.sleep(self._poll_interval_s)

            should_reload = self._sighup_received

            if self._config_path and self._config_path.exists():
                try:
                    mtime = os.path.getmtime(self._config_path)
                    if mtime > self._last_mtime:
                        should_reload = True
                        self._last_mtime = mtime
                except OSError:
                    pass

            if should_reload:
                self._sighup_received = False
                config = self._load_config()
                if config is not None:
                    logger.info("Config reloaded from %s", self._config_path)
                    self._fire(config)
