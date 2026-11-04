"""Animador: um único relógio para luzes ambiente, tweens e rolagem suave."""

from __future__ import annotations

import time
import traceback


def ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def linear(t: float) -> float:
    return t


class Animator:
    def __init__(self, root) -> None:
        self.root = root
        self._tickers: list[tuple[str, object]] = []
        self._tweens: dict = {}
        self._job = None
        self._closed = False
        self._t0 = self._last = time.perf_counter()
        self._schedule()

    # -- tickers: funções chamadas a cada frame ---------------------------
    def add(self, fn, group: str = "scene"):
        self._tickers.append((group, fn))
        return fn

    def clear(self, group: str) -> None:
        self._tickers = [item for item in self._tickers if item[0] != group]

    # -- tweens: transições com começo e fim -------------------------------
    def tween(self, duration, fn, *, ease=ease_out, key=None, on_done=None):
        if key is None:
            key = object()
        self._tweens[key] = [time.perf_counter(), max(0.001, duration), fn, ease, on_done]
        return key

    def cancel(self, key) -> None:
        self._tweens.pop(key, None)

    def stop(self) -> None:
        self._closed = True
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    # -- laço --------------------------------------------------------------
    def _schedule(self) -> None:
        if self._closed:
            return
        self._job = self.root.after(16 if self._tweens else 33, self._frame)

    def _frame(self) -> None:
        self._job = None
        if self._closed:
            return
        now = time.perf_counter()
        t, dt = now - self._t0, now - self._last
        self._last = now

        for entry in list(self._tickers):
            try:
                entry[1](t, dt)
            except Exception:
                traceback.print_exc()
                if entry in self._tickers:
                    self._tickers.remove(entry)

        finished = []
        for key, (start, dur, fn, ease, on_done) in list(self._tweens.items()):
            p = min(1.0, (now - start) / dur)
            try:
                fn(ease(p))
            except Exception:
                traceback.print_exc()
                finished.append((key, None))
                continue
            if p >= 1.0:
                finished.append((key, on_done))
        for key, on_done in finished:
            self._tweens.pop(key, None)
            if on_done:
                try:
                    on_done()
                except Exception:
                    traceback.print_exc()

        self._schedule()
