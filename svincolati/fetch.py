"""Rate-limited Playwright fetcher. Only file that touches the network."""
from __future__ import annotations

import json
import time

from playwright.sync_api import sync_playwright

from .tm import LISTS

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class TM:
    def __init__(self, sleep_s: float = 1.5, headless: bool = True):
        self.sleep_s = sleep_s
        self.headless = headless

    def __enter__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_context(locale="it-IT", user_agent=UA).new_page()
        self._warm_up()
        return self

    def _warm_up(self) -> None:
        # Anti-bot challenge is solved by JS on an HTML page; JSON endpoints need the resulting cookies.
        try:
            self._goto(LISTS["A"], "table.items")
        except Exception as e:  # best effort: json() retries and raises on its own
            print(f"[fetch] warm-up failed {type(e).__name__}", flush=True)
        time.sleep(2)

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()

    def _goto(self, url: str, wait_selector: str | None) -> None:
        for attempt in range(2):
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                if wait_selector:
                    self._page.wait_for_selector(wait_selector, state="attached", timeout=30_000)
                return
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(5)

    def html(self, url: str, wait_selector: str) -> str:
        self._goto(url, wait_selector)
        time.sleep(self.sleep_s)
        return self._page.content()

    def json(self, url: str) -> dict:
        for attempt in range(2):
            self._goto(url, None)
            txt = self._page.inner_text("body")
            time.sleep(self.sleep_s)
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
                self._warm_up()
        raise RuntimeError("unreachable")
