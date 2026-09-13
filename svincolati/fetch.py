"""Rate-limited Playwright fetcher. Only file that touches the network."""
from __future__ import annotations

import json
import time

from playwright.sync_api import sync_playwright

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
        return self

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()

    def _goto(self, url: str, wait_selector: str | None) -> None:
        for attempt in range(2):
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                if wait_selector:
                    self._page.wait_for_selector(wait_selector, timeout=30_000)
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
        self._goto(url, None)
        txt = self._page.inner_text("body")
        time.sleep(self.sleep_s)
        return json.loads(txt)
