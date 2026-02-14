"""
~/.cft/ Chrome for Testing(CfT) 자동 감지 (경량 버전).

외부 의존성 없이 stdlib만 사용하여 ~/.cft/ 디렉토리에서
CfT 바이너리와 ChromeDriver를 자동 탐색합니다.
"""

from __future__ import annotations

import platform
from pathlib import Path

_CFT_ROOT = Path.home() / ".cft"


def detect_cft_paths() -> tuple[str | None, str | None]:
    """~/.cft/에서 CfT 바이너리를 자동 탐색.

    Returns
    -------
    tuple[str | None, str | None]
        (browser_path, driver_path). 미설치 시 (None, None).
    """
    if not _CFT_ROOT.exists():
        return None, None

    system = platform.system()

    if system == "Darwin":
        browser_names = [
            "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            "chrome",
        ]
        driver_name = "chromedriver"
    elif system == "Linux":
        browser_names = ["chrome"]
        driver_name = "chromedriver"
    else:  # Windows
        browser_names = ["chrome.exe"]
        driver_name = "chromedriver.exe"

    chrome_dirs = sorted(
        [d for d in _CFT_ROOT.iterdir() if d.is_dir() and d.name.startswith("chrome-")],
        reverse=True,
    )
    driver_dirs = sorted(
        [d for d in _CFT_ROOT.iterdir() if d.is_dir() and d.name.startswith("chromedriver-")],
        reverse=True,
    )

    browser_path = _find_binary(chrome_dirs, browser_names)
    driver_path = _find_binary(driver_dirs, [driver_name])

    return browser_path, driver_path


def _find_binary(dirs: list[Path], names: list[str]) -> str | None:
    """디렉토리 목록에서 바이너리 탐색 (서브디렉토리 포함)."""
    for d in dirs:
        for name in names:
            candidate = d / name
            if candidate.exists():
                return str(candidate)
            for sub in d.iterdir():
                if sub.is_dir():
                    candidate = sub / name
                    if candidate.exists():
                        return str(candidate)
    return None
