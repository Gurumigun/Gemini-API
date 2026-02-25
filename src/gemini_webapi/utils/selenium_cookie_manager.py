"""
Selenium/UC 기반 Google Gemini 쿠키 관리자.

undetected-chromedriver를 사용하여 Google 봇 감지를 우회하고,
브라우저 프로필에서 Gemini 인증 쿠키를 추출합니다.
"""

from __future__ import annotations

import asyncio
import json as stdlib_json
import os
import signal
import time
from pathlib import Path

from .logger import logger

GEMINI_URL = "https://gemini.google.com/app"
REQUIRED_COOKIE = "__Secure-1PSID"
OPTIONAL_COOKIE = "__Secure-1PSIDTS"
DEFAULT_CACHE_FILE = ".gemini_cookies.json"


class SeleniumCookieManager:
    """
    undetected-chromedriver 기반 쿠키 관리자.

    Chrome 프로필 디렉토리를 사용하여 로그인 상태를 유지하고,
    Gemini 인증에 필요한 쿠키를 추출합니다.

    Parameters
    ----------
    profile_dir : str | None
        Chrome 프로필 디렉토리 경로. 지정 시 기존 프로필에서 쿠키 추출 (로그인 없이).
        미지정 시 임시 프로필 생성.
    headless : bool
        헤드리스 모드. 쿠키 갱신 시만 사용하며, 최초 수동 로그인은 GUI 필요.
    browser_executable_path : str | None
        Chrome 바이너리 경로 (예: CfT 경로). None이면 시스템 Chrome 사용.
    driver_executable_path : str | None
        ChromeDriver 바이너리 경로 (예: CfT 경로). None이면 자동 감지.
    """

    def __init__(
        self,
        profile_dir: str | None = None,
        headless: bool = False,
        browser_executable_path: str | None = None,
        driver_executable_path: str | None = None,
    ):
        self.profile_dir = profile_dir
        self.headless = headless
        self.browser_executable_path = browser_executable_path
        self.driver_executable_path = driver_executable_path
        self._cookies: dict[str, str] = {}

    def _create_driver(self):
        """undetected-chromedriver 인스턴스 생성. CfT를 최우선으로 사용."""
        try:
            import undetected_chromedriver as uc
        except ImportError:
            raise ImportError(
                "undetected-chromedriver가 설치되지 않았습니다. "
                '`pip install -e ".[selenium]"` 로 설치해 주세요.'
            )

        options = uc.ChromeOptions()

        # CfT 최우선 사용: 항상 CfT를 먼저 탐색하고, 없을 때만 명시적 경로/시스템 Chrome 사용
        browser_path = None
        driver_path = None
        try:
            from .cft_detector import detect_cft_paths

            cft_browser, cft_driver = detect_cft_paths()
            if cft_browser:
                browser_path = cft_browser
                driver_path = cft_driver
                logger.info(f"CfT 최우선 사용: browser={cft_browser}, driver={cft_driver}")
        except Exception as e:
            logger.debug(f"CfT 자동 감지 실패: {e}")

        # CfT를 찾지 못한 경우 명시적 경로 사용
        if not browser_path:
            browser_path = self.browser_executable_path
        if not driver_path:
            driver_path = self.driver_executable_path

        if browser_path:
            options.binary_location = browser_path

        if self.profile_dir:
            profile_path = Path(self.profile_dir).expanduser().resolve()
            profile_path.mkdir(parents=True, exist_ok=True)
            self._repair_profile_if_corrupted(profile_path)
            options.add_argument(f"--user-data-dir={profile_path}")

        if self.headless:
            options.add_argument("--headless=new")

        # 렌더러 연결 안정성 향상 옵션
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # 비정상 종료 후 세션 복원 프롬프트 방지
        options.add_argument("--no-first-run")
        options.add_argument("--disable-session-crashed-bubble")
        options.add_argument("--hide-crash-restore-bubble")

        kwargs = {}
        if driver_path:
            kwargs["driver_executable_path"] = driver_path

        # Chrome 메이저 버전을 감지하여 ChromeDriver 버전 불일치 방지
        version_main = self._detect_chrome_major_version(browser_path)
        if version_main:
            kwargs["version_main"] = version_main
            logger.debug(f"Chrome 메이저 버전 감지: {version_main}")

        driver = uc.Chrome(options=options, **kwargs)
        return driver

    def _detect_chrome_major_version(self, browser_path: str | None) -> int | None:
        """Chrome 바이너리에서 메이저 버전 번호를 추출."""
        import subprocess

        candidates = []
        if browser_path:
            candidates.append(browser_path)
        # 시스템 Chrome 경로 후보
        import platform
        system = platform.system()
        if system == "Darwin":
            candidates.append(
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            )
        elif system == "Linux":
            candidates.extend(["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"])

        for path in candidates:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # "Google Chrome 144.0.7559.132" → 144
                version_str = result.stdout.strip()
                parts = version_str.split()
                if parts:
                    major = int(parts[-1].split(".")[0])
                    return major
            except Exception:
                continue
        return None

    async def login_and_get_cookies(self, timeout: int = 300, max_retries: int = 2) -> dict[str, str]:
        """
        UC 브라우저를 열고 사용자가 Google 로그인할 때까지 대기 후 쿠키 추출.

        브라우저가 열리면 사용자가 직접 Google 계정으로 로그인해야 합니다.
        __Secure-1PSID 쿠키가 감지되면 자동으로 추출하고 브라우저를 닫습니다.

        Parameters
        ----------
        timeout : int
            로그인 대기 시간 (초). 기본값 300초 (5분).
        max_retries : int
            세션 생성 실패 시 재시도 횟수. 기본값 2.

        Returns
        -------
        dict[str, str]
            추출된 쿠키 딕셔너리 (__Secure-1PSID, __Secure-1PSIDTS 등).

        Raises
        ------
        TimeoutError
            지정된 시간 내에 로그인이 완료되지 않은 경우.
        """
        driver = await self._create_driver_with_retry(max_retries)
        try:
            await asyncio.to_thread(driver.get, GEMINI_URL)
            logger.info(
                f"브라우저가 열렸습니다. {timeout}초 내에 Google 계정으로 로그인하세요."
            )

            cookies = await self._poll_for_cookie(driver, timeout)
            self._cookies = cookies

            # 프로필 디렉토리가 있으면 쿠키를 캐시 파일에 저장
            if self.profile_dir:
                cache_path = (
                    Path(self.profile_dir).expanduser().resolve() / DEFAULT_CACHE_FILE
                )
                self._save_cookies_to_file(str(cache_path))

            logger.success("쿠키 추출 완료.")
            return cookies
        finally:
            try:
                await asyncio.to_thread(driver.quit)
            except Exception:
                pass

    async def get_cookies_from_profile(self, max_retries: int = 2) -> dict[str, str]:
        """
        이미 로그인된 Chrome 프로필에서 쿠키 추출.

        별도 로그인 없이 기존 프로필 디렉토리의 세션을 사용합니다.

        Parameters
        ----------
        max_retries : int
            세션 생성 실패 시 재시도 횟수. 기본값 2.

        Returns
        -------
        dict[str, str]
            추출된 쿠키 딕셔너리.

        Raises
        ------
        ValueError
            profile_dir이 설정되지 않은 경우.
        RuntimeError
            프로필에서 필수 쿠키를 찾을 수 없는 경우.
        """
        if not self.profile_dir:
            raise ValueError(
                "profile_dir이 설정되지 않았습니다. "
                "기존 프로필을 사용하려면 profile_dir을 지정하세요."
            )

        driver = await self._create_driver_with_retry(max_retries)
        try:
            await asyncio.to_thread(driver.get, GEMINI_URL)

            # 프로필이 이미 로그인 상태이므로 짧은 대기 후 쿠키 추출
            cookies = await self._poll_for_cookie(driver, timeout=30)
            self._cookies = cookies
            logger.success("프로필에서 쿠키 추출 완료.")
            return cookies
        finally:
            try:
                await asyncio.to_thread(driver.quit)
            except Exception:
                pass

    async def refresh_cookies(self) -> dict[str, str]:
        """
        저장된 프로필로 쿠키 갱신.

        profile_dir이 있으면 get_cookies_from_profile(),
        없으면 login_and_get_cookies()를 호출합니다.

        Returns
        -------
        dict[str, str]
            갱신된 쿠키 딕셔너리.
        """
        if self.profile_dir:
            return await self.get_cookies_from_profile()
        return await self.login_and_get_cookies()

    def save_cookies(self, path: str) -> None:
        """
        현재 쿠키를 JSON 파일로 저장.

        Parameters
        ----------
        path : str
            저장할 파일 경로.
        """
        self._save_cookies_to_file(path)

    def load_cookies(self, path: str) -> dict[str, str] | None:
        """
        JSON 파일에서 쿠키 로드.

        Parameters
        ----------
        path : str
            로드할 파일 경로.

        Returns
        -------
        dict[str, str] | None
            로드된 쿠키 딕셔너리. 파일이 없거나 잘못된 형식이면 None.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"쿠키 파일을 찾을 수 없습니다: {path}")
            return None

        try:
            data = stdlib_json.loads(file_path.read_text(encoding="utf-8"))
            if REQUIRED_COOKIE in data:
                self._cookies = data
                return data
            logger.warning(f"쿠키 파일에 {REQUIRED_COOKIE}이 없습니다.")
            return None
        except (stdlib_json.JSONDecodeError, OSError) as e:
            logger.warning(f"쿠키 파일 로드 실패: {e}")
            return None

    @property
    def cookies(self) -> dict[str, str]:
        """현재 저장된 쿠키 반환."""
        return self._cookies

    # --- 내부 메소드 ---



    def _repair_profile_if_corrupted(self, profile_path: Path) -> None:
        """손상된 Chrome 프로필을 복구하여 브라우저 시작 실패를 방지.

        Preferences/Local State 파일이 0바이트이거나 유효한 JSON이 아니면
        해당 파일을 삭제하여 Chrome이 새로 생성하도록 함.
        """
        import json as _json

        targets = ["Default/Preferences", "Local State"]
        for rel in targets:
            fpath = profile_path / rel
            if not fpath.exists():
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if not content.strip():
                    raise ValueError("빈 파일")
                _json.loads(content)
            except (ValueError, _json.JSONDecodeError) as e:
                logger.warning(f"손상된 프로필 파일 감지 ({rel}): {e} → 삭제하여 재생성 유도")
                try:
                    fpath.unlink()
                except OSError as oe:
                    logger.warning(f"프로필 파일 삭제 실패 ({rel}): {oe}")

    def _cleanup_profile_locks(self) -> None:
        """Chrome 프로필 디렉토리의 잠금 파일을 정리하여 세션 재생성을 가능하게 함."""
        if not self.profile_dir:
            return

        profile_path = Path(self.profile_dir)
        if not profile_path.exists():
            return

        lock_files = ["SingletonLock", "SingletonSocket", "SingletonCookie", "lockfile"]
        for lock_name in lock_files:
            lock_path = profile_path / lock_name
            if not lock_path.exists() and not lock_path.is_symlink():
                continue

            # Linux/Mac: SingletonLock은 symlink로 PID를 포함함
            if lock_name == "SingletonLock" and os.name != "nt":
                try:
                    link_target = os.readlink(str(lock_path))
                    # 형식: hostname-pid
                    pid_str = link_target.split("-")[-1]
                    pid = int(pid_str)
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Chrome 프로세스 종료 (PID: {pid})")
                except (ValueError, ProcessLookupError, OSError):
                    pass

            try:
                lock_path.unlink(missing_ok=True)
                logger.info(f"잠금 파일 제거: {lock_path.name}")
            except OSError as e:
                logger.warning(f"잠금 파일 제거 실패 ({lock_path.name}): {e}")

    async def _create_driver_with_retry(self, max_retries: int = 2):
        """세션 생성 실패 시 프로필 잠금을 정리하고 재시도하며 드라이버를 생성."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self._cleanup_profile_locks()
                    wait_sec = 3 * attempt
                    logger.info(f"브라우저 세션 재시도 ({attempt}/{max_retries}), {wait_sec}초 대기...")
                    await asyncio.sleep(wait_sec)
                driver = await asyncio.to_thread(self._create_driver)
                return driver
            except Exception as e:
                last_error = e
                error_msg = str(e)
                if "session not created" in error_msg or "unable to connect to renderer" in error_msg:
                    logger.warning(f"브라우저 세션 생성 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
                    continue
                raise
        raise last_error

    async def _poll_for_cookie(
        self, driver, timeout: int
    ) -> dict[str, str]:
        """브라우저에서 필수 쿠키가 나타날 때까지 폴링."""
        start = time.time()
        while time.time() - start < timeout:
            browser_cookies = await asyncio.to_thread(driver.get_cookies)
            cookie_dict = {c["name"]: c["value"] for c in browser_cookies}

            if REQUIRED_COOKIE in cookie_dict and cookie_dict[REQUIRED_COOKIE]:
                result = {REQUIRED_COOKIE: cookie_dict[REQUIRED_COOKIE]}
                if OPTIONAL_COOKIE in cookie_dict:
                    result[OPTIONAL_COOKIE] = cookie_dict[OPTIONAL_COOKIE]
                return result

            await asyncio.sleep(2)

        raise TimeoutError(
            f"{timeout}초 내에 {REQUIRED_COOKIE} 쿠키를 찾을 수 없습니다. "
            "로그인이 완료되었는지 확인하세요."
        )

    def _save_cookies_to_file(self, path: str) -> None:
        """쿠키를 JSON 파일에 저장."""
        if not self._cookies:
            logger.warning("저장할 쿠키가 없습니다.")
            return

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            stdlib_json.dumps(self._cookies, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(f"쿠키 저장 완료: {file_path}")
