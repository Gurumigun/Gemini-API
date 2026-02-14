"""
Selenium/UC 기반 Google Gemini 쿠키 관리자.

undetected-chromedriver를 사용하여 Google 봇 감지를 우회하고,
브라우저 프로필에서 Gemini 인증 쿠키를 추출합니다.
"""

from __future__ import annotations

import asyncio
import json as stdlib_json
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
        """undetected-chromedriver 인스턴스 생성."""
        try:
            import undetected_chromedriver as uc
        except ImportError:
            raise ImportError(
                "undetected-chromedriver가 설치되지 않았습니다. "
                '`pip install -e ".[selenium]"` 로 설치해 주세요.'
            )

        options = uc.ChromeOptions()

        # CfT 자동 감지: 명시적 경로가 없으면 ~/.cft/ 탐색
        browser_path = self.browser_executable_path
        driver_path = self.driver_executable_path
        if not browser_path:
            try:
                from .cft_detector import detect_cft_paths

                cft_browser, cft_driver = detect_cft_paths()
                if cft_browser:
                    browser_path = cft_browser
                    driver_path = driver_path or cft_driver
                    logger.info(f"CfT 자동 감지: browser={cft_browser}")
            except Exception as e:
                logger.debug(f"CfT 자동 감지 실패 (무시): {e}")

        if browser_path:
            options.binary_location = browser_path

        if self.profile_dir:
            profile_path = Path(self.profile_dir).expanduser().resolve()
            profile_path.mkdir(parents=True, exist_ok=True)
            options.add_argument(f"--user-data-dir={profile_path}")

        if self.headless:
            options.add_argument("--headless=new")

        kwargs = {}
        if driver_path:
            kwargs["driver_executable_path"] = driver_path

        driver = uc.Chrome(options=options, **kwargs)
        return driver

    async def login_and_get_cookies(self, timeout: int = 300) -> dict[str, str]:
        """
        UC 브라우저를 열고 사용자가 Google 로그인할 때까지 대기 후 쿠키 추출.

        브라우저가 열리면 사용자가 직접 Google 계정으로 로그인해야 합니다.
        __Secure-1PSID 쿠키가 감지되면 자동으로 추출하고 브라우저를 닫습니다.

        Parameters
        ----------
        timeout : int
            로그인 대기 시간 (초). 기본값 300초 (5분).

        Returns
        -------
        dict[str, str]
            추출된 쿠키 딕셔너리 (__Secure-1PSID, __Secure-1PSIDTS 등).

        Raises
        ------
        TimeoutError
            지정된 시간 내에 로그인이 완료되지 않은 경우.
        """
        driver = None
        try:
            driver = await asyncio.to_thread(self._create_driver)
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
            if driver:
                await asyncio.to_thread(driver.quit)

    async def get_cookies_from_profile(self) -> dict[str, str]:
        """
        이미 로그인된 Chrome 프로필에서 쿠키 추출.

        별도 로그인 없이 기존 프로필 디렉토리의 세션을 사용합니다.

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

        driver = None
        try:
            driver = await asyncio.to_thread(self._create_driver)
            await asyncio.to_thread(driver.get, GEMINI_URL)

            # 프로필이 이미 로그인 상태이므로 짧은 대기 후 쿠키 추출
            cookies = await self._poll_for_cookie(driver, timeout=30)
            self._cookies = cookies
            logger.success("프로필에서 쿠키 추출 완료.")
            return cookies
        finally:
            if driver:
                await asyncio.to_thread(driver.quit)

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
