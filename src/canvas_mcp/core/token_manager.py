"""OAuth2 token manager for Canvas API.

Supports two authentication modes:
- static: Uses CANVAS_API_TOKEN directly (default)
- oauth2: Uses CANVAS_CLIENT_ID, CANVAS_CLIENT_SECRET, and CANVAS_REFRESH_TOKEN
  to obtain and refresh access tokens automatically.

Thread-safe: concurrent token requests wait for a single in-flight refresh
rather than each triggering their own.
"""

import asyncio
import os
import time
from typing import Any

import httpx

from .config import get_config
from .logging import log_error, log_info, log_warning


class TokenManager:
    """Manages Canvas API tokens with optional OAuth2 refresh support."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._current_token: str = ""
        self._expires_at: float = 0
        self._refresh_token: str = ""
        self._initialized = False

    def _ensure_lock(self) -> asyncio.Lock:
        """Get or create the async lock for the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def initialize(self) -> None:
        """Initialize the token manager based on auth mode."""
        if self._initialized:
            return

        config = get_config()

        if config.auth_mode == "oauth2":
            # Load initial refresh token from config
            self._refresh_token = config.refresh_token
            # Try to load cached access token
            self._load_cached_token()
            # If no cached token or expired, do an initial refresh
            if not self._current_token or time.time() >= self._expires_at:
                await self._refresh_access_token()
        else:
            # Static mode: use the configured token directly
            self._current_token = config.canvas_api_token
            self._expires_at = float("inf")  # Never expires

        self._initialized = True

    def _load_cached_token(self) -> None:
        """Load cached access token from .token_cache file if available."""
        cache_file = self._get_cache_file_path()
        if not os.path.exists(cache_file):
            return

        try:
            with open(cache_file, "r") as f:
                data = f.read().strip()
                if ":" in data:
                    token, expires_str = data.split(":", 1)
                    self._current_token = token
                    self._expires_at = float(expires_str)
                    log_info("Loaded cached OAuth2 access token")
        except (ValueError, OSError) as e:
            log_warning(f"Failed to load token cache: {e}")

    def _save_cached_token(self) -> None:
        """Save access token to .token_cache file."""
        if not self._current_token or self._expires_at == float("inf"):
            return

        cache_file = self._get_cache_file_path()
        try:
            with open(cache_file, "w") as f:
                f.write(f"{self._current_token}:{self._expires_at}")
            log_info("Cached OAuth2 access token")
        except OSError as e:
            log_warning(f"Failed to cache token: {e}")

    def _get_cache_file_path(self) -> str:
        """Get path for token cache file."""
        config = get_config()
        # Store cache next to .env file
        env_dir = os.path.dirname(os.path.abspath(".env")) if os.path.exists(".env") else "."
        return os.path.join(env_dir, ".token_cache")

    async def get_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            A valid access token string.

        Raises:
            RuntimeError: If token refresh fails.
        """
        await self.initialize()

        config = get_config()
        if config.auth_mode == "static":
            return self._current_token

        # Check if token needs refresh (within 5 minutes of expiry)
        if time.time() >= self._expires_at - 300:
            lock = self._ensure_lock()
            async with lock:
                # Double-check after acquiring lock
                if time.time() >= self._expires_at - 300:
                    await self._refresh_access_token()

        return self._current_token

    async def _refresh_access_token(self) -> None:
        """Refresh the access token using the refresh token.

        Raises:
            RuntimeError: If refresh fails.
        """
        config = get_config()

        log_info("Refreshing OAuth2 access token...")

        try:
            async with httpx.AsyncClient(timeout=config.api_timeout) as client:
                response = await client.post(
                    config.oauth_token_url,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": config.client_id,
                        "client_secret": config.client_secret,
                        "refresh_token": self._refresh_token,
                    },
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise RuntimeError(
                        f"Token refresh failed (HTTP {response.status_code}): {error_text}"
                    )

                data = response.json()

                # Update tokens
                self._current_token = data.get("access_token", "")
                new_refresh = data.get("refresh_token")
                if new_refresh:
                    self._refresh_token = new_refresh

                # Calculate expiry (expires_in is in seconds)
                expires_in = data.get("expires_in", 3600)
                self._expires_at = time.time() + expires_in

                # Persist new tokens
                self._save_cached_token()
                self._update_env_tokens()

                log_info(
                    f"OAuth2 token refreshed successfully (expires in {expires_in}s)"
                )

        except httpx.TimeoutException:
            raise RuntimeError("Token refresh timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Token refresh network error: {e}")

    def _update_env_tokens(self) -> None:
        """Update tokens in .env file for persistence across restarts."""
        env_file = ".env"
        if not os.path.exists(env_file):
            return

        try:
            with open(env_file, "r") as f:
                lines = f.readlines()

            updated_lines = []
            refresh_token_updated = False

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("CANVAS_REFRESH_TOKEN="):
                    updated_lines.append(f"CANVAS_REFRESH_TOKEN={self._refresh_token}\n")
                    refresh_token_updated = True
                elif stripped.startswith("CANVAS_API_TOKEN=") and self._current_token:
                    # In oauth2 mode, update the API token too for fallback
                    updated_lines.append(f"CANVAS_API_TOKEN={self._current_token}\n")
                else:
                    updated_lines.append(line)

            # Add refresh token if not found
            if not refresh_token_updated and self._refresh_token:
                updated_lines.append(f"CANVAS_REFRESH_TOKEN={self._refresh_token}\n")

            with open(env_file, "w") as f:
                f.writelines(updated_lines)

            log_info("Updated tokens in .env file")

        except OSError as e:
            log_warning(f"Failed to update .env file with new tokens: {e}")


# Global token manager instance
_token_manager: TokenManager | None = None


def get_token_manager() -> TokenManager:
    """Get or create the global token manager."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


async def get_valid_token() -> str:
    """Get a valid Canvas API token.

    In static mode, returns CANVAS_API_TOKEN directly.
    In oauth2 mode, returns a fresh access token (refreshing if needed).
    """
    manager = get_token_manager()
    return await manager.get_token()


def reset_token_manager() -> None:
    """Reset the global token manager (for tests)."""
    global _token_manager
    _token_manager = None
