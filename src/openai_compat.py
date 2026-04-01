import os
from typing import Optional

import httpx
from openai import OpenAI

try:
    from openai._constants import DEFAULT_TIMEOUT
except Exception:
    DEFAULT_TIMEOUT = httpx.Timeout(60.0)


def build_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    Create an OpenAI client that stays compatible across httpx versions.

    Older OpenAI packages may try to construct ``httpx.Client(proxies=...)``,
    which breaks on newer httpx releases that renamed the argument to
    ``proxy``. When that mismatch appears, fall back to an explicitly
    constructed httpx client.
    """
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except TypeError as exc:
        if "unexpected keyword argument 'proxies'" not in str(exc):
            raise

    http_client = httpx.Client(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
        trust_env=not bool(
            os.getenv("OPENAI_HTTP_PROXY") or os.getenv("OPENAI_HTTPS_PROXY")
        ),
        proxy=os.getenv("OPENAI_HTTPS_PROXY") or os.getenv("OPENAI_HTTP_PROXY"),
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
    )
