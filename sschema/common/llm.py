"""Chat LLM client factory with HTTP keep-alive pooling. Copied and pruned from
utils/util.py (fb-2s).

Environment variables:
  LLM_BASE_URL      OpenAI-compatible endpoint  (default http://localhost:4142/v1)
  LLM_MODEL         model name                  (default gpt-4o)
  OPENAI_API_KEY    api key                     (default "sk-placeholder")
  LLM_MAX_WORKERS   http pool size              (default 100)

Note: A single singleton OpenAI client is shared across threads so the
underlying httpx pool is reused — new TCP sockets per call would exhaust the
ssh-tunnel fd limit (see memory: tunnel_fd_limit).
"""
import os
import threading

import httpx
from openai import OpenAI


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4142/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-placeholder")


_chat_client = None
_client_lock = threading.Lock()


def _build_http_client(pool_size):
    limits = httpx.Limits(
        max_connections=pool_size,
        max_keepalive_connections=pool_size,
        keepalive_expiry=300.0,
    )
    return httpx.Client(limits=limits)


def make_chat_client():
    global _chat_client
    if _chat_client is None:
        with _client_lock:
            if _chat_client is None:
                pool = int(os.environ.get("LLM_MAX_WORKERS", "100"))
                _chat_client = OpenAI(
                    base_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    timeout=600.0,
                    max_retries=0,
                    http_client=_build_http_client(pool),
                )
    return _chat_client
