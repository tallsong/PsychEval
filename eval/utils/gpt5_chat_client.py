# pip install openai>=1.0.0 tenacity aiolimiter
import asyncio
from typing import Any, Dict, List, Optional

import re

from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
import os
import httpx

def _is_retryable_status(exc: Exception) -> bool:
    """对 429 / 5xx 的状态异常进行重试。"""
    return isinstance(exc, APIStatusError) and (exc.status_code == 429 or 500 <= exc.status_code < 600)


_RETRY_COND = (
    retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError))
    | retry_if_exception(_is_retryable_status)
)


class GPT5ChatClient:
    """
    异步 Chat Completions 客户端封装，带并发控制 / 限流 / 重试。

    参数:
        api_key: 可选，默认读取环境变量 OPENAI_API_KEY
        base_url: 可选，自定义网关/代理时使用
        max_concurrency: 最大并发中的请求数
        rps: 每个实例的每秒请求次数 (None 关闭限流)
        rps_period: 限流周期（秒），如 10/60 表示 60 秒内最多 10 次
        default_timeout: 每次请求超时（秒）
    """
        
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        max_concurrency: int = 512,
        rps: Optional[int] = 100,      # 将 32 提高到 100 (根据你的 API 额度调整)
        rps_period: float = 1.0,       # 周期改为 1 秒，方便精确控制
        default_timeout: float = 300.0,
        model: str = "deepseek-ai/DeepSeek-V3.1-Terminus"
    ):    
        base_url = os.getenv("CHAT_API_BASE", None)
        api_key = os.getenv("CHAT_API_KEY", None)
        
        model = os.getenv("CHAT_MODEL_NAME", None)
        
        
        # openai.telemetry.disable()
        
        self.rps = rps        
        assert api_key, "API key must be provided via argument or CHAT_API_KEY env var"
        assert base_url, "Base URL must be provided via argument or CHAT_API_BASE env var"
        
        self.api_key = api_key
        
        self.base_url = base_url
        
        self.model = model
        
        proxies = ''
        http_client = httpx.AsyncClient(proxy=proxies, timeout=600)  # 如需公司 CA，可以加 verify="/path/to/cacert.pem"
        
        self._sdk = AsyncOpenAI(api_key=api_key, base_url=base_url,http_client=http_client)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._limiter = AsyncLimiter(max_rate=rps, time_period=rps_period) if rps else None
        self._default_timeout = default_timeout

    async def __aenter__(self) -> "GPT5ChatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # OpenAI SDK 当前不强制要求关闭；若未来提供 aclose，可在此调用
        pass
    
    def _strip_fences(self, s: str) -> str:
        """剥掉 ```json ... ``` 或 ``` ... ``` 代码围栏；若无则原样返回。"""
        if not isinstance(s, str):
            return s
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
        return (m.group(1) if m else s).strip()

    # ---- 公共方法 ---------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 6,
        **extra_kwargs: Any,
    ):
        """
        发送一次 Chat Completion 调用（带重试）。
        返回 SDK 的响应对象（含 choices / usage 等）。
        """
        # 动态包装重试，使 max_retries 可配置
        @retry(
            retry=_RETRY_COND,
            stop=stop_after_attempt(max_retries),
            wait=wait_random_exponential(min=1, max=60),
            reraise=True,
        )
        async def _do_call():
            # 限流（不占用并发槽）
            if self._limiter is not None:
                async with self._limiter:
                    pass
                
            self._sdk.chat.completions.parse
            # 并发闸门（仅在真正发起 HTTP 时占用）
            async with self._sem:
                return await self._sdk.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    timeout=timeout or self._default_timeout,
                    **extra_kwargs,  # 例如 temperature, tools, tool_choice, n, top_p 等
                )

        return await _do_call()

    async def chat_text(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """
        便捷方法：直接返回第一条 choice 的 message.content。
        """
        resp = await self.chat_completion(messages, **kwargs)
        
        if not resp.choices or not getattr(resp.choices[0], "message", None):
            raise RuntimeError(f"Unexpected response shape: {resp}")
        
        response = resp.choices[0].message.content or ""
        
        response = self._strip_fences(response)
        
        # response = response.replace("```json\n", "").replace("\n```", "")
        
        # response = response.replace("```json", "").replace("```", "")
        
        return response.strip()

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 6,
        **extra_kwargs: Any,
    ):
        """
        流式（server-sent events）接口。
        注意：若中途断流不会自动续传；仅在建立流前的错误会触发重试。
        使用示例：
            async for chunk in client.chat_completion_stream(...):
                ...
        """
        @retry(
            retry=_RETRY_COND,
            stop=stop_after_attempt(max_retries),
            wait=wait_random_exponential(min=1, max=60),
            reraise=True,
        )
        async def _open_stream():
            if self._limiter is not None:
                async with self._limiter:
                    pass
            async with self._sem:
                return await self._sdk.chat.completions.create(
                    # 这里加了个self
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    timeout=timeout or self._default_timeout,
                    stream=True,
                    **extra_kwargs,
                )

        stream = await _open_stream()
        async for chunk in stream:
            yield chunk

    # ---- 可选：工具方法 ----------------------------------------------------

    @staticmethod
    def to_user_text(resp) -> str:
        """从标准响应中提取首条文本内容。"""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            raise RuntimeError(f"Unexpected response shape: {resp}")


# ---------------- 使用示例 ----------------
# async def main():
#     messages = [{"role": "user", "content": "用一句话介绍量子计算"}]
#     async with OpenAIChatClient(rps=20, rps_period=60, max_concurrency=32) as client:
#         text = await client.chat_text("gpt-4o-mini", messages, temperature=0.3)
#         print(text)
#
#         # 或者拿到完整响应对象
#         resp = await client.chat_completion("gpt-4o-mini", messages)
#         print(resp.usage)
#
#         # 流式
#         async for chunk in client.chat_completion_stream("gpt-4o-mini", messages):
#             delta = chunk.choices[0].delta.content or ""
#             if delta:
#                 print(delta, end="", flush=True)
# asyncio.run(main())
