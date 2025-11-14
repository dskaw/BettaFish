"""
Unified OpenAI-compatible LLM client for the Report Engine, with retry support.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Generator
from loguru import logger

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(project_root, "utils")
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

try:
    from retry_helper import with_retry, LLM_RETRY_CONFIG
    from cli_llm import CodexCLIError, CodexCLIInvoker
except ImportError:
    def with_retry(config=None):
        def decorator(func):
            return func
        return decorator

    LLM_RETRY_CONFIG = None

    CodexCLIError = RuntimeError  # type: ignore

    class CodexCLIInvoker:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CodexCLIInvoker requires utils.cli_llm")


class LLMClient:
    """Wrapper around either an OpenAI-compatible API or the Codex CLI."""

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str,
        base_url: Optional[str] = None,
        cli_command: Optional[str] = None,
    ):
        if not model_name:
            raise ValueError("Report Engine model name is required.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.provider = model_name
        self.cli_command = cli_command
        self._cli_invoker: Optional[CodexCLIInvoker] = None

        timeout_fallback = os.getenv("LLM_REQUEST_TIMEOUT") or os.getenv("REPORT_ENGINE_REQUEST_TIMEOUT") or "3000"
        try:
            self.timeout = float(timeout_fallback)
        except ValueError:
            self.timeout = 3000.0

        if cli_command:
            self.provider = "codex-cli"
            self._cli_invoker = CodexCLIInvoker(cli_command, self.timeout)
            self.client = None  # type: ignore[assignment]
        else:
            if not api_key:
                raise ValueError("Report Engine LLM API key is required when Codex CLI command is not set.")

            client_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "max_retries": 0,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

    @with_retry(LLM_RETRY_CONFIG)
    def invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = self._prepare_messages(system_prompt, user_prompt)

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "stream"}
        extra_params = {key: value for key, value in kwargs.items() if key in allowed_keys and value is not None}

        timeout = kwargs.pop("timeout", self.timeout)

        if self._cli_invoker:
            prompt = self._render_messages(messages)
            try:
                response_text = self._cli_invoker.run(prompt, model_name=self.model_name, timeout=timeout)
            except CodexCLIError as exc:
                logger.error(f"Codex CLI 调用失败: {exc}")
                raise
            return self.validate_response(response_text)

        response = self.client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model_name,
            messages=messages,
            timeout=timeout,
            **extra_params,
        )

        if response.choices and response.choices[0].message:
            return self.validate_response(response.choices[0].message.content)
        return ""

    def stream_invoke(self, system_prompt: str, user_prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式调用LLM，逐步返回响应内容"""

        messages = self._prepare_messages(system_prompt, user_prompt)

        allowed_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
        extra_params = {key: value for key, value in kwargs.items() if key in allowed_keys and value is not None}
        extra_params["stream"] = True

        timeout = kwargs.pop("timeout", self.timeout)

        if self._cli_invoker:
            prompt = self._render_messages(messages)
            try:
                for chunk in self._cli_invoker.stream(prompt, model_name=self.model_name, timeout=timeout):
                    if chunk:
                        yield chunk
            except CodexCLIError as exc:
                logger.error(f"Codex CLI 流式请求失败: {exc}")
                raise
            return

        try:
            stream = self.client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model_name,
                messages=messages,
                timeout=timeout,
                **extra_params,
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"流式请求失败: {str(e)}")
            raise e
    
    @with_retry(LLM_RETRY_CONFIG)
    def stream_invoke_to_string(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        流式调用LLM并安全地拼接为完整字符串（避免UTF-8多字节字符截断）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 额外参数（temperature, top_p等）
            
        Returns:
            完整的响应字符串
        """
        # 以字节形式收集所有块
        byte_chunks = []
        for chunk in self.stream_invoke(system_prompt, user_prompt, **kwargs):
            byte_chunks.append(chunk.encode('utf-8'))
        
        # 拼接所有字节，然后一次性解码
        if byte_chunks:
            return b''.join(byte_chunks).decode('utf-8', errors='replace')
        return ""

    @staticmethod
    def validate_response(response: Optional[str]) -> str:
        if response is None:
            return ""
        return response.strip()

    @staticmethod
    def _prepare_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _render_messages(messages: List[Dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts).strip()

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model_name,
        }
        if self._cli_invoker:
            info["cli_command"] = self.cli_command
        else:
            info["api_base"] = self.base_url or "default"
        return info
