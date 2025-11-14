"""Utility helpers for invoking local CLI-based LLM providers.

These helpers are primarily designed for Codex or other OpenAI-compatible
command line clients that authenticate via an interactive login instead of an
API key.  The CLI command is expected to read the prompt from STDIN and write
its response to STDOUT.  The command template may include ``{model}`` which
will be substituted with the configured model name before execution.
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

from loguru import logger


class CodexCLIError(RuntimeError):
    """Raised when invoking the Codex CLI fails."""


@dataclass
class CodexCLIInvoker:
    """Simple wrapper around a Codex-compatible CLI command."""

    command_template: str
    default_timeout: float

    def _build_args(self, model_name: Optional[str]) -> Iterable[str]:
        formatted = self.command_template.format(model=model_name or "")
        return shlex.split(formatted)

    def run(self, prompt: str, *, model_name: Optional[str] = None, timeout: Optional[float] = None) -> str:
        args = list(self._build_args(model_name))
        logger.debug(f"Running Codex CLI command: {' '.join(args)}")
        try:
            completed = subprocess.run(
                args,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout or self.default_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CodexCLIError(f"Codex CLI command not found: {args[0] if args else self.command_template}") from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexCLIError(f"Codex CLI command timed out after {timeout or self.default_timeout} seconds") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise CodexCLIError(
                f"Codex CLI command failed with exit code {completed.returncode}: {stderr}"
            )

        return (completed.stdout or "").strip()

    def stream(
        self,
        prompt: str,
        *,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Generator[str, None, None]:
        output = self.run(prompt, model_name=model_name, timeout=timeout)
        if not output:
            return
        for chunk in output.splitlines(keepends=True):
            yield chunk
        if output and not output.endswith("\n"):
            # Ensure the consumer receives the trailing content even if no newline.
            yield ""
