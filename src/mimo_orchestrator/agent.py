"""Core Agent — the primary unit of work in MiMo Orchestrator."""

import re
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mimo_orchestrator.client import MiMoClient, Message
from mimo_orchestrator.tool_registry import ToolRegistry
from mimo_orchestrator.skill_manager import SkillManager

console = Console()


@dataclass
class AgentResult:
    """The outcome of an agent run."""
    success: bool
    content: str
    iterations: int
    tool_calls_made: int
    total_tokens: int
    duration_ms: float
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "iterations": self.iterations,
            "tool_calls_made": self.tool_calls_made,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class Agent:
    """
    A single autonomous agent powered by Xiaomi MiMo.

    An agent:
    - Has a system prompt (persona + capabilities)
    - Maintains conversation history
    - Can call tools (via the ToolRegistry)
    - Can load and execute skills
    - Runs in a loop until it produces a final answer or hits max iterations
    """

    SYSTEM_PROMPT = (
        "You are a helpful, precise AI coding assistant powered by Xiaomi MiMo. "
        "You have access to a set of tools you can call to accomplish tasks. "
        "Think step by step. When you need to use a tool, call it with the proper JSON arguments. "
        "When you're done, respond with your final answer clearly formatted."
    )

    def __init__(
        self,
        name: str = "agent",
        persona: str = "",
        mimo_client: Optional[MiMoClient] = None,
        max_iterations: int = 30,
        timeout: int = 120,
        temperature: float = 0.7,
        skill_manager: Optional[SkillManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verbose: bool = False,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.persona = persona or self.SYSTEM_PROMPT
        self.mimo = mimo_client or MiMoClient()
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.temperature = temperature
        self.skill_manager = skill_manager or SkillManager()
        self.tool_registry = tool_registry or ToolRegistry()
        self.verbose = verbose

        self._history: list[Message] = []
        self._tool_results: dict[str, str] = {}  # tool_call_id -> result

    # ────────────────────────────────────────────────────────────────
    # Main run loop
    # ────────────────────────────────────────────────────────────────

    def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        """
        Run the agent on a single task until completion.

        Args:
            task:     The user's request / task description
            context:  Optional extra context (files, data, etc.)

        Returns:
            AgentResult with outcome, stats, and any errors
        """
        t0 = time.monotonic()
        total_tokens = 0
        tool_calls_made = 0
        iterations = 0
        error_msg: Optional[str] = None

        # Build system message with skill context
        system_msg = self._build_system_message(context)

        # Seed conversation
        self._history = [Message(role="system", content=system_msg)]
        if context:
            ctx_str = self._format_context(context)
            self._history.append(Message(role="user", content=f"[Context]\n{ctx_str}\n\n[Task]\n{task}"))
        else:
            self._history.append(Message(role="user", content=task))

        if self.verbose:
            console.print(f"\n[bold cyan]Agent {self.name}[/] starting on: {task[:80]}...")

        try:
            for iteration in range(self.max_iterations):
                iterations = iteration + 1

                # Call MiMo
                resp = self.mimo.chat(
                    messages=self._history,
                    temperature=self.temperature,
                    tools=self.tool_registry.get_tool_schemas(),
                )

                total_tokens += resp.usage.total_tokens

                # Append assistant's response to history
                asst_msg = Message(role="assistant", content=resp.content)
                if resp.tool_calls:
                    asst_msg.tool_calls = resp.tool_calls
                self._history.append(asst_msg)

                if self.verbose:
                    console.print(f"[dim]  iteration {iterations} | tokens: {resp.usage.total_tokens} | "
                                 f"latency: {resp.latency_ms:.0f}ms")

                # No tool calls → final answer
                if not resp.tool_calls:
                    if self.verbose:
                        console.print(f"[green]Agent {self.name} done (no tools needed)")
                    return AgentResult(
                        success=True,
                        content=resp.content,
                        iterations=iterations,
                        tool_calls_made=tool_calls_made,
                        total_tokens=total_tokens,
                        duration_ms=(time.monotonic() - t0) * 1000,
                    )

                # Execute tool calls
                for tc in resp.tool_calls:
                    tool_calls_made += 1
                    result = self._execute_tool_call(tc)
                    self._history.append(result)

                    if self.verbose:
                        fn_name = tc.get("function", {}).get("name", "?")
                        console.print(f"  [yellow]→ tool:[/yellow] {fn_name}")

            # Max iterations hit
            error_msg = f"Max iterations ({self.max_iterations}) reached without final answer."
            if self.verbose:
                console.print(f"[red]Agent {self.name}: {error_msg}[/red]")

        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                console.print(f"[red]Agent {self.name} error: {e}[/red]")

        return AgentResult(
            success=False,
            content=self._history[-1].content if self._history else "",
            iterations=iterations,
            tool_calls_made=tool_calls_made,
            total_tokens=total_tokens,
            duration_ms=(time.monotonic() - t0) * 1000,
            error=error_msg,
        )

    # ────────────────────────────────────────────────────────────────
    # Tool execution
    # ────────────────────────────────────────────────────────────────

    def _execute_tool_call(self, tc: dict) -> Message:
        """Execute a single tool call and return the result message."""
        func = tc.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        # Parse arguments (handle both str and dict)
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"raw": raw_args}
        else:
            args = raw_args

        tool_call_id = tc.get("id", str(uuid.uuid4()))

        # Look up and execute the tool
        handler = self.tool_registry.get_handler(name)
        if handler:
            try:
                result = handler(**args)
                result_str = self._serialize_result(result)
            except Exception as e:
                result_str = f"Error: {e}"
        else:
            # No handler registered — simulate a reasonable response
            result_str = self._fallback_tool(name, args)

        self._tool_results[tool_call_id] = result_str
        return Message(
            role="tool",
            content=result_str,
            tool_call_id=tool_call_id,
            name=name,
        )

    def _serialize_result(self, result: Any) -> str:
        """Convert tool result to a string for the message history."""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
            return repr(result)

    def _fallback_tool(self, name: str, args: dict) -> str:
        """Fallback for unregistered tools — returns a helpful message."""
        return (
            f"Tool '{name}' is not registered. "
            f"Registered tools: {', '.join(self.tool_registry.list_tools())}. "
            f"Please implement it or register it via tool_registry.register()."
        )

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _build_system_message(self, context: Optional[dict]) -> str:
        """Build the system prompt, including skill capabilities."""
        skills = self.skill_manager.list_skills()
        skill_section = ""
        if skills:
            skill_lines = "\n".join(f"  - {s}" for s in skills)
            skill_section = f"\n\nAvailable skills:\n{skill_lines}"

        tools = self.tool_registry.list_tools()
        tool_section = ""
        if tools:
            tool_lines = "\n".join(f"  - {t}" for t in tools)
            tool_section = f"\n\nAvailable tools:\n{tool_lines}"

        return (
            f"{self.persona}\n\n"
            f"You are operating as agent [{self.name}] (id={self.id})."
            f"{skill_section}{tool_section}"
        )

    def _format_context(self, context: dict) -> str:
        """Format context dict into a readable string."""
        parts = []
        for k, v in context.items():
            if isinstance(v, dict):
                parts.append(f"  {k}:\n" + json.dumps(v, ensure_ascii=False, indent=4))
            elif isinstance(v, list):
                parts.append(f"  {k}: {', '.join(str(x) for x in v)}")
            else:
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    def reset(self) -> None:
        """Clear conversation history and tool results."""
        self._history.clear()
        self._tool_results.clear()

    def history(self) -> list[Message]:
        """Return a copy of the conversation history."""
        return list(self._history)

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, id={self.id}, iterations={self.max_iterations})"
