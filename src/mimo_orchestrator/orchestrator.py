"""Orchestrator — coordinate multiple agents working in parallel on complex tasks."""

import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional
from rich.console import Console
from rich.table import Table

from mimo_orchestrator.client import MiMoClient
from mimo_orchestrator.agent import Agent, AgentResult
from mimo_orchestrator.skill_manager import SkillManager
from mimo_orchestrator.tool_registry import ToolRegistry

console = Console()


@dataclass
class SubTask:
    """A unit of work assigned to one agent in a parallel workflow."""
    id: str
    description: str
    assigned_agent: str = "worker"
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)  # task IDs this depends on
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of a full parallel workflow run."""
    success: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_tokens: int
    duration_ms: float
    task_results: dict[str, dict]
    errors: list[str]

    def summary_table(self) -> Table:
        """Render a Rich table of results."""
        table = Table(title="Workflow Results", show_header=True, header_style="bold magenta")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status", style="")
        table.add_column("Iterations", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Error", style="red")

        for tid, result in self.task_results.items():
            status = "[green]OK[/green]" if result.get("success") else "[red]FAIL[/red]"
            err = result.get("error", "")[:40]
            table.add_row(
                tid,
                status,
                str(result.get("iterations", "-")),
                str(result.get("total_tokens", "-")),
                f"{result.get('duration_ms', 0):.0f}ms",
                err,
            )
        return table


class Orchestrator:
    """
    Coordinates multiple MiMo-powered agents to tackle complex, multi-part tasks.

    Key capabilities:
    - Parallel task execution (bounded concurrency)
    - Sequential dependency chains
    - Skill auto-detection per task
    - Shared tool registry and memory across agents
    - Comprehensive workflow result reporting

    Example:
        orch = Orchestrator(max_workers=3)
        result = orch.run_parallel([
            SubTask(id="a", description="Research MiMo API pricing"),
            SubTask(id="b", description="Write a sample Python script", dependencies=["a"]),
        ])
        orch.print_summary(result)
    """

    def __init__(
        self,
        mimo: Optional[MiMoClient] = None,
        max_workers: int = 3,
        skill_manager: Optional[SkillManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        default_timeout: int = 120,
        verbose: bool = False,
    ):
        self.mimo = mimo or MiMoClient()
        self.max_workers = max_workers
        self.skill_manager = skill_manager or SkillManager()
        self.tool_registry = tool_registry or ToolRegistry()
        self.default_timeout = default_timeout
        self.verbose = verbose

        # Shared context propagated across all agents
        self.shared_context: dict[str, Any] = {}

        # Agent factory
        self._agent_counter = 0

    def _make_agent(self, name: str) -> Agent:
        """Create a new agent with shared resources."""
        self._agent_counter += 1
        agent = Agent(
            name=f"{name}-{self._agent_counter}",
            mimo_client=self.mimo,
            skill_manager=self.skill_manager,
            tool_registry=self.tool_registry,
            max_iterations=30,
            timeout=self.default_timeout,
            verbose=self.verbose,
        )
        return agent

    # ────────────────────────────────────────────────────────────────
    # Core parallel execution
    # ────────────────────────────────────────────────────────────────

    def run_parallel(self, tasks: list[SubTask]) -> WorkflowResult:
        """
        Execute tasks in parallel, respecting dependencies.

        Tasks with no dependencies run immediately.
        Tasks with dependencies wait for all their dependencies to complete,
        then run with the results of those dependencies injected as context.

        Args:
            tasks: List of SubTask objects describing the work

        Returns:
            WorkflowResult with per-task results and overall stats
        """
        t0 = time.monotonic()
        total_tokens = 0
        completed = 0
        failed = 0
        task_results: dict[str, dict] = {}
        errors: list[str] = []
        done: set[str] = set()

        # Phase 1: tasks with no dependencies — run in parallel
        initial = [t for t in tasks if not t.dependencies]
        remaining = [t for t in tasks if t.dependencies]

        def run_task(task: SubTask, context: Optional[dict] = None) -> tuple[str, dict]:
            agent = self._make_agent(task.assigned_agent)
            result = agent.run(task.description, context=context)
            return task.id, result.to_dict()

        # Run initial batch
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(run_task, t): t for t in initial}
            for future in as_completed(futures):
                tid, result = future.result()
                task_results[tid] = result
                if result["success"]:
                    completed += 1
                    done.add(tid)
                    # Inject result into shared context
                    self.shared_context[tid] = result.get("content", "")
                else:
                    failed += 1
                    errors.append(f"{tid}: {result.get('error', 'unknown')}")
                total_tokens += result.get("total_tokens", 0)

        # Phase 2: remaining tasks — resolve dependencies then run
        while remaining:
            ready = [t for t in remaining if all(d in done for d in t.dependencies)]
            if not ready:
                # Circular or broken dependencies
                for t in remaining:
                    errors.append(f"{t.id}: unresolved dependencies {t.dependencies}")
                    task_results[t.id] = {"success": False, "error": "unresolved dependencies"}
                    failed += 1
                    remaining.remove(t)
                break

            # Build context for each ready task
            contexts = {}
            for t in ready:
                ctx = dict(self.shared_context)
                for dep_id in t.dependencies:
                    ctx[f"dep_{dep_id}"] = task_results.get(dep_id, {}).get("content", "")
                contexts[t.id] = ctx

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(run_task, t, contexts[t.id]): t for t in ready}
                for future in as_completed(futures):
                    tid, result = future.result()
                    task_results[tid] = result
                    if result["success"]:
                        completed += 1
                        done.add(tid)
                        self.shared_context[tid] = result.get("content", "")
                    else:
                        failed += 1
                        errors.append(f"{tid}: {result.get('error', 'unknown')}")
                    total_tokens += result.get("total_tokens", 0)
                remaining = [t for t in remaining if t.id not in done]

        duration = (time.monotonic() - t0) * 1000
        return WorkflowResult(
            success=failed == 0,
            total_tasks=len(tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            total_tokens=total_tokens,
            duration_ms=duration,
            task_results=task_results,
            errors=errors,
        )

    def run_single(self, task: str, context: Optional[dict] = None) -> AgentResult:
        """Convenience: run a single task (same as creating an Agent and calling run)."""
        agent = self._make_agent("orchestrator")
        return agent.run(task, context=context)

    def print_summary(self, result: WorkflowResult) -> None:
        """Pretty-print a workflow result using Rich."""
        console.print()
        if result.success:
            console.print(f"[bold green]Workflow completed successfully[/] "
                        f"({result.completed_tasks}/{result.total_tasks} tasks)")
        else:
            console.print(f"[bold red]Workflow finished with {result.failed_tasks} failure(s)[/]")

        console.print(f"  Total tokens: {result.total_tokens:,} | Duration: {result.duration_ms:.0f}ms")
        console.print()
        console.print(result.summary_table())

        if result.errors:
            console.print("\n[bold red]Errors:[/]")
            for e in result.errors:
                console.print(f"  - {e}")
        console.print()

    def __repr__(self) -> str:
        return (f"Orchestrator(max_workers={self.max_workers}, "
                f"tools={len(self.tool_registry.list_tools())}, "
                f"skills={len(self.skill_manager.list_skills())})")
