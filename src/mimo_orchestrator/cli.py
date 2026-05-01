"""CLI entry points for MiMo Orchestrator."""

import os
import sys
import json
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from mimo_orchestrator import __version__
from mimo_orchestrator.client import MiMoClient
from mimo_orchestrator.agent import Agent
from mimo_orchestrator.orchestrator import Orchestrator, SubTask
from mimo_orchestrator.skill_manager import SkillManager
from mimo_orchestrator.tool_registry import ToolRegistry

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Global context (shared state for subcommands)
# ─────────────────────────────────────────────────────────────────────────────

def get_mimo() -> MiMoClient:
    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        console.print("[red]Error: MIMO_API_KEY not set.[/]")
        console.print("  Set it with: export MIMO_API_KEY=your_key")
        console.print("  Get a key at: https://platform.xiaomimimo.com/")
        sys.exit(1)
    return MiMoClient(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version=__version__)
def cli():
    """
    MiMo Orchestrator — Multi-agent workflow engine powered by Xiaomi MiMo.

    Get started:
      mimo agent "build a REST API with authentication"
      mimo parallel --task "research" --task "implement"

    Set your API key:
      export MIMO_API_KEY=your_key_here
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# mimo agent — single agent run
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("agent")
@click.argument("task", type=str)
@click.option("--model", default=None, help="Override MiMo model")
@click.option("--temperature", default=0.7, help="Sampling temperature (0.0–2.0)")
@click.option("--max-iter", default=30, help="Max agent iterations")
@click.option("--verbose", is_flag=True, help="Show iteration details")
def agent_cli(task: str, model: str | None, temperature: float, max_iter: int, verbose: bool):
    """Run a single MiMo-powered agent on a task."""
    mimo = get_mimo()
    if model:
        mimo.model = model

    skill_mgr = SkillManager()
    skill_mgr.load_all()

    tool_reg = ToolRegistry()

    agent = Agent(
        persona=(
            "You are a world-class AI coding assistant. "
            "You think carefully, write clean code, and explain your reasoning."
        ),
        mimo_client=mimo,
        skill_manager=skill_mgr,
        tool_registry=tool_reg,
        max_iterations=max_iter,
        verbose=verbose,
    )

    console.print(f"\n[cyan]Running agent on:[/] {task[:100]}{'...' if len(task) > 100 else ''}")
    console.print(f"[dim]Model: {mimo.model} | Max iter: {max_iter} | Temp: {temperature}[/]\n")

    result = agent.run(task)

    if result.success:
        console.print(Panel(
            Markdown(result.content),
            title=f"[green]Result[/green] ({result.iterations} iter, "
                  f"{result.total_tokens} tokens, {result.duration_ms:.0f}ms)",
            border_style="green",
        ))
    else:
        console.print(Panel(
            result.content or result.error or "No output",
            title=f"[red]Error[/red] — {result.error}",
            border_style="red",
        ))
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# mimo parallel — parallel multi-agent workflow
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("parallel")
@click.option("--task", "-t", multiple=True, help="Task description (repeat for multiple)")
@click.option("--workers", default=3, help="Max parallel agents")
@click.option("--verbose", is_flag=True, help="Show iteration details")
def parallel_cli(task: tuple[str, ...], workers: int, verbose: bool):
    """Run multiple tasks in parallel using the Orchestrator."""
    if not task:
        console.print("[yellow]No tasks provided. Use: mimo parallel -t 'task 1' -t 'task 2'[/]")
        sys.exit(0)

    mimo = get_mimo()
    orch = Orchestrator(mimo=mimo, max_workers=workers, verbose=verbose)

    tasks = [
        SubTask(id=f"task-{i}", description=desc, assigned_agent="worker")
        for i, desc in enumerate(task)
    ]

    console.print(f"[cyan]Running {len(tasks)} tasks in parallel (max {workers} workers)[/]\n")

    result = orch.run_parallel(tasks)
    orch.print_summary(result)

    sys.exit(0 if result.success else 1)


# ─────────────────────────────────────────────────────────────────────────────
# mimo skill — skill management
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("skill")
@click.argument("action", type=click.Choice(["list", "info", "load"]))
@click.argument("name", required=False)
@click.option("--dir", "skills_dir", default="skills", help="Skills directory")
def skill_cli(action: str, name: str | None, skills_dir: str):
    """Manage and inspect skills."""
    sm = SkillManager(skills_dir=skills_dir)
    sm.load_all()

    if action == "list":
        skills = sm.list_skills()
        if not skills:
            console.print("[yellow]No skills found in", skills_dir)
        else:
            console.print(f"[green]Skills ({len(skills)}):[/]")
            for s in skills:
                info = sm.skill_info(s)
                desc = info.get("description", "")[:60] if info else ""
                triggers = info.get("trigger_keywords", [])[:3] if info else []
                console.print(f"  [cyan]{s}[/]")
                if desc:
                    console.print(f"    {desc}")
                if triggers:
                    console.print(f"    triggers: {', '.join(triggers)}")

    elif action == "info":
        if not name:
            console.print("[red]Provide a skill name: mimo skill info <name>[/]")
            sys.exit(1)
        info = sm.skill_info(name)
        if not info:
            console.print(f"[red]Skill '{name}' not found[/]")
            sys.exit(1)
        console.print_json(json.dumps(info, indent=2, ensure_ascii=False))

    elif action == "load":
        count = len(sm.list_skills())
        console.print(f"[green]Loaded {count} skills from {skills_dir}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# mimo tools — list registered tools
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("tools")
def tools_cli():
    """List all registered tools available to agents."""
    reg = ToolRegistry()
    tools = reg.list_tools()
    console.print(f"[green]Registered tools ({len(tools)}):[/]")
    for t in tools:
        info = reg.get_info(t)
        desc = (info.get("description", "")[:70] + "...") if info else ""
        cat = info.get("category", "?") if info else "?"
        console.print(f"  [cyan]{t}[/] [dim]({cat})[/]")
        if desc:
            console.print(f"    {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# mimo shell — interactive shell
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("shell")
@click.option("--model", default=None)
@click.option("--verbose", is_flag=True)
def shell_cli(model: str | None, verbose: bool):
    """Start an interactive chat shell with a MiMo agent."""
    mimo = get_mimo()
    if model:
        mimo.model = model

    skill_mgr = SkillManager()
    skill_mgr.load_all()

    agent = Agent(
        mimo_client=mimo,
        skill_manager=skill_mgr,
        tool_registry=ToolRegistry(),
        verbose=verbose,
    )

    console.print(Panel(
        f"[bold cyan]MiMo Orchestrator Shell[/] | Model: {mimo.model} | "
        f"Type [bold]exit[/] or [bold]quit[/] to end session",
        border_style="cyan",
    ))
    console.print()

    while True:
        try:
            user_input = console.input("[bold cyan]>[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye[/]")
            break

        if user_input.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye[/]")
            break

        if not user_input.strip():
            continue

        result = agent.run(user_input)
        if result.success:
            console.print(Panel(
                Markdown(result.content),
                border_style="green",
            ))
        else:
            console.print(f"[red]Error:[/] {result.error}")
        console.print()


if __name__ == "__main__":
    cli()
