#!/usr/bin/env python3
"""
Quick-start examples for MiMo Orchestrator.
Run: python examples/quickstart.py
"""

import os
from mimo_orchestrator import (
    MiMoClient, Agent, Orchestrator, SubTask, ToolRegistry, SkillManager
)


def example_single_agent():
    """Run a single agent on a coding task."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Agent")
    print("=" * 60)

    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("Set MIMO_API_KEY first!")
        return

    mimo = MiMoClient(api_key=api_key)
    agent = Agent(mimo_client=mimo, tool_registry=ToolRegistry(), verbose=True)

    result = agent.run(
        "Write a Python function that finds all prime numbers up to N, "
        "using the Sieve of Eratosthenes algorithm. Include type hints and a docstring."
    )

    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print("\n--- Agent Output ---")
    print(result.content[:1000])


def example_parallel():
    """Run a 3-stage parallel workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Parallel Multi-Agent Workflow")
    print("=" * 60)

    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("Set MIMO_API_KEY first!")
        return

    mimo = MiMoClient(api_key=api_key)
    orch = Orchestrator(mimo=mimo, max_workers=3, verbose=True)

    tasks = [
        SubTask(
            id="plan",
            description="List 5 creative ideas for a CLI productivity tool in Python. Be specific about features.",
        ),
        SubTask(
            id="scaffold",
            description="Scaffold a Python CLI project: pyproject.toml, src layout, .gitignore, tests/. Use best practices.",
            dependencies=["plan"],
        ),
        SubTask(
            id="implement",
            description="Write a working Python CLI tool using Click that manages a to-do list stored in JSON. Include add, list, done, remove commands.",
            dependencies=["scaffold"],
        ),
    ]

    result = orch.run_parallel(tasks)
    orch.print_summary(result)


def example_custom_tools():
    """Register and use a custom tool."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Tools")
    print("=" * 60)

    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("Set MIMO_API_KEY first!")
        return

    registry = ToolRegistry()

    def calculator(expression: str) -> str:
        """Evaluate a math expression safely."""
        import ast, operator
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        try:
            node = ast.parse(expression, mode="eval")
            return str(_eval_node(node.body, ops))
        except Exception as e:
            return f"Error: {e}"

    def _eval_node(node, ops):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval_node(node.left, ops), _eval_node(node.right, ops))
        return 0

    registry.register(
        name="calculate",
        handler=calculator,
        description="Safely evaluate a mathematical expression (supports +, -, *, /, **).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2**10 + 15'"}
            },
            "required": ["expression"],
        },
        category="utility",
    )

    mimo = MiMoClient(api_key=api_key)
    agent = Agent(mimo_client=mimo, tool_registry=registry, max_iterations=5)

    result = agent.run("Calculate 2^20 + 1000 using the calculate tool")

    print(f"Success: {result.success}")
    print(f"Tool calls: {result.tool_calls_made}")
    print("\n--- Output ---")
    print(result.content)


if __name__ == "__main__":
    print("MiMo Orchestrator — Quick Start Examples")
    print("Make sure to: export MIMO_API_KEY=your_key_here")

    example_single_agent()
    # Uncomment to run more examples:
    # example_parallel()
    # example_custom_tools()
