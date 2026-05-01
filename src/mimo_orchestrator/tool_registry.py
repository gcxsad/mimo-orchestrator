"""Tool registry — register, discover, and call tools for agents."""

import json
import inspect
import re
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    """A callable tool with metadata and schema."""
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    category: str = "general"

    def to_openai_schema(self) -> dict:
        """Return an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or self._infer_parameters(),
            },
        }

    def _infer_parameters(self) -> dict:
        """Auto-generate JSON Schema from function signature."""
        try:
            sig = inspect.signature(self.handler)
            required = []
            properties = {}
            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue
                param_type = "string"
                default = param.default
                if default is inspect.Parameter.empty:
                    required.append(name)
                    type_map = {
                        int: "integer",
                        float: "number",
                        bool: "boolean",
                        list: "array",
                        dict: "object",
                    }
                    param_type = type_map.get(param.annotation, "string")
                else:
                    param_type = "string"
                properties[name] = {"type": param_type, "description": f"Parameter: {name}"}
            return {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        except Exception:
            return {"type": "object", "properties": {}, "required": []}


class ToolRegistry:
    """
    Central registry for all tools an agent can call.

    Tools are registered with a name, description, and handler function.
    The registry generates OpenAI-compatible tool schemas for LLM tool-calling.

    Usage:
        registry = ToolRegistry()
        registry.register("read_file", read_file_tool, description="Read a file")
        schemas = registry.get_tool_schemas()
    """

    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        # Register built-in tools on init
        self._register_builtins()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        parameters: Optional[dict] = None,
        category: str = "general",
        examples: Optional[list[str]] = None,
    ) -> "ToolRegistry":
        """
        Register a tool.

        Args:
            name:        Unique tool name (snake_case recommended)
            handler:     The callable that executes the tool
            description: Human-readable description for the LLM
            parameters:  OpenAI-style JSON Schema for parameters
            category:   Grouping category (e.g., "file", "web", "code")
            examples:   Example usage strings

        Returns:
            self (for chaining)
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered. Use replace() to update.")

        tool = Tool(
            name=name,
            description=description or f"Tool: {name}",
            handler=handler,
            parameters=parameters or {},
            examples=examples or [],
            category=category,
        )
        self._tools[name] = tool
        return self

    def replace(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        parameters: Optional[dict] = None,
        category: str = "general",
    ) -> "ToolRegistry":
        """Replace an existing tool or register if not found."""
        self._tools.pop(name, None)
        return self.register(name, handler, description, parameters, category)

    def unregister(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def get_handler(self, name: str) -> Optional[Callable[..., Any]]:
        """Get the handler function for a tool, or None if not found."""
        tool = self._tools.get(name)
        return tool.handler if tool else None

    def get_schema(self, name: str) -> Optional[dict]:
        """Get the OpenAI tool schema for a specific tool."""
        tool = self._tools.get(name)
        return tool.to_openai_schema() if tool else None

    def get_tool_schemas(self) -> list[dict]:
        """Get all registered tools as OpenAI-compatible tool schemas."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List tool names in a specific category."""
        return sorted(
            t.name for t in self._tools.values() if t.category == category
        )

    def get_info(self, name: str) -> Optional[dict]:
        """Get full info about a tool (for help/debug)."""
        tool = self._tools.get(name)
        if not tool:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "parameters": tool.to_openai_schema()["function"]["parameters"],
            "examples": tool.examples,
        }

    # ────────────────────────────────────────────────────────────────
    # Built-in tools
    # ────────────────────────────────────────────────────────────────

    def _register_builtins(self) -> None:
        """Register the standard built-in tools."""
        self.register(
            "shell",
            _shell_tool,
            description=(
                "Execute a shell command and return stdout+stderr. "
                "Use for file operations, git, npm, pip, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max seconds (default: 30)"},
                },
                "required": ["command"],
            },
            category="system",
        )

        self.register(
            "search_web",
            _search_web_tool,
            description="Search the web for information. Returns titles, URLs, and snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
            category="web",
        )

        self.register(
            "read_file",
            _read_file_tool,
            description="Read the contents of a file. Use for examining code, configs, logs.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "Start line (1-indexed, default: 1)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default: 200)"},
                },
                "required": ["path"],
            },
            category="file",
        )

        self.register(
            "search_code",
            _search_code_tool,
            description="Search for patterns inside files using regex. Fast, multi-file grep.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search"},
                    "path": {"type": "string", "description": "Directory to search in (default: .)"},
                    "file_glob": {"type": "string", "description": "File glob filter, e.g. *.py"},
                    "output_mode": {"type": "string", "description": "content|files_only|count"},
                },
                "required": ["pattern"],
            },
            category="file",
        )

        self.register(
            "write_file",
            _write_file_tool,
            description="Write content to a file (overwrites existing). Creates parent dirs automatically.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            category="file",
        )

        self.register(
            "list_directory",
            _list_dir_tool,
            description="List files in a directory. Shows names, sizes, and modification times.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: .)"},
                    "include_hidden": {"type": "boolean", "description": "Include hidden files"},
                },
            },
            category="file",
        )

        self.register(
            "make_directory",
            _mkdir_tool,
            description="Create a directory (and parents if needed).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create"},
                },
                "required": ["path"],
            },
            category="file",
        )


# ────────────────────────────────────────────────────────────────────
# Built-in tool implementations
# ────────────────────────────────────────────────────────────────────

def _shell_tool(command: str, timeout: int = 30) -> str:
    import subprocess
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = f"[exit {result.returncode}]\n"
        if result.stdout:
            out += f"[stdout]\n{result.stdout}"
        if result.stderr:
            out += f"\n[stderr]\n{result.stderr}"
        return out.strip()
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


def _search_web_tool(query: str, num_results: int = 5) -> str:
    import subprocess
    try:
        result = subprocess.run(
            f"ddg '{query}' -n {num_results}",
            shell=True, capture_output=True, text=True, timeout=15
        )
        return result.stdout or "[no results]"
    except Exception as e:
        return f"[error] {e}"


def _read_file_tool(path: str, offset: int = 1, limit: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(0, offset - 1)
        end = min(start + limit, total)
        content = "".join(lines[start:end])
        return f"[{path}] ({total} total lines, showing {start+1}-{end})\n{content}"
    except Exception as e:
        return f"[error reading {path}]: {e}"


def _write_file_tool(path: str, content: str) -> str:
    import os
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[ok] Written to {path}"
    except Exception as e:
        return f"[error writing {path}]: {e}"


def _search_code_tool(
    pattern: str,
    path: str = ".",
    file_glob: str = "",
    output_mode: str = "content",
) -> str:
    import subprocess
    try:
        cmd = ["grep", "-n", "-r", pattern, path]
        if file_glob:
            cmd.insert(1, "--include=" + file_glob)
        if output_mode == "files_only":
            cmd[2] = "-l"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout or "[no matches]"
    except Exception as e:
        return f"[error]: {e}"


def _list_dir_tool(path: str = ".", include_hidden: bool = False) -> str:
    import os
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
        lines = []
        for e in entries:
            if not include_hidden and e.name.startswith("."):
                continue
            size = e.stat().st_size if e.is_file() else 0
            mtime = e.stat().st_mtime
            lines.append(f"  {e.name}  {'dir' if e.is_dir() else size}")
        return "\n".join(lines) or "[empty]"
    except Exception as e:
        return f"[error]: {e}"


def _mkdir_tool(path: str) -> str:
    import os
    try:
        os.makedirs(path, exist_ok=True)
        return f"[ok] Created {path}"
    except Exception as e:
        return f"[error]: {e}"
