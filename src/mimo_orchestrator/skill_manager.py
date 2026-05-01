"""Skill manager — load, inspect, and execute modular skill packs."""

import os
import re
import json
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
from mimo_orchestrator.client import MiMoClient, Message


@dataclass
class Skill:
    """A reusable skill pack: trigger, steps, and optional scripts."""
    name: str
    description: str
    trigger_keywords: list[str] = field(default_factory=list)
    system_prompt_addition: str = ""
    tools: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Optional Python module with custom logic
    module_path: Optional[str] = None

    def matches(self, text: str) -> bool:
        """Check if a task description matches this skill's triggers."""
        t = text.lower()
        return any(kw.lower() in t for kw in self.trigger_keywords)


class SkillManager:
    """
    Manages a collection of skills — reusable agent behaviour packs.

    Each skill has:
    - trigger keywords (for auto-detection)
    - system prompt additions (extra context/instructions)
    - tool requirements (which tools must be available)
    - optional Python module for advanced logic

    Skills are stored as YAML/JSON files in a skills/ directory,
    or registered programmatically.

    Usage:
        sm = SkillManager(skills_dir="./skills")
        sm.load_all()
        relevant = sm.find_relevant("build a REST API with auth")
        # relevant = [Skill(name="web-server", ...)]
    """

    DEFAULT_SKILL_DIR = os.path.join(os.getcwd(), "skills")

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir or self.DEFAULT_SKILL_DIR)
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    # ────────────────────────────────────────────────────────────────
    # Loading
    # ────────────────────────────────────────────────────────────────

    def load_all(self) -> "SkillManager":
        """Discover and load all skills from the skills directory."""
        if not self.skills_dir.exists():
            return self

        for path in self.skills_dir.rglob("skill.yaml"):
            self.load_skill_file(path)
        for path in self.skills_dir.rglob("skill.yml"):
            self.load_skill_file(path)
        for path in self.skills_dir.rglob("skill.json"):
            self.load_skill_json(path)

        self._loaded = True
        return self

    def load_skill_file(self, path: Path) -> Optional[Skill]:
        """Load a single YAML skill file."""
        try:
            import yaml  # type: ignore
        except ImportError:
            # Fallback: simple regex-based YAML parsing
            return self._load_skill_yaml_simple(path)

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return self._register_from_dict(data)
        except Exception as e:
            print(f"[SkillManager] Failed to load {path}: {e}")
            return None

    def load_skill_json(self, path: Path) -> Optional[Skill]:
        """Load a single JSON skill file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._register_from_dict(data)
        except Exception as e:
            print(f"[SkillManager] Failed to load {path}: {e}")
            return None

    def _load_skill_yaml_simple(self, path: Path) -> Optional[Skill]:
        """Minimal YAML loader without the pyyaml dep — for key: value pairs."""
        try:
            text = path.read_text(encoding="utf-8")
            data: dict[str, Any] = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ": " in line:
                    key, val = line.split(": ", 1)
                    data[key.strip()] = val.strip().strip('"').strip("'")
                elif line.endswith(":"):
                    data[line[:-1].strip()] = True
            return self._register_from_dict(data)
        except Exception:
            return None

    def _register_from_dict(self, data: dict) -> Optional[Skill]:
        """Register a skill from a dict."""
        name = data.get("name") or data.get("skill", "")
        if not name:
            return None

        skill = Skill(
            name=str(name),
            description=str(data.get("description", "")),
            trigger_keywords=data.get("trigger_keywords", [])
                        if isinstance(data.get("trigger_keywords"), list)
                        else [],
            system_prompt_addition=str(data.get("system_prompt_addition", "")),
            tools=data.get("tools", []) if isinstance(data.get("tools"), list) else [],
            examples=data.get("examples", []) if isinstance(data.get("examples"), list) else [],
            metadata=data.get("metadata", {}),
            module_path=data.get("module_path"),
        )

        self._skills[skill.name] = skill
        return skill

    # ────────────────────────────────────────────────────────────────
    # Query
    # ────────────────────────────────────────────────────────────────

    def register(self, skill: Skill) -> "SkillManager":
        """Register a skill programmatically."""
        self._skills[skill.name] = skill
        return self

    def find_relevant(self, task: str) -> list[Skill]:
        """Find all skills whose trigger keywords match the task."""
        return [s for s in self._skills.values() if s.matches(task)]

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """List all registered skill names."""
        return sorted(self._skills.keys())

    def skill_info(self, name: str) -> Optional[dict]:
        """Get full info about a skill."""
        s = self._skills.get(name)
        if not s:
            return None
        return {
            "name": s.name,
            "description": s.description,
            "trigger_keywords": s.trigger_keywords,
            "tools": s.tools,
            "examples": s.examples,
            "metadata": s.metadata,
        }

    # ────────────────────────────────────────────────────────────────
    # Skill execution (for skills with Python modules)
    # ────────────────────────────────────────────────────────────────

    def execute_skill(
        self,
        name: str,
        task: str,
        mimo: MiMoClient,
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Execute a skill's Python module if it has one."""
        skill = self._skills.get(name)
        if not skill or not skill.module_path:
            return {"error": f"Skill '{name}' has no module to execute."}

        try:
            module = self._load_module(skill.module_path)
            if hasattr(module, "run"):
                return module.run(task=task, mimo=mimo, context=context or {})
            else:
                return {"error": f"Module for skill '{name}' has no run() function."}
        except Exception as e:
            return {"error": str(e)}

    def _load_module(self, module_path: str):
        """Load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location("skill_module", module_path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def __repr__(self) -> str:
        return f"SkillManager({len(self._skills)} skills, dir={self.skills_dir})"
