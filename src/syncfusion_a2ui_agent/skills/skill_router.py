"""SkillRouter — VS Code-style single-call skill routing.

How it works
------------
1. All skill descriptions are injected into the system prompt via
   :meth:`get_catalog_block` so the model knows what skills exist.
2. A synthetic ``read_skill`` tool is added to the provider's tool list via
   :meth:`get_skill_tool_definition`.
3. The model calls ``read_skill(name=...)`` for each component it plans to
   render. The agent's tool-call loop dispatches those calls to
   :meth:`handle_read_skill_call`, which reads and returns the full SKILL.md.
4. The model generates A2UI JSON informed by the skill content — all within
   one AI session (two HTTP calls total when parallel_tool_calls is enabled).
"""

from __future__ import annotations

import logging
from typing import Any

from .skill_registry import SkillRegistry

logger = logging.getLogger("syncfusion_a2ui_agent.skills")

_READ_SKILL_TOOL_NAME = "read_skill"

_READ_SKILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _READ_SKILL_TOOL_NAME,
        "description": (
            "Read the full knowledge document (SKILL.md) for a named skill. "
            "Call this BEFORE generating A2UI JSON for a component you find in "
            "the Available Skills list. The document contains exact API props, "
            "code patterns, and rules you must follow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The exact skill name from the Available Skills list.",
                }
            },
            "required": ["name"],
        },
    },
}


class SkillRouter:
    """Provides skill catalog injection and on-demand SKILL.md delivery.

    Parameters
    ----------
    registry:
        Pre-populated :class:`SkillRegistry`. Discovery is triggered lazily
        on first access.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def get_skill_tool_definition(self) -> dict[str, Any]:
        """Return the ``read_skill`` tool definition for the provider.

        Add this to the ``tools`` list passed to the AI provider so the
        model can fetch SKILL.md content on demand.
        """
        return _READ_SKILL_TOOL

    def get_catalog_block(self) -> str:
        """Return a system-prompt block listing all available skills.

        Inject this into the system prompt once so the model knows which
        skills exist and can call ``read_skill`` for relevant ones.
        Returns an empty string when no skills are registered.
        """
        skills = self._registry.all_skills()
        if not skills:
            return ""
        lines = [
            "## Available Skills",
            "",
            "The following skills contain authoritative knowledge for specific "
            "Syncfusion React components. You MUST call the `read_skill` tool "
            "for EVERY component type you plan to render that appears in this list — "
            "even if you think you know the API. The skill document contains exact "
            "prop names, required Inject services, and A2UI JSON patterns that "
            "differ from your training data. Call `read_skill` for each relevant "
            "skill BEFORE writing any A2UI JSON.",
            "",
        ]
        for s in skills:
            lines.append(f"- **{s.name}**: {s.description}")
        lines.append("")
        return "\n".join(lines)

    def handle_read_skill_call(self, name: str) -> str:
        """Handle a ``read_skill`` tool call from the model.

        Resolves the skill by name (exact match, then substring fallback)
        and returns the full SKILL.md content, or a descriptive error string
        if not found so the model can recover gracefully.
        """
        skills = self._registry.all_skills()
        name_lower = name.strip().lower()

        for s in skills:
            if s.name.lower() == name_lower:
                try:
                    return s.load_content()
                except Exception as exc:
                    return f"<error reading skill '{s.name}': {exc}>"

        # Substring fallback for minor name mismatches.
        for s in skills:
            if name_lower in s.name.lower() or s.name.lower() in name_lower:
                try:
                    return s.load_content()
                except Exception as exc:
                    return f"<error reading skill '{s.name}': {exc}>"

        return f"<skill '{name}' not found. Available: {[s.name for s in skills]}>"

    @staticmethod
    def is_read_skill_call(tool_name: str) -> bool:
        """Return True when *tool_name* matches the ``read_skill`` synthetic tool."""
        return tool_name == _READ_SKILL_TOOL_NAME
