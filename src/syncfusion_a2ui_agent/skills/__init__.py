"""Skills sub-package — local skill discovery, routing, and execution.

Public surface::

    from syncfusion_a2ui_agent.skills import SkillRegistry, SkillRouter, SkillMetadata
"""

from .skill_registry import SkillMetadata, SkillRegistry
from .skill_router import SkillRouter

__all__ = ["SkillMetadata", "SkillRegistry", "SkillRouter"]
