"""SkillRegistry — discovers and loads local skill definitions.

Skills are directories under SKILLS_PATH.  Each directory must contain a
``SKILL.md`` file whose YAML front-matter declares at minimum:

    ---
    name: <skill-name>
    description: "<one-sentence description used for matching>"
    ---

The registry is intentionally *static* — it reads the filesystem once at
construction time.  No file-watching: add new skills and restart the agent.
No extra dependencies: uses stdlib ``re`` for front-matter parsing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("syncfusion_a2ui_agent.skills")

# ---------------------------------------------------------------------------
# Front-matter parsing (no PyYAML dependency)
# ---------------------------------------------------------------------------

_FM_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)
_FIELD_RE = re.compile(r'^(\w+)\s*:\s*"?(.*?)"?\s*$', re.MULTILINE)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Return the YAML front-matter key→value map from *text*.

    Only scalar string values are extracted; multi-line / lists are ignored.
    Returns an empty dict if no front-matter is found.
    """
    m = _FM_RE.match(text)
    if not m:
        return {}
    return {k: v for k, v in _FIELD_RE.findall(m.group(1))}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SkillMetadata:
    """Lightweight descriptor for a discovered skill.

    ``content`` is *not* loaded eagerly — call :meth:`SkillRegistry.load_content`
    when you actually need the full SKILL.md body.
    """

    name: str
    description: str
    path: Path  # absolute path to the skill directory
    _content: str | None = field(default=None, repr=False)

    @property
    def skill_file(self) -> Path:
        return self.path / "SKILL.md"

    def load_content(self) -> str:
        """Read and cache the full SKILL.md text."""
        if self._content is None:
            self._content = self.skill_file.read_text(encoding="utf-8")
        return self._content


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Discovers skills under *skills_path* and exposes them for routing.

    Parameters
    ----------
    skills_path:
        Absolute (or cwd-relative) path to the directory that contains
        individual skill sub-directories.  When *None* the value is read
        from the ``SKILLS_PATH`` environment variable; if that is also
        absent the registry is empty (no skills available).
    """

    def __init__(self, skills_path: str | None = None) -> None:
        resolved = skills_path or os.environ.get("SKILLS_PATH", "")
        self._root: Path | None = Path(resolved) if resolved else None
        self._skills: list[SkillMetadata] = []
        self._discovered = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[SkillMetadata]:
        """Scan the skills directory and return all valid skill descriptors.

        Results are cached after the first call — subsequent calls return the
        same list without touching the filesystem.
        """
        if self._discovered:
            return self._skills

        self._discovered = True
        if self._root is None or not self._root.is_dir():
            if self._root is not None:
                logger.warning(
                    "SKILLS_PATH %r is not a directory — no skills loaded.",
                    str(self._root),
                )
            return self._skills

        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                logger.debug("Skipping %r — no SKILL.md found.", entry.name)
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
                fm = _parse_frontmatter(text)
                name = fm.get("name", entry.name)
                description = fm.get("description", "")
                if not description:
                    logger.debug("Skill %r has no description — still registered.", name)
                self._skills.append(SkillMetadata(name=name, description=description, path=entry))
                logger.debug("Discovered skill %r from %r", name, str(entry))
            except Exception as exc:
                logger.warning("Failed to load skill at %r: %s", str(entry), exc)

        logger.info(
            "SkillRegistry: discovered %d skill(s) under %r",
            len(self._skills),
            str(self._root),
        )
        return self._skills

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> SkillMetadata | None:
        """Return a skill by *exact* name, or ``None``."""
        for s in self.discover():
            if s.name == name:
                return s
        return None

    def all_skills(self) -> list[SkillMetadata]:
        """Return all discovered skills (triggers discovery on first call)."""
        return self.discover()

    def is_empty(self) -> bool:
        return len(self.discover()) == 0
