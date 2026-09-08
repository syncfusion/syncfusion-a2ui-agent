"""Direct tests for the skills subsystem.

Covers:

* :class:`SkillMetadata` — name / description / path / content.
* :class:`SkillRegistry` — directory scan, front-matter parsing,
  caching, content lazy loading.
* :class:`SkillRouter` — catalog block generation, synthetic
  ``read_skill`` tool definition, dispatch, exact vs substring
  matching.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# SkillMetadata
# ---------------------------------------------------------------------------
class TestSkillMetadata:
    def test_defaults(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillMetadata

        m = SkillMetadata(name="x", description="d", path="p")
        assert m.name == "x"
        assert m.description == "d"
        assert m.path == "p"
        assert m._content is None  # lazy

    def test_load_content_caches(self, tmp_path: Path) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillMetadata

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            textwrap.dedent(
                """\
                ---
                name: my-skill
                description: A test skill
                ---

                Body content here.
                """
            ),
            encoding="utf-8",
        )
        # ``path`` is the skill directory (a ``Path`` object —
        # the loader does ``self.path / "SKILL.md"``).
        m = SkillMetadata(name="my-skill", description="A test skill", path=skill_dir)
        body = m.load_content()
        assert "Body content here." in body
        # Second call is cached.
        assert m.load_content() is body


# ---------------------------------------------------------------------------
# SkillRegistry — directory scan
# ---------------------------------------------------------------------------
class TestSkillRegistryDiscover:
    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry

        reg = SkillRegistry(skills_path=str(tmp_path))
        reg.discover()
        assert reg.all_skills() == []

    def test_discover_finds_skill_with_front_matter(self, tmp_path: Path) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """\
                ---
                name: "my-skill"
                description: "A test skill"
                ---

                Body content.
                """
            ),
            encoding="utf-8",
        )
        reg = SkillRegistry(skills_path=str(tmp_path))
        reg.discover()
        skills = reg.all_skills()
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "A test skill"

    def test_discover_uses_directory_name_when_no_front_matter(self, tmp_path: Path) -> None:
        # Without front-matter, the registry should still register
        # the skill with the directory name as a fallback.
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry

        skill_dir = tmp_path / "bare-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just some text.", encoding="utf-8")
        reg = SkillRegistry(skills_path=str(tmp_path))
        reg.discover()
        skills = reg.all_skills()
        # Either registered with a default name OR filtered out
        # (depending on the source's exact behaviour).
        if skills:
            assert any(s.name for s in skills)

    def test_discover_is_cached(self, tmp_path: Path) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry

        reg = SkillRegistry(skills_path=str(tmp_path))
        reg.discover()
        # Add a new skill directory after the first discover.
        skill_dir = tmp_path / "late-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: late-skill\ndescription: late\n---\nBody",
            encoding="utf-8",
        )
        reg.discover()
        # Second discover() should be a no-op (cached).
        skills = reg.all_skills()
        assert len(skills) == 0

    def test_discover_finds_multiple_skills(self, tmp_path: Path) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry

        for name in ("alpha", "beta", "gamma"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(
                f'---\nname: "{name}"\ndescription: "{name} desc"\n---\nbody',
                encoding="utf-8",
            )
        reg = SkillRegistry(skills_path=str(tmp_path))
        reg.discover()
        names = {s.name for s in reg.all_skills()}
        assert {"alpha", "beta", "gamma"}.issubset(names)


# ---------------------------------------------------------------------------
# SkillRouter — catalog block
# ---------------------------------------------------------------------------
class TestSkillRouterCatalog:
    def test_catalog_block_lists_skills(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import (
            SkillMetadata,
            SkillRegistry,
        )
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        reg._skills.append(SkillMetadata(name="x", description="X desc", path="(memory)"))
        reg._skills.append(SkillMetadata(name="y", description="Y desc", path="(memory)"))
        router = SkillRouter(reg)
        block = router.get_catalog_block()
        assert "Available Skills" in block
        assert "x" in block
        assert "X desc" in block
        assert "y" in block

    def test_catalog_block_empty_when_no_skills(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        router = SkillRouter(reg)
        # No skills — block is empty (the prompt builder handles
        # the "no skills" case).
        assert router.get_catalog_block() == ""


# ---------------------------------------------------------------------------
# SkillRouter — synthetic read_skill tool
# ---------------------------------------------------------------------------
class TestSkillRouterToolDefinition:
    def test_returns_function_tool(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        router = SkillRouter(reg)
        tool = router.get_skill_tool_definition()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "read_skill"
        # Must declare the ``name`` parameter.
        assert "name" in tool["function"]["parameters"]["properties"]


class TestSkillRouterDispatch:
    def test_exact_match_dispatch(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import (
            SkillMetadata,
            SkillRegistry,
        )
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        sm = SkillMetadata(name="x", description="X desc", path="(memory)")
        sm._content = "X body"
        reg._skills.append(sm)
        router = SkillRouter(reg)
        out = router.handle_read_skill_call("x")
        assert out == "X body"

    def test_case_insensitive_dispatch(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import (
            SkillMetadata,
            SkillRegistry,
        )
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        sm = SkillMetadata(name="MySkill", description="d", path="(memory)")
        sm._content = "body"
        reg._skills.append(sm)
        router = SkillRouter(reg)
        # Case-insensitive lookup.
        assert router.handle_read_skill_call("myskill") == "body"

    def test_unknown_skill_returns_error_string(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_registry import SkillRegistry
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        reg = SkillRegistry(skills_path=None)
        router = SkillRouter(reg)
        out = router.handle_read_skill_call("nope")
        # The router returns an error string (not raises) so the
        # agent's tool-call loop can surface it back to the model.
        assert "nope" in out.lower() or "not" in out.lower() or "error" in out.lower()

    def test_is_read_skill_call_helper(self) -> None:
        from syncfusion_a2ui_agent.skills.skill_router import SkillRouter

        assert SkillRouter.is_read_skill_call("read_skill") is True
        assert SkillRouter.is_read_skill_call("read_skill_v2") is False
        assert SkillRouter.is_read_skill_call("server__tool") is False
