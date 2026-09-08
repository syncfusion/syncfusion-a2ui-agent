"""SyncfusionAgent — the public façade for the ADK.

Wires together:
  - PromptBuilder      (default + customer extension + history)
  - ProviderManager    (Gemini / OpenAI / Azure / Claude / Ollama / custom)
  - MCPManager         (any MCP, platform or customer)
  - ToolOrchestrator   (MCP results only)
  - ResponseValidator  (A2UI JSON schema + bounded retry)
  - LoopEngine         (the per-request prompt + tool-call + retry loop,
                        lives in :mod:`._engine`)

The agent exposes the API documented in spec §3 and runs the flow in §5.
The runtime loop itself lives in :class:`_engine.LoopEngine`; this
module is intentionally limited to construction, registration, prompt
configuration, and thin delegators to the engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from ._engine import (
    LoopEngine,
    PromptBuilderWithExtension,
    RequestSetup,
    RequestSetupError,
    StreamContext,
)
from .mcp.mcp_manager import MCPManager, MCPServerConfig
from .orchestrator.tool_orchestrator import ToolOrchestrator
from .providers.base import AIProvider
from .providers.provider_registry import resolve_provider
from .skills.skill_registry import SkillRegistry
from .skills.skill_router import SkillRouter
from .validation.response_validator import ResponseValidator

logger = logging.getLogger("syncfusion_a2ui_agent")


class SyncfusionAgent:
    """High-level façade for Syncfusion A2UI agents."""

    DEFAULT_AI_TIMEOUT = 120.0

    def __init__(
        self,
        model: str | AIProvider | type[AIProvider] = "gemini",
        api_key: str | None = None,
        syncfusion_api_key: str | None = None,
        validation_max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        """Construct a :class:`SyncfusionAgent`.

        Parameters
        ----------
        model:
            Either an :class:`AIProvider` instance, an :class:`AIProvider`
            subclass, or a string key from the provider registry
            (``"gemini"``, ``"openai"``, ``"azure_openai"``, ``"claude"``,
            ``"ollama"``). For string keys, ``api_key`` is forwarded to
            the provider's constructor.
        api_key:
            Convenience kwarg forwarded to the resolved provider when
            ``model`` is a string. Ignored if ``model`` is an instance or
            a class.
        syncfusion_api_key:
            Forwarded to the platform MCP and stored on the agent for
            downstream consumers.
        validation_max_retries:
            Number of times :class:`ResponseValidator` may re-prompt the
            model when its output fails A2UI envelope validation. Total
            attempts are ``validation_max_retries + 1``. The default
            (2 retries → 3 attempts) matches the ADK policy; raise this
            for slower / less reliable models.
        **kwargs:
            Forwarded to the provider's constructor when ``model`` is a
            string or a class. Provider-specific options like
            ``reasoning_effort``, ``max_completion_tokens``, and
            ``azure_endpoint`` go here.
        """
        # ---- Provider -----------------------------------------------------------
        if isinstance(model, AIProvider):
            self._provider = model
        elif isinstance(model, type) and issubclass(model, AIProvider):
            self._provider = model(**kwargs)
        else:
            # Resolve via registry; `api_key` is the most common kwarg.
            if api_key and "api_key" not in kwargs:
                kwargs["api_key"] = api_key
            if syncfusion_api_key and "syncfusion_api_key" not in kwargs:
                kwargs["syncfusion_api_key"] = syncfusion_api_key
            self._provider = resolve_provider(model, **kwargs)
        self.syncfusion_api_key = syncfusion_api_key

        # ---- Sub-managers --------------------------------------------------------
        self.mcp = MCPManager()

        self.orchestrator = ToolOrchestrator(mcp_manager=self.mcp)
        self.validator = ResponseValidator(max_retries=validation_max_retries)
        self.prompt_builder = PromptBuilderWithExtension(self.extend_system_prompt)

        # ---- Skills (optional) --------------------------------------------------
        # Auto-enable when SKILLS_PATH is set in the environment.
        self._skill_router: SkillRouter | None = None
        if os.environ.get("SKILLS_PATH"):
            self.enable_skills()

        # ---- Request loop engine -----------------------------------------------
        # Owns the prompt → tool-call → validation-retry loop. Per-request
        # state lives on the caller's frame (see :class:`_engine.StreamContext`)
        # so concurrent A2A requests do not share mutable state.
        self._engine = LoopEngine(
            provider=self._provider,
            mcp_manager=self.mcp,
            orchestrator=self.orchestrator,
            validator=self.validator,
            prompt_builder=self.prompt_builder,
            skill_router=self._skill_router,
        )

    # ------------------------------------------------------------------ skills
    def enable_skills(
        self,
        skills_path: str | None = None,
    ) -> SkillRegistry:
        """Enable local-skill routing for this agent.

        Parameters
        ----------
        skills_path:
            Absolute (or cwd-relative) path to the skills directory.  When
            *None* the value is read from the ``SKILLS_PATH`` environment
            variable (set in your ``.env`` file).  The registry scans all
            sub-directories for a ``SKILL.md`` file on the first request.

        Returns
        -------
        The :class:`SkillRegistry` so callers can inspect discovered skills.
        """
        registry = SkillRegistry(skills_path=skills_path)
        self._skill_router = SkillRouter(registry=registry)
        # Keep the engine in sync with the new router so its
        # tool-call dispatch knows about the synthetic read_skill tool.
        self._engine._skill_router = self._skill_router  # noqa: SLF001 — see _engine.py
        logger.info(
            "Skills enabled — path: %r",
            str(registry._root) if registry._root else "(from SKILLS_PATH env)",
        )
        return registry

    def disable_skills(self) -> None:
        """Remove the skill router; all requests will go directly to MCP."""
        self._skill_router = None
        self._engine._skill_router = None  # noqa: SLF001 — see _engine.py

    # ------------------------------------------------------------------ extend hook
    def extend_system_prompt(self) -> str:
        """Override this to add a customer business prompt.

        The default prompt (UI rules) is ALWAYS prepended — you extend, not replace.
        """
        return ""

    # ------------------------------------------------------------------ catalog reference
    def set_catalog_reference(
        self,
        catalog_id: str,
        reference_text: str,
    ) -> None:
        """Inject a catalog reference (Knowledge Transfer) into the system prompt.

        The reference text is appended to the A2UI v0.9 grammar so the model
        knows which components the mounted renderer/client can display. It is
        purely a prompt asset — runtime validation remains generic.
        """
        from .prompts.prompt_builder import CatalogReference

        self.prompt_builder.set_catalog_reference(
            CatalogReference(catalog_id=catalog_id, reference_text=reference_text)
        )

    def set_catalog_reference_from_file(
        self,
        path: str,
        catalog_id: str | None = None,
    ) -> str:
        """Load a catalog reference from a JSON file and inject it.

        The file can be a basic A2UI v0.9 catalog (with `components`, `catalogId`,
        etc.) or any JSON file — its JSON is what's shown to the model. Returns
        the catalogId that was registered.
        """
        import json

        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        if catalog_id is None:
            catalog_id = (
                blob.get("catalogId") if isinstance(blob, dict) else None
            ) or "syncfusion-a2ui-catalog"
        text = json.dumps(blob, indent=2, ensure_ascii=False)
        self.set_catalog_reference(catalog_id=catalog_id, reference_text=text)
        return catalog_id

    def set_data_source(self, data: dict | list | str) -> None:
        """Inject an application data source into the system prompt.

        The LLM is instructed to use ONLY the values from this source when
        populating grids, charts, KPIs, and any other data-bound components.

        Args:
            data: A dict, list, or JSON string representing the data source.
                  Pass a dict whose keys match the well-known data categories
                  (employees, salesReps, regionalSales, inventory, etc.) or
                  any JSON-serialisable structure.
        """
        if isinstance(data, str):
            # Validate it's parseable JSON then store as-is
            import json as _json

            _json.loads(data)  # raises if invalid
            self.prompt_builder.set_data_context(data)
        else:
            import json as _json

            self.prompt_builder.set_data_context(_json.dumps(data, indent=2, ensure_ascii=False))

    def set_data_source_from_file(self, path: str) -> None:
        """Load a JSON file and inject it as the application data source.

        Equivalent to ``set_data_source(json.load(open(path)))``.

        Args:
            path: Absolute or relative path to a JSON file.
        """
        import json as _json

        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        self.set_data_source(data)

    # -- Design contract (A2UI integration) ---------------------
    # One method, many input shapes. The agent treats the design as an
    # authoritative template and echoes its structure verbatim on every
    # request, only varying data values derived from the user's prompt.
    # See ``prompts/prompt_builder.py`` for the prompt block.

    def set_design(
        self,
        design: str | dict[str, Any] | list[Any],
    ) -> None:
        """Inject one or more A2UI design contracts into the system prompt.

        A single call accepts almost any reasonable input:

        * **A dict / list** — a parsed A2UI v0.9 envelope.
        * **A JSON string** — an A2UI envelope serialized as JSON.
        * **A file path** (string ending in ``.json``) — the file is loaded
          and validated automatically.
        * **A directory path** (existing directory) — every ``*.json`` in
          the directory is loaded as a separate design (multi-page mode).
        * **A list of any of the above** — every element is processed in
          turn, producing a multi-design catalog.

        When multiple designs are provided, they are concatenated into a
        single markdown catalog so the LLM can see every page at once
        and pick the one that matches the user's intent.

        Examples:

            agent.set_design(design_dict)                          # one design
            agent.set_design('{"version":"v0.9", ...}')            # one JSON string
            agent.set_design('designs/finance.json')               # one file
            agent.set_design('designs/')                           # all .json in dir
            agent.set_design([                                     # mixed list
                'designs/dashboard.json',
                'designs/employees.json',
                {'version': 'v0.9', 'createSurface': {...}},        # inline dict
            ])

        Args:
            design: A single A2UI design (dict / list / str / path) or
                a list of any of those.
        """
        import json as _json

        # Normalise to a list of "raw design" entries. Each entry is either
        # a dict/list (parsed JSON) or a markdown text block.
        items = self._normalise_design(design)
        if not items:
            raise ValueError("set_design: no designs to inject")

        # If there's only one design, treat it as a single A2UI envelope
        # (no catalog headers) — the LLM just echoes it.
        if len(items) == 1:
            payload = items[0]
            # If it's a dict/list, serialise it for the prompt.
            if isinstance(payload, (dict, list)):
                payload = _json.dumps(payload, indent=2, ensure_ascii=False)
            self.prompt_builder.set_surface_examples(payload)
            return

        # Multiple designs → build a multi-page markdown catalog.
        blocks: list[str] = []
        for entry in items:
            sid = self._extract_surface_id(entry) or "?"
            if isinstance(entry, (dict, list)):
                body = _json.dumps(entry, indent=2, ensure_ascii=False)
            else:
                body = entry  # already text
            blocks.append(
                f"### Page: `{sid}`\n**Design (A2UI v0.9 envelope):**\n```json\n{body}\n```"
            )
        catalog = "\n\n".join(blocks)
        self.prompt_builder.set_surface_examples(catalog)

    @staticmethod
    def _normalise_design(
        design: str | dict[str, Any] | list[Any],
    ) -> list[str | dict[str, Any] | list[dict[str, Any]]]:
        """Turn any accepted input shape into a flat list of designs.

        Each returned item is either a dict/list (parsed A2UI envelope)
        or a str (pre-rendered text). The caller decides how to render.

        A list whose elements are all A2UI op-dicts (i.e. each element
        has one of the canonical op keys: ``version`` + ``createSurface`` /
        ``updateComponents`` / ``updateDataModel`` / ``deleteSurface``)
        is treated as **one design** — the canonical envelope shape.
        Otherwise, the list is treated as a **list of designs** and
        each element is normalised recursively.
        """
        import json as _json
        import os as _os

        _OP_KEYS = ("createSurface", "updateComponents", "updateDataModel", "deleteSurface")

        # List → either one envelope (all elements are op-dicts) or a
        # list of designs (mixed elements / non-op-dicts).
        if isinstance(design, list):
            if design and all(
                isinstance(elem, dict) and any(k in elem for k in _OP_KEYS) for elem in design
            ):
                # Canonical envelope — keep as one design.
                return [design]
            # List of designs — recurse over elements.
            out: list = []
            for elem in design:
                out.extend(SyncfusionAgent._normalise_design(elem))
            return out

        # Dict → already a parsed envelope.
        if isinstance(design, dict):
            return [design]

        # Str → could be a JSON string, a file path, or a directory path.
        if isinstance(design, str):
            s = design.strip()
            # Directory → load every *.json file.
            if _os.path.isdir(s):
                out = []
                for fn in sorted(_os.listdir(s)):
                    if fn.endswith(".json"):
                        with open(_os.path.join(s, fn), encoding="utf-8") as f:
                            out.append(_json.load(f))
                return out
            # File path → load it.
            if s.endswith(".json") and _os.path.isfile(s):
                with open(s, encoding="utf-8") as f:
                    return [_json.load(f)]
            # JSON string → try to parse.
            try:
                return [_json.loads(s)]
            except _json.JSONDecodeError:
                # Treat as opaque text (markdown catalog the user built).
                return [s]

        raise TypeError(
            f"set_design: unsupported input type {type(design).__name__}. "
            "Pass a dict, list, JSON string, file path, or directory path."
        )

    @staticmethod
    def _extract_surface_id(
        design: Any,
    ) -> str | None:
        """Best-effort surfaceId extraction from a parsed A2UI envelope."""
        if not isinstance(design, list):
            return None
        for m in design:
            if isinstance(m, dict) and "createSurface" in m:
                sid = m["createSurface"].get("surfaceId")
                if sid:
                    return sid
        return None

    def warm_catalog(self) -> asyncio.Task | None:
        """Spawn a background task that warms the MCP tool catalog now.

        Call this at boot (e.g. right after ``add_mcp`` + catalog
        reference) so the first user-facing request doesn't pay the
        5-second npx spawn cost.

        Returns the asyncio.Task (when called from a running event loop),
        or schedules it to run on the next available loop and returns
        ``None`` when called from sync code at boot. Either way, the
        warmup is fire-and-forget — you can ignore the return value.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Sync caller at boot (e.g. before uvicorn starts). Schedule
            # the warmup so it runs the first time an event loop becomes
            # available; the first user-facing request will then pay
            # only the in-flight warmup tail, not a fresh spawn.
            self.mcp._deferred_warmup = True
            return None
        return self.mcp.warm_start()

    # ------------------------------------------------------------------ registration
    def add_mcp(
        self,
        name: str,
        command: list[str] | str | None = None,
        url: str | None = None,
        kind: str = "customer",
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> MCPServerConfig:
        return self.mcp.add_mcp(
            name=name, command=command, url=url, kind=kind, env=env, timeout=timeout
        )

    def remove_mcp(self, name: str) -> None:
        self.mcp.remove_mcp(name)

    # ------------------------------------------------------------------ runtime
    async def handle_request(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Core runtime flow (spec §5).

        Returns a dict shaped like:
            {
                "envelope":  [...],   # the validated A2UI v0.9 envelope
                "surfaceIds":[...],
            }
        so the A2A transport can forward the response to the client.
        """
        return await self._engine.handle(user_message, conversation_history)

    async def handle_request_stream(
        self,
        user_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming variant of :meth:`handle_request` (Level 2 of the streaming plan).

        Yields a sequence of events shaped like::

            {"type": "op",        "op": {...}, "kind": "createSurface"|"updateComponents"|"updateDataModel"}
            {"type": "surfaceIds","surfaceIds": [...]}
            {"type": "error",     "error": "..."}

        Thin delegator to :meth:`_engine.LoopEngine.handle_stream`.
        """
        async for event in self._engine.handle_stream(user_message, conversation_history):
            yield event

    # ------------------------------------------------------------------ serve
    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        warmup: bool = False,
        allowed_origins: list[str] | None = None,
    ) -> None:  # pragma: no cover
        """Boot the Google A2A SDK server and block.

        Thin delegator to :func:`syncfusion_a2ui_agent.agent_serve.serve`
        — kept on the agent class for API ergonomics. See the helper's
        docstring for the full parameter reference and the CORS / mcp
        teardown suppression contract.
        """
        from .agent_serve import serve as _serve

        _serve(
            agent=self,
            host=host,
            port=port,
            warmup=warmup,
            allowed_origins=allowed_origins,
        )


__all__ = [
    "SyncfusionAgent",
    "LoopEngine",
    "PromptBuilderWithExtension",
    "RequestSetup",
    "RequestSetupError",
    "StreamContext",
]
