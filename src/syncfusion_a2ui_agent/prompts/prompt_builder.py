"""PromptBuilder — merges the default A2UI system prompt with the customer
extension + an optional catalog reference + history.

The builder is the single point where the generic A2UI v0.9 grammar, the
customer's business prompt, the mounted platform's component reference, and
the current conversation are joined. Per the spec:
  - customers extend, they never replace the default.
  - history is preserved as a separate section the provider can format.
  - the catalog reference is a *prompt asset* (KT) — it teaches the model
    what components are mounted on the renderer. Validation against the
    catalog is the renderer's job, not this SDK's.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .default_system_prompt import get_default_system_prompt


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""

    role: str  # "user" | "assistant" | "tool"
    content: str
    name: str | None = None


@dataclass
class ToolResult:
    """The merged result of orchestrator tool calls, ready for AI context."""

    source: str
    payload: Any


@dataclass
class CatalogReference:
    """An optional catalog asset embedded into the prompt as Knowledge Transfer."""

    catalog_id: str
    reference_text: str


@dataclass
class BuiltPrompt:
    """The fully-built prompt handed to the AI provider."""

    system: str
    history: list[ConversationTurn] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    user_message: str = ""

    def render_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        for turn in self.history:
            msgs.append({"role": turn.role, "content": turn.content})
        if self.tool_results:
            joined = "\n\n".join(f"[{tr.source}]\n{tr.payload}" for tr in self.tool_results)
            msgs.append(
                {
                    "role": "system",
                    "content": f"Tool results available:\n{joined}",
                }
            )
        if self.user_message:
            msgs.append({"role": "user", "content": self.user_message})
        return msgs


class PromptBuilder:
    """Builds the system prompt + history + tool-results into a single request."""

    def __init__(
        self,
        default_system_prompt: str | None = None,
        catalog_reference: CatalogReference | None = None,
    ) -> None:
        self._default = default_system_prompt or get_default_system_prompt()
        self._catalog_reference = catalog_reference
        self._data_context: str | None = None
        self._skill_catalog: str | None = None
        self._surface_examples: str | None = None
        self._history_provider: Callable[[], Iterable[ConversationTurn]] = lambda: []

    def set_history_provider(self, provider: Callable[[], Iterable[ConversationTurn]]) -> None:
        self._history_provider = provider

    def set_catalog_reference(self, reference: CatalogReference | None) -> None:
        self._catalog_reference = reference

    def set_data_context(self, data_json: str | None) -> None:
        """Inject a JSON data context block into the system prompt.

        The LLM is instructed to use ONLY the values in this block when
        populating dataSource arrays, KPI values, dates, names, etc.  This
        makes responses reproducible and grounded in real application data.
        """
        self._data_context = data_json

    def set_skill_catalog(self, catalog_block: str | None) -> None:
        """Inject a skill catalog block into the system prompt.

        The catalog lists available skills and instructs the model to call
        the ``read_skill`` tool before generating A2UI JSON for those
        components (VS Code Copilot-style single-call approach).
        """
        self._skill_catalog = catalog_block

    def set_surface_examples(self, examples_json: str | None) -> None:
        """Inject UI surface examples into the system prompt.

        The model is instructed to follow the structure and patterns of
        these example surfaces when generating new UI. Useful for teaching
        the model a specific layout style or component composition pattern.
        """
        self._surface_examples = examples_json

    def build(
        self,
        user_message: str,
        tool_results: Iterable[ToolResult] | None = None,
    ) -> BuiltPrompt:
        customer_extension = self._customer_extension()
        catalog_intro = self._render_catalog_intro()
        data_context_block = self._render_data_context()
        skill_catalog_block = self._skill_catalog or ""
        surface_examples_block = self._render_surface_examples()
        system = self._default
        if customer_extension:
            system += "\n\n" + customer_extension
        if data_context_block:
            system += "\n\n" + data_context_block
        if skill_catalog_block:
            system += "\n\n" + skill_catalog_block
        if surface_examples_block:
            system += "\n\n" + surface_examples_block
        if catalog_intro:
            system += "\n\n" + catalog_intro
        history = list(self._history_provider())
        return BuiltPrompt(
            system=system,
            history=history,
            tool_results=list(tool_results) if tool_results else [],
            user_message=user_message,
        )

    def _customer_extension(self) -> str:
        """Hook for subclasses (e.g. :class:`SyncfusionAgent`) to extend
        the system prompt.

        Override this method to inject domain-specific instructions,
        persona, or business rules. The returned string is appended to
        the default A2UI system prompt on every :meth:`build` call. The
        default implementation returns an empty string.

        Most customers should subclass :class:`SyncfusionAgent` and
        override :meth:`SyncfusionAgent.extend_system_prompt` instead,
        which uses the same hook under the hood.
        """
        return ""

    def _render_surface_examples(self) -> str:
        if not self._surface_examples:
            return ""
        return f"""## UI Surface Design Contract (authoritative)

            The JSON below is the AUTHORITATIVE design for every UI you emit. You MUST
            echo the structure verbatim — same component IDs, same component types, same
            props, same child references, same data model paths, same event names.

            RULES:
            1. **Structure is fixed.** Every component, id, prop, and child reference in
            the reference must appear in your output unchanged.
            2. **Only data values vary.** Populate text labels, placeholders, computed
            values (e.g. fares, dates, totals), and the initial dataModel from the
            user's prompt.
            3. **Do not add, remove, or reorder components.** Do not introduce new ids.
            4. **Do not change prop names or cssClass values.** If a prop is a path
            reference (e.g. `value:{{path:"/form/origin"}}`), keep it as a path
            reference — do not inline a literal value.
            5. **Preserve every event handler** (e.g. `onClick:{{event:{{name:"..."}}}}`)
            exactly as written.
            6. **Update the dataModel** in your `updateDataModel` operation so the
            form/summary cards render with the right initial values.
            7. **Generate a NEW surfaceId on every response.** The reference's surfaceId
            (e.g. `workspace`, `stage_2_booking`) is just a template — the renderer
            rejects re-used ids with "Surface already exists". Pick a fresh id per
            response, e.g. `workspace_<random>` or `<stage>_<timestamp>`. The same
            new id must appear in `createSurface.surfaceId`,
            `updateComponents.surfaceId`, and every `updateDataModel.surfaceId`.

            REFERENCE DESIGN (treat this as a template — your output must match its shape):

            {self._surface_examples}

            Emit one A2UI v0.9 envelope that contains:
            - `createSurface` with a NEW unique surfaceId and the same catalogId
            - `updateComponents` with the same component tree (same ids, same children, same props) on the new surfaceId
            - `updateDataModel` with the path/value pairs needed to fill the design, all on the new surfaceId
            """

    def _render_data_context(self) -> str:
        if not self._data_context:
            return ""
        return (
            "=== Application Data Source ===\n"
            "The following JSON contains the AUTHORITATIVE data for this application.\n"
            "RULES:\n"
            "1. When populating dataSource arrays (grids, charts, lists), use ONLY\n"
            "   the records from the matching key below — do NOT invent new rows.\n"
            "2. When showing KPI values, names, dates, amounts, or any other factual\n"
            "   numbers, use ONLY the values present in this data — do NOT fabricate.\n"
            "3. Use the key that best matches the user request:\n"
            "   employees → HR grids  |  salesReps → sales leaderboards\n"
            "   regionalSales → regional reports  |  inventory → stock grids\n"
            "   transactions → ledger/finance  |  projects → project portfolio\n"
            "   incidents → ops/monitoring  |  monthlyRevenue → revenue charts\n"
            "   productRevenue → product breakdown charts\n"
            "   calendarEvents → scheduler/calendar  |  vendors → vendor forms\n"
            "4. You MAY still generate component structure (IDs, layout, styling)\n"
            "   freely — only the data values must come from this source.\n\n"
            f"{self._data_context}\n"
            "=== End Application Data Source ==="
        )

    def _render_catalog_intro(self) -> str:
        ref = self._catalog_reference
        if ref is None:
            return ""
        return (
            "=== Supported Components (catalog reference for the renderer "
            f"mounted on this client) ===\n"
            f"CATALOG_ID={ref.catalog_id}\n"
            f"{ref.reference_text}\n"
            f'CRITICAL: In your createSurface operation, set catalogId to "{ref.catalog_id}"\n'
            "(Use these components verbatim in your updateComponents ops. "
            "Component-level property validation is the renderer's job; "
            "you only need to follow the A2UI v0.9 envelope contract.)"
        )
