"""Default :class:`a2a.types.AgentCard` builder for the ADK.

Customers may override this with their own card by passing
``agent_card=...`` to :meth:`A2ATransport.build_app`. The
default card advertises a single ``a2ui-generate`` skill and
reports ``streaming=True`` so clients know the agent supports
``message/stream`` over SSE.

The builder is import-time tolerant: when the ``a2a-sdk`` is not
installed (e.g. in unit tests that don't pull the optional dep),
it returns a plain ``dict`` that mirrors the AgentCard shape well
enough for serialization in tests.
"""

from __future__ import annotations

from typing import Any

from .transport import _A2A_AVAILABLE

# ``A2UI_GENERATE`` is the canonical skill id advertised by the
# ADK. Renderers / orchestrators can match against it to detect
# that the agent speaks A2UI v0.9.
A2UI_GENERATE_SKILL_ID = "a2ui-generate"


def default_agent_card(
    name: str,
    url: str = "http://localhost:10004/",
    version: str = "0.1.0",
) -> Any:
    """Build a minimal AgentCard for the agent.

    Parameters
    ----------
    name:
        The agent's display name. Defaults to the class name when
        the caller doesn't pass one.
    url:
        The public URL the agent is reachable at. The transport
        uses this for the agent card's ``url`` field; it should
        match the host/port the A2A server is bound to.
    version:
        The ADK version string embedded in the card. Defaults to
        the package's ``__version__``; pass explicitly when
        embedding the agent card in a downstream application that
        wants its own version.

    Returns
    -------
    a2a.types.AgentCard
        When the ``a2a-sdk`` is importable.

    dict
        A minimal plain-dict fallback otherwise. The shape mirrors
        AgentCard closely enough that ``json.dumps(card)`` works
        in tests and dev environments without the optional dep.
    """
    description = "Syncfusion A2UI Agent — generates A2UI JSON for the Syncfusion React renderer."
    if not _A2A_AVAILABLE:  # pragma: no cover - import-time guard
        return {
            "name": name,
            "description": description,
            "url": url,
            "version": version,
            "capabilities": {"streaming": True},
            "default_input_modes": ["text/plain"],
            "default_output_modes": ["application/json"],
            "skills": [
                {
                    "id": A2UI_GENERATE_SKILL_ID,
                    "name": "Generate A2UI",
                    "description": ("Generate A2UI JSON for the Syncfusion React renderer."),
                    "tags": ["a2ui", "syncfusion", "react"],
                }
            ],
        }
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    return AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id=A2UI_GENERATE_SKILL_ID,
                name="Generate A2UI",
                description=("Generate A2UI JSON for the Syncfusion React renderer."),
                tags=["a2ui", "syncfusion", "react"],
            )
        ],
    )


__all__ = ["A2UI_GENERATE_SKILL_ID", "default_agent_card"]
