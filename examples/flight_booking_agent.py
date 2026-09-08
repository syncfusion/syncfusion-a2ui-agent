"""SkyWave Airlines flight-booking agent.

A thin runtime wrapper around :class:`SyncfusionAgent` that wires up the
SkyWave A2UI design catalog at boot and serves the A2A protocol on
``http://127.0.0.1:10006`` for the bundled React client (and the
``test-page.html`` harness).

Run with::

    python examples/flight_booking_agent.py
"""

from __future__ import annotations

import logging
import os
from typing import Union

from syncfusion_a2ui_agent import SyncfusionAgent
from syncfusion_a2ui_agent.providers.base import AIProvider

_log = logging.getLogger("syncfusion_a2ui_agent")

DESIGNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "designs")

# Speed knobs applied to the Azure OpenAI provider when running the
# design-echo path. ``gpt-5.4`` is a reasoning model; the default
# reasoning effort burns 15-25 s on the thinking pass for what is
# essentially a verbatim copy. ``low`` keeps correctness and cuts the
# think step ~5x. The completion cap is a tight ceiling above the
# ~3K tokens needed to echo a single 11-13 KB design envelope; the
# validator will catch any truncation and the SDK re-asks once.
_LLM_SPEED_KNOBS: dict[str, Union[str, int]] = {
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_completion_tokens": 4096,
}

_REQUIRED_ENV_VARS = (
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
    "MODEL_NAME",
)


class FlightBookingAgent(SyncfusionAgent):
    """Echoes one of the three pre-defined SkyWave pages per request."""

    def __init__(self, model: Union[str, AIProvider, type[AIProvider]]) -> None:
        super().__init__(model=model)

        # Pre-load the full catalog once at boot. The hot path will swap
        # in a single design for clear-intent queries so the model only
        # ever sees one page.
        if not os.path.isdir(DESIGNS_DIR):
            raise FileNotFoundError(
                f"Designs directory not found: {DESIGNS_DIR!r}. "
                "Create it with stage1/stage2/stage3 .json files before "
                "starting the agent."
            )
        self.set_design(DESIGNS_DIR)

        # Optional local skill routing when SKILLS_PATH is set in .env.
        skills_path = os.environ.get("SKILLS_PATH", "")
        if skills_path:
            registry = self.enable_skills(skills_path)
            skills = registry.all_skills()
            _log.info(
                "Loaded %d skill(s) from %s: %s",
                len(skills),
                skills_path,
                [s.name for s in skills],
            )
        else:
            _log.warning(
                "SKILLS_PATH not set - skill routing disabled. "
                "Add SKILLS_PATH=<path> to your .env to enable it."
            )

    def extend_system_prompt(self) -> str:
        """Append a directive that pins the verbatim-echo contract.

        The catalog of pages is already injected by ``set_design`` on the
        base class, so we only need to tell the model what to do with
        it: echo verbatim, or fall back to the out-of-scope stub.
        """
        return (
            "You are the SkyWave Airlines flight-booking assistant. "
            "Echo the design above verbatim (same surfaceId, same components, "
            "same dataModel). Vary only the data values. "
            "Out of scope -> workspace surface with AppBar + Message "
            "(severity Information, content 'I can only assist with "
            "SkyWave Airlines flight bookings.') + Button 'Book a Flight'."
        )


def _require_env(env_path: str) -> None:
    """Raise a helpful error if ``.env`` or required vars are missing."""
    if not os.path.isfile(env_path):
        raise FileNotFoundError(
            f".env file not found at {env_path!r}. Copy .env.example to .env "
            "and fill in your Azure OpenAI credentials before starting the agent."
        )
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Set them in your .env file or shell before starting the agent."
        )


def main() -> None:
    from dotenv import load_dotenv
    from syncfusion_a2ui_agent.providers.azure_openai_provider import (
        AzureOpenAIProvider,
    )

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    _require_env(env_path)

    provider = AzureOpenAIProvider(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
        azure_deployment=os.environ["MODEL_NAME"],
        **_LLM_SPEED_KNOBS,
    )
    agent = FlightBookingAgent(model=provider)
    # A2A server - the bundled test-page.html POSTs to this URL.
    agent.serve(host="127.0.0.1", port=10006)


if __name__ == "__main__":
    main()
