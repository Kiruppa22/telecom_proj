import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


async def main():
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv(
        "ANTHROPIC_MODEL",
        "claude-haiku-4-5-20251001",
    )

    if not anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found in .env"
        )

    claude_model = LiteLlm(
        model=f"anthropic/{anthropic_model}"
    )

    remote_network_agent = RemoteA2aAgent(
        name="network_diagnostics_remote",
        description=(
            "Remote Network Diagnostics specialist for telecom "
            "tower and connectivity investigations."
        ),
        agent_card=(
            "http://localhost:8001/"
            ".well-known/agent-card.json"
        ),
    )

    proxy_agent = Agent(
        name="network_diagnostics_proxy",
        model=claude_model,
        description=(
            "Proxy agent used to communicate with the remote "
            "Network Diagnostics A2A service."
        ),
        instruction="""
You are a proxy for the remote Network Diagnostics agent.

For questions about network towers, tower performance,
connectivity problems, active network incidents, or regional
network health, delegate the request to the
network_diagnostics_remote agent.

Do not invent operational network information.

Do not answer operational network questions from your own
knowledge.

The remote Network Diagnostics agent is the source of truth
for operational network data.
""",
        sub_agents=[
            remote_network_agent,
        ],
    )

    runner = InMemoryRunner(
        agent=proxy_agent,
        app_name="network_a2a_test",
    )

    print(
        "Sending request to Network Diagnostics "
        "A2A service..."
    )
    print()

    await runner.run_debug(
        "What is the current status of tower TX-512?"
    )


if __name__ == "__main__":
    asyncio.run(main())