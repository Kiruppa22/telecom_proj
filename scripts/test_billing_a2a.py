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

    remote_billing_agent = RemoteA2aAgent(
        name="billing_resolution_remote",
        description=(
            "Remote Billing Resolution specialist for "
            "telecom billing investigations and credits."
        ),
        agent_card=(
            "http://localhost:8002/"
            ".well-known/agent-card.json"
        ),
    )

    proxy_agent = Agent(
        name="billing_resolution_proxy",

        model=claude_model,

        description=(
            "Proxy agent used to communicate with the "
            "remote Billing Resolution A2A service."
        ),

        instruction="""
You are a proxy for the remote Billing Resolution agent.

For questions about customer billing accounts, duplicate
charges, billing disputes, billing credits, current
balances, or credit approval status, delegate the request
to the billing_resolution_remote agent.

Do not invent billing information.

Do not answer operational billing questions from your own
knowledge.

The remote Billing Resolution agent and its SQLite-backed
tools are the source of truth for operational billing data.

Report APPLIED and PENDING_APPROVAL credits accurately.

A PENDING_APPROVAL credit has not yet changed the customer
balance.
""",

        sub_agents=[
            remote_billing_agent
        ],
    )

    runner = InMemoryRunner(
        agent=proxy_agent,
        app_name="billing_a2a_test",
    )

    print(
        "Sending request to Billing Resolution "
        "A2A service..."
    )

    print()

    await runner.run_debug(
        (
            "Customer CUST-10002 says they were charged "
            "twice for Unlimited Plus in September 2026. "
            "Investigate the duplicate charge and resolve "
            "it according to the billing credit policy. "
            "Tell me the current balance and whether any "
            "credit is applied or pending approval."
        )
    )


if __name__ == "__main__":
    asyncio.run(main())