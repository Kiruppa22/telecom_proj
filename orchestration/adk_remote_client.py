
import asyncio
import os
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

NETWORK_A2A_URL = (
    "http://localhost:8001/.well-known/agent-card.json"
)

BILLING_A2A_URL = (
    "http://localhost:8002/.well-known/agent-card.json"
)


def create_remote_runner(agent_type: str):
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY was not found in .env"
        )

    if agent_type == "network":
        remote_name = "network_diagnostics_remote"
        proxy_name = "network_diagnostics_proxy"
        agent_card_url = NETWORK_A2A_URL

        description = (
            "Remote Network Diagnostics specialist for "
            "tower status, network incidents, connectivity "
            "diagnostics, and regional network health."
        )

        instruction = """
You are a proxy for the remote Network Diagnostics agent.

Delegate all operational network questions to the
network_diagnostics_remote agent.

Do not answer network questions using your own knowledge.

The remote agent's SQLite-backed tools are the source
of truth.

Return the specialist's findings accurately.

Distinguish confirmed network facts from hypotheses.
"""

    elif agent_type == "billing":
        remote_name = "billing_resolution_remote"
        proxy_name = "billing_resolution_proxy"
        agent_card_url = BILLING_A2A_URL

        description = (
            "Remote Billing Resolution specialist for "
            "customer accounts, duplicate charges, "
            "billing disputes, and credits."
        )

        instruction = """
You are a proxy for the remote Billing Resolution agent.

Delegate all operational billing questions to the
billing_resolution_remote agent.

Do not invent customer account information.

The remote agent's SQLite-backed tools are the source
of truth.

Report APPLIED and PENDING_APPROVAL credits accurately.

A PENDING_APPROVAL credit has not yet reduced the
customer's balance.

Do not create unnecessary additional billing credits.
"""

    else:
        raise ValueError(
            f"Unsupported agent type: {agent_type}"
        )

    claude_model = LiteLlm(
        model=f"anthropic/{ANTHROPIC_MODEL}"
    )

    remote_agent = RemoteA2aAgent(
        name=remote_name,
        description=description,
        agent_card=agent_card_url,
    )

    proxy_agent = Agent(
        name=proxy_name,
        model=claude_model,
        description=(
            f"Proxy for {remote_name} via A2A."
        ),
        instruction=instruction,
        sub_agents=[remote_agent],
    )

    runner = InMemoryRunner(
        agent=proxy_agent,
        app_name=f"{agent_type}_remote_client",
    )

    return runner


async def check_service_availability(
    agent_type: str,
) -> dict:
    if agent_type == "network":
        url = NETWORK_A2A_URL
    elif agent_type == "billing":
        url = BILLING_A2A_URL
    else:
        return {
            "available": False,
            "error": f"Unknown agent type: {agent_type}",
        }

    try:
        async with httpx.AsyncClient(
            timeout=5.0
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            agent_card = response.json()

        return {
            "available": True,
            "agent_name": agent_card.get("name"),
            "agent_card_url": url,
        }

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        ValueError,
    ) as error:
        return {
            "available": False,
            "agent_card_url": url,
            "error": str(error),
        }


async def invoke_remote_agent_async(
    agent_type: str,
    question: str,
) -> dict:
    if not question or not question.strip():
        return {
            "success": False,
            "agent_type": agent_type,
            "response": "",
            "error": "Question cannot be empty.",
        }

    availability = await check_service_availability(
        agent_type
    )

    if not availability["available"]:
        return {
            "success": False,
            "agent_type": agent_type,
            "response": "",
            "error": (
                f"{agent_type.title()} A2A service "
                f"is unavailable: "
                f"{availability.get('error')}"
            ),
        }

    try:
        runner = create_remote_runner(agent_type)

        user_id = "langgraph_supervisor"
        session_id = str(uuid.uuid4())

        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        user_message = types.Content(
            role="user",
            parts=[
                types.Part(text=question)
            ],
        )

        final_response = ""
        collected_responses = []

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=user_message,
        ):
            if not event.content:
                continue

            if not event.content.parts:
                continue

            text_parts = [
                part.text
                for part in event.content.parts
                if getattr(part, "text", None)
            ]

            if not text_parts:
                continue

            response_text = "\n".join(text_parts)

            if event.is_final_response():
                final_response = response_text

            if event.author == (
                "network_diagnostics_remote"
                if agent_type == "network"
                else "billing_resolution_remote"
            ):
                collected_responses.append(
                    response_text
                )

        if not final_response and collected_responses:
            final_response = collected_responses[-1]

        if not final_response:
            return {
                "success": False,
                "agent_type": agent_type,
                "response": "",
                "error": (
                    "The remote agent returned no "
                    "text response."
                ),
            }

        return {
            "success": True,
            "agent_type": agent_type,
            "agent_name": availability.get(
                "agent_name"
            ),
            "response": final_response,
            "error": None,
        }

    except Exception as error:
        return {
            "success": False,
            "agent_type": agent_type,
            "response": "",
            "error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def invoke_remote_agent(
    agent_type: str,
    question: str,
) -> dict:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            invoke_remote_agent_async(
                agent_type,
                question,
            )
        )

    raise RuntimeError(
        "invoke_remote_agent() cannot run inside "
        "an active event loop. Use "
        "invoke_remote_agent_async() instead."
    )


def ask_network_diagnostics(question: str) -> dict:
    return invoke_remote_agent(
        "network",
        question,
    )


def ask_billing_resolution(question: str) -> dict:
    return invoke_remote_agent(
        "billing",
        question,
    )
