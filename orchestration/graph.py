import json
import os
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from llamaindex_rag.policy_rag_worker import ask_policy_question
from llamaindex_rag.sql_semantic_search import ask_network_analytics
from orchestration.adk_remote_client import (
    ask_billing_resolution,
    ask_network_diagnostics,
)
from orchestration.crew_nodes import run_customer_comms
from orchestration.state import AgentState


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY was not found in .env")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Settings for the supervisor's routing call
SUPERVISOR_MAX_TOKENS = 100
SUPERVISOR_TEMPERATURE = 0

# Safety limit so a routing mistake can never loop forever
RECURSION_LIMIT = 20


# Internal worker name -> name shown in the UI and execution trace
WORKER_DISPLAY_NAMES = {
    "policy_rag": "PolicyRAG",
    "network_analytics": "NetworkAnalytics",
    "network_diagnostics_adk": "NetworkDiagnosticsADK",
    "billing_resolution_adk": "BillingResolutionADK",
    "customer_comms_crew": "CustomerCommsCrew",
}

# Workers that gather facts (the comms crew is always the final step)
SPECIALIST_WORKERS = {
    "policy_rag",
    "network_analytics",
    "network_diagnostics_adk",
    "billing_resolution_adk",
}


# ---------------------------------------------------------------
# Helper functions for the shared state
# ---------------------------------------------------------------

def append_context(existing_context: str, worker_name: str, output: str) -> str:
    """Adds a worker's output to the shared context as a labeled section."""
    section = f"\n\n--- {worker_name} ---\n{output.strip()}"

    if not existing_context:
        return section.strip()

    return existing_context.rstrip() + section


def append_trace(
    state: AgentState,
    worker_name: str,
    output: str,
) -> list[dict[str, str]]:
    """Returns the execution trace with one more step added."""
    trace = list(state.get("execution_trace", []))
    trace.append({"worker": worker_name, "output": output.strip()})
    return trace


def add_completed_worker(state: AgentState, worker: str) -> list[str]:
    """Returns the completed-workers list with this worker included once."""
    completed = list(state.get("completed_workers", []))

    if worker not in completed:
        completed.append(worker)

    return completed


def build_worker_update(state: AgentState, worker: str, output: str) -> dict:
    """Builds the state update returned by every specialist worker."""
    display_name = WORKER_DISPLAY_NAMES[worker]

    return {
        "messages": [f"{display_name}: {output}"],
        "agent_context": append_context(
            state.get("agent_context", ""),
            display_name,
            output,
        ),
        "completed_workers": add_completed_worker(state, worker),
        "execution_trace": append_trace(state, display_name, output),
    }


def extract_json_object(text: str) -> dict:
    """Pulls a JSON object out of the model's reply (handles ``` fences)."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1)
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Supervisor did not return a JSON object.")

    return json.loads(cleaned[start:end + 1])


# ---------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------

def call_supervisor_llm(state: AgentState) -> str:
    """Asks the LLM which worker should run next and returns its name."""
    user_query = state.get("user_query", "")
    completed = state.get("completed_workers", [])
    context = state.get("agent_context", "")

    prompt = f"""
You are the Supervisor for a telecom AI Operations
Center.

Your job is to decide which specialist worker should
run NEXT.

ORIGINAL USER QUERY:
{user_query}

ALREADY COMPLETED SPECIALISTS:
{completed}

CURRENT VERIFIED CONTEXT:
{context if context else "No specialist context yet."}

Available specialist workers:

1. policy_rag
Use for:
- telecom policies
- roaming policies and charges
- SLA rules
- outage procedures
- billing dispute policies
- device upgrade policies
- 5G FAQ information

2. network_analytics
Use for:
- aggregate operational network analysis
- outage counts
- outage severity
- regions with the most outages
- historical network analytics
- cross-tower comparisons
- database analytics

3. network_diagnostics_adk
Use for:
- a specific tower
- tower status
- tower performance
- packet loss
- latency
- throughput
- signal strength
- active network incidents
- regional tower health
- connectivity diagnostics

4. billing_resolution_adk
Use for:
- a specific customer account
- duplicate charges
- billing account investigation
- billing credits
- billing disputes
- resolving duplicate billing charges

5. customer_comms_crew
Use ONLY when all specialists required to answer the
user's query have already completed.

Important routing rules:

- Never run the same specialist twice.
- Choose only ONE next worker.
- customer_comms_crew MUST be the final worker.
- Do not choose customer_comms_crew if another
  specialist is still needed.
- If the question requires multiple specialist domains,
  route them one at a time.

Examples:

"What are the roaming charges in Japan?"
policy_rag -> customer_comms_crew

"Which region had the most CRITICAL outages?"
network_analytics -> customer_comms_crew

"What is happening with tower TX-512?"
network_diagnostics_adk -> customer_comms_crew

"Customer CUST-10002 was charged twice."
billing_resolution_adk -> customer_comms_crew

"A CRITICAL outage in the Midwest lasted 6 hours.
Did it violate SLA?"
network_analytics -> policy_rag ->
customer_comms_crew

Return ONLY valid JSON in exactly this form:

{{
  "next": "worker_name"
}}

worker_name must be exactly one of:

policy_rag
network_analytics
network_diagnostics_adk
billing_resolution_adk
customer_comms_crew
"""

    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=SUPERVISOR_MAX_TOKENS,
        temperature=SUPERVISOR_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]

    response_text = "\n".join(text_parts).strip()
    data = extract_json_object(response_text)
    next_worker = data.get("next")

    allowed = SPECIALIST_WORKERS | {"customer_comms_crew"}

    if next_worker not in allowed:
        raise ValueError(
            f"Supervisor selected an invalid worker: {next_worker}"
        )

    return next_worker


def supervisor_node(state: AgentState) -> dict:
    """Graph node: decides the next worker, with safe fallbacks."""
    try:
        next_worker = call_supervisor_llm(state)
        completed = state.get("completed_workers", [])

        # Guard: never run the same specialist twice
        if next_worker in completed and next_worker != "customer_comms_crew":
            next_worker = "customer_comms_crew"

        return {"next": next_worker}

    except Exception as error:
        # If routing fails, go straight to the final reply and record why
        error_message = (
            f"Supervisor routing failed: {type(error).__name__}: {error}"
        )

        return {
            "next": "customer_comms_crew",
            "error": error_message,
            "agent_context": append_context(
                state.get("agent_context", ""),
                "Supervisor Error",
                error_message,
            ),
        }


# ---------------------------------------------------------------
# Specialist worker nodes
# ---------------------------------------------------------------

def policy_rag_node(state: AgentState) -> dict:
    """Answers policy, SLA and FAQ questions from the policy documents."""
    try:
        result = ask_policy_question(state["user_query"])

        if isinstance(result, dict):
            output = str(result.get("answer", result))
        else:
            output = str(result)

    except Exception as error:
        output = f"Policy RAG failed: {type(error).__name__}: {error}"

    return build_worker_update(state, "policy_rag", output)


def network_analytics_node(state: AgentState) -> dict:
    """Answers aggregate network questions by querying the database."""
    try:
        output = str(ask_network_analytics(state["user_query"]))

    except Exception as error:
        output = (
            f"Network analytics failed: {type(error).__name__}: {error}"
        )

    return build_worker_update(state, "network_analytics", output)


def network_diagnostics_node(state: AgentState) -> dict:
    """Calls the remote network diagnostics agent (A2A, port 8001)."""
    try:
        result = ask_network_diagnostics(state["user_query"])

        if result.get("success"):
            output = result["response"]
        else:
            output = (
                "Network Diagnostics service could not complete the "
                f"request. {result.get('error')}"
            )

    except Exception as error:
        output = (
            f"Network Diagnostics failed: {type(error).__name__}: {error}"
        )

    return build_worker_update(state, "network_diagnostics_adk", output)


def billing_resolution_node(state: AgentState) -> dict:
    """Calls the remote billing resolution agent (A2A, port 8002)."""
    try:
        result = ask_billing_resolution(state["user_query"])

        if result.get("success"):
            output = result["response"]
        else:
            output = (
                "Billing Resolution service could not complete the "
                f"request. {result.get('error')}"
            )

    except Exception as error:
        output = (
            f"Billing Resolution failed: {type(error).__name__}: {error}"
        )

    return build_worker_update(state, "billing_resolution_adk", output)


def customer_comms_node(state: AgentState) -> dict:
    """Final step: turns the gathered findings into the customer reply."""
    worker = "customer_comms_crew"
    display_name = WORKER_DISPLAY_NAMES[worker]

    try:
        output = run_customer_comms(
            state["user_query"],
            state.get("agent_context", ""),
        )

    except Exception as error:
        output = (
            "Customer communications failed: "
            f"{type(error).__name__}: {error}"
        )

    return {
        "messages": [f"{display_name}: {output}"],
        "final_response": output,
        "completed_workers": add_completed_worker(state, worker),
        "execution_trace": append_trace(state, display_name, output),
    }


def route_from_supervisor(
    state: AgentState,
) -> Literal[
    "policy_rag",
    "network_analytics",
    "network_diagnostics_adk",
    "billing_resolution_adk",
    "customer_comms_crew",
]:
    """Tells LangGraph which node to run after the supervisor."""
    return state["next"]


# ---------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("policy_rag", policy_rag_node)
builder.add_node("network_analytics", network_analytics_node)
builder.add_node("network_diagnostics_adk", network_diagnostics_node)
builder.add_node("billing_resolution_adk", billing_resolution_node)
builder.add_node("customer_comms_crew", customer_comms_node)

# Every request starts at the supervisor
builder.add_edge(START, "supervisor")

# The supervisor chooses which worker runs next
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "policy_rag": "policy_rag",
        "network_analytics": "network_analytics",
        "network_diagnostics_adk": "network_diagnostics_adk",
        "billing_resolution_adk": "billing_resolution_adk",
        "customer_comms_crew": "customer_comms_crew",
    },
)

# After each specialist finishes, control returns to the supervisor
builder.add_edge("policy_rag", "supervisor")
builder.add_edge("network_analytics", "supervisor")
builder.add_edge("network_diagnostics_adk", "supervisor")
builder.add_edge("billing_resolution_adk", "supervisor")

# The communications crew is always the last step
builder.add_edge("customer_comms_crew", END)

graph = builder.compile()


def run_ai_operations_center(user_query: str) -> dict:
    """Runs the full workflow for one inquiry and returns the results."""
    initial_state: AgentState = {
        "messages": [],
        "next": "",
        "user_query": user_query,
        "agent_context": "",
        "completed_workers": [],
        "execution_trace": [],
        "final_response": "",
        "error": "",
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": RECURSION_LIMIT},
    )

    return {
        "final_response": result.get("final_response", ""),
        "execution_trace": result.get("execution_trace", []),
        "agent_context": result.get("agent_context", ""),
        "completed_workers": result.get("completed_workers", []),
        "error": result.get("error", ""),
    }