import sys
import time
from pathlib import Path

import httpx
import streamlit as st

# Make the project root importable so "orchestration" can be found
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from orchestration.graph import run_ai_operations_center


# Agent card URLs used to check whether the A2A services are running
NETWORK_AGENT_CARD = "http://localhost:8001/.well-known/agent-card.json"
BILLING_AGENT_CARD = "http://localhost:8002/.well-known/agent-card.json"

# Ready-made inquiries shown as buttons in the sidebar
SAMPLE_QUERIES = {
    "Policy: roaming in Japan": (
        "What are the roaming charges in Japan?"
    ),
    "Analytics: critical outages": (
        "Which region had the most CRITICAL network outages?"
    ),
    "Diagnostics: tower TX-512": (
        "What is the current status of tower TX-512? "
        "Include its latest performance metrics and any open incidents."
    ),
    "Billing: duplicate charge": (
        "Customer CUST-10002 says they were charged twice for "
        "Unlimited Plus in September 2026. Investigate and resolve it."
    ),
    "SLA: Midwest outage": (
        "A CRITICAL network outage in the Midwest on September 12, "
        "2026 lasted 6 hours and affected 8,500 customers. "
        "Analyze the outage and determine whether it met the SLA."
    ),
}

st.set_page_config(
    page_title="Telecom AI Operations Hub",
    page_icon="🛰️",
    layout="wide",
)


def check_service(url: str) -> bool:
    """Returns True if the service at the given URL answers with HTTP 200."""
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def truncate_output(text: str, limit: int = 1000) -> str:
    """Shortens long agent output so the trace stays readable."""
    if not text:
        return "No output returned."

    text = str(text).strip()

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "\n\n..."


def set_query(text: str) -> None:
    """Button callback: puts a sample inquiry into the text box."""
    st.session_state["user_query"] = text


def clear_all() -> None:
    """Button callback: clears the text box and any previous result."""
    st.session_state["user_query"] = ""
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_query", None)
    st.session_state.pop("latest_elapsed", None)


def display_execution_trace(trace: list) -> None:
    """Shows each agent that ran, with its output, in expandable boxes."""
    st.subheader("Agent Execution Trace")

    if not trace:
        st.info("No agent execution trace is available for this request.")
        return

    for index, item in enumerate(trace, start=1):
        worker = item.get("worker", "Unknown Worker")
        output = item.get("output", "")

        with st.expander(f"Step {index}: {worker}", expanded=False):
            st.text(truncate_output(output))


def display_sidebar() -> None:
    """Sidebar: service status, sample inquiries and framework overview."""
    with st.sidebar:
        st.title("System Status")

        st.subheader("Remote Agent Services")

        if check_service(NETWORK_AGENT_CARD):
            st.success("Network Diagnostics: Online")
        else:
            st.error("Network Diagnostics: Offline")

        if check_service(BILLING_AGENT_CARD):
            st.success("Billing Resolution: Online")
        else:
            st.error("Billing Resolution: Offline")

        st.divider()

        st.subheader("Try a Sample Inquiry")

        for label, query in SAMPLE_QUERIES.items():
            st.button(
                label,
                key=f"sample_{label}",
                on_click=set_query,
                args=(query,),
                use_container_width=True,
            )

        st.divider()

        st.subheader("Framework Map")

        st.markdown(
            """
| Role | Framework |
| --- | --- |
| Supervisor | LangGraph |
| Policy and FAQ | LlamaIndex RAG |
| Network analytics | LlamaIndex SQL |
| Tower diagnostics | Google ADK + A2A |
| Billing resolution | Google ADK + A2A |
| Customer reply | CrewAI |
"""
        )

        st.caption(
            "Operational data is stored in SQLite. Specialist agents "
            "are coordinated by a LangGraph supervisor."
        )


display_sidebar()

st.title("🛰️ Telecom AI Operations Hub")

st.caption(
    "A multi-agent assistant for policy answers, network analytics, "
    "tower diagnostics, billing resolution and customer replies."
)

st.divider()

st.subheader("Customer Inquiry")

# The text box is tied to session state so sample buttons can fill it in
user_query = st.text_area(
    "Describe the customer's question or the operational request:",
    key="user_query",
    height=150,
    placeholder="Example: What is the current status of tower TX-512?",
)

submit_col, clear_col = st.columns([4, 1])

with submit_col:
    submit = st.button(
        "Submit Inquiry",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    st.button(
        "Clear",
        on_click=clear_all,
        use_container_width=True,
    )

if submit:
    if not user_query.strip():
        st.warning("Please enter an inquiry before submitting.")

    else:
        started = time.perf_counter()

        with st.spinner("Routing the inquiry to the right specialists..."):
            try:
                result = run_ai_operations_center(user_query.strip())

            except Exception as error:
                # Show the error in the page instead of crashing the app
                result = {
                    "final_response": "",
                    "execution_trace": [],
                    "agent_context": "",
                    "completed_workers": [],
                    "error": f"{type(error).__name__}: {error}",
                }

        st.session_state["latest_result"] = result
        st.session_state["latest_query"] = user_query.strip()
        st.session_state["latest_elapsed"] = time.perf_counter() - started

result = st.session_state.get("latest_result")

if result is not None:
    st.divider()

    error = result.get("error", "")
    final_response = result.get("final_response", "")
    trace = result.get("execution_trace", [])
    completed_workers = result.get("completed_workers", [])
    elapsed = st.session_state.get("latest_elapsed")

    if error:
        st.error(f"Workflow warning: {error}")

    # Quick summary of the run
    metric_a, metric_b, metric_c = st.columns(3)

    metric_a.metric("Agents Used", len(completed_workers))
    metric_b.metric(
        "Response Time",
        f"{elapsed:.1f} s" if elapsed is not None else "n/a",
    )
    metric_c.metric("Trace Steps", len(trace))

    st.subheader("Final Response")

    if final_response:
        st.markdown(final_response)

        st.download_button(
            "Download Response (.txt)",
            data=final_response,
            file_name="customer_response.txt",
            mime="text/plain",
        )
    else:
        st.warning(
            "The workflow finished without returning a final response."
        )

    st.divider()

    display_execution_trace(trace)

    if completed_workers:
        st.divider()
        st.caption("Workflow path: " + " → ".join(completed_workers))