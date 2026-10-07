import sys
from pathlib import Path

import httpx
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )

from orchestration.graph import (
    run_ai_operations_center,
)


NETWORK_AGENT_CARD = (
    "http://localhost:8001/"
    ".well-known/agent-card.json"
)

BILLING_AGENT_CARD = (
    "http://localhost:8002/"
    ".well-known/agent-card.json"
)


st.set_page_config(
    page_title="Prodapt AI Operations Center",
    page_icon="📡",
    layout="wide",
)


def check_service(
    url: str,
) -> bool:
    try:
        response = httpx.get(
            url,
            timeout=2.0,
        )

        return (
            response.status_code == 200
        )

    except Exception:
        return False


def truncate_output(
    text: str,
    limit: int = 700,
) -> str:
    if not text:
        return "No output returned."

    text = str(text).strip()

    if len(text) <= limit:
        return text

    return (
        text[:limit].rstrip()
        + "\n\n..."
    )


def display_execution_trace(
    trace: list,
):
    st.subheader(
        "Agent Execution Trace"
    )

    if not trace:
        st.info(
            "No agent execution trace is "
            "available for this request."
        )
        return

    for index, item in enumerate(
        trace,
        start=1,
    ):
        worker = item.get(
            "worker",
            "Unknown Worker",
        )

        output = item.get(
            "output",
            "",
        )

        with st.expander(
            f"{index}. {worker}",
            expanded=True,
        ):
            st.text(
                truncate_output(output)
            )


def display_sidebar():
    with st.sidebar:
        st.title(
            "System Status"
        )

        network_online = check_service(
            NETWORK_AGENT_CARD
        )

        billing_online = check_service(
            BILLING_AGENT_CARD
        )

        st.subheader(
            "A2A Services"
        )

        if network_online:
            st.success(
                "Network Diagnostics — Online"
            )
        else:
            st.error(
                "Network Diagnostics — Offline"
            )

        if billing_online:
            st.success(
                "Billing Resolution — Online"
            )
        else:
            st.error(
                "Billing Resolution — Offline"
            )

        st.divider()

        st.subheader(
            "Framework Map"
        )

        st.markdown(
            """
**Supervisor**
- LangGraph

**Policy & FAQ**
- LlamaIndex Document RAG

**Network Analytics**
- LlamaIndex Semantic SQL

**Network Diagnostics**
- Google ADK
- A2A

**Billing Resolution**
- Google ADK
- A2A

**Customer Communications**
- CrewAI
"""
        )

        st.divider()

        st.caption(
            "Operational data is backed by "
            "SQLite. Specialist workers are "
            "orchestrated through LangGraph."
        )


display_sidebar()


st.title(
    "📡 Prodapt AI Operations Center"
)

st.caption(
    "Multi-agent telecom operations assistant "
    "for policy, network analytics, diagnostics, "
    "billing resolution, and customer communications."
)

st.divider()


st.subheader(
    "Customer Inquiry"
)

user_query = st.text_area(
    "Enter the customer's question or operational request:",
    height=150,
    placeholder=(
        "Example: What is the current status "
        "of tower TX-512?"
    ),
)


submit = st.button(
    "Submit Inquiry",
    type="primary",
    use_container_width=True,
)


if submit:
    if not user_query.strip():
        st.warning(
            "Please enter an inquiry before submitting."
        )

    else:
        with st.spinner(
            "Analyzing the inquiry and coordinating "
            "specialist agents..."
        ):
            try:
                result = (
                    run_ai_operations_center(
                        user_query.strip()
                    )
                )

                st.session_state[
                    "latest_result"
                ] = result

                st.session_state[
                    "latest_query"
                ] = user_query.strip()

            except Exception as error:
                st.session_state[
                    "latest_result"
                ] = {
                    "final_response": "",
                    "execution_trace": [],
                    "agent_context": "",
                    "completed_workers": [],
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }

                st.session_state[
                    "latest_query"
                ] = user_query.strip()


result = st.session_state.get(
    "latest_result"
)


if result is not None:
    st.divider()

    error = result.get(
        "error",
        "",
    )

    final_response = result.get(
        "final_response",
        "",
    )

    trace = result.get(
        "execution_trace",
        [],
    )

    if error:
        st.error(
            f"Workflow warning: {error}"
        )

    st.subheader(
        "Final Response"
    )

    if final_response:
        st.markdown(
            final_response
        )
    else:
        st.warning(
            "The workflow completed without "
            "returning a final response."
        )

    st.divider()

    display_execution_trace(
        trace
    )

    completed_workers = result.get(
        "completed_workers",
        [],
    )

    if completed_workers:
        st.divider()

        st.caption(
            "Completed workflow: "
            + " → ".join(
                completed_workers
            )
        )