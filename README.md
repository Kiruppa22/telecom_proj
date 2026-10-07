
# Prodapt AI Operations Center

## Overview

The **Prodapt AI Operations Center** is a multi-agent telecom operations application that handles policy questions, network analytics, tower diagnostics, billing disputes, and SLA analysis.

A **LangGraph Supervisor** routes each customer inquiry to the required specialist workers. The results are then passed to **CrewAI** to generate a customer-ready response. **Streamlit** provides the user interface and displays the agent execution trace.

## Technology Stack

| Component              | Technology       |
| ---------------------- | ---------------- |
| UI                     | Streamlit        |
| Orchestration          | LangGraph        |
| Document RAG           | LlamaIndex       |
| Semantic SQL           | LlamaIndex       |
| Network Diagnostics    | Google ADK + A2A |
| Billing Resolution     | Google ADK + A2A |
| Customer Communication | CrewAI           |
| Database               | SQLite           |
| LLM                    | Anthropic Claude |
| Embeddings             | all-MiniLM-L6-v2 |

## Architecture

```text
                    Streamlit UI
                         |
                         v
                LangGraph Supervisor
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   Policy RAG      Network Analytics   ADK Agents
   LlamaIndex         LlamaIndex       /       \
        |                |        Network     Billing
    Documents          SQLite      :8001       :8002
        |                |             \       /
        +----------------+--------------+-----+
                         |
                         v
                    CrewAI
          Communications Specialist
                         |
                  Quality Reviewer
                         |
                         v
                 Final Response
```

## Project Structure

```text
Project - 3/
├── adk-services/
│   ├── network_diagnostics/
│   └── billing_resolution/
├── data/
│   ├── documents/
│   ├── vector_index/
│   └── telecom_ops.db
├── llamaindex_rag/
├── orchestration/
├── scripts/
├── sql/
├── ui/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```text
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

Initialize the database:

```powershell
python scripts/setup_database.py
python scripts/validate_database.py
```

## Running the Application

### 1. Network Diagnostics Service

```powershell
cd "adk-services\network_diagnostics"
uvicorn agent:a2a_app --host 0.0.0.0 --port 8001
```

### 2. Billing Resolution Service

```powershell
cd "adk-services\billing_resolution"
uvicorn agent:a2a_app --host 0.0.0.0 --port 8002
```

### 3. Streamlit

From the project root:

```powershell
streamlit run ui/app.py
```

## Demo Scenarios

### Policy RAG

```text
What are the roaming charges in Japan?
```

Route:

```text
PolicyRAG → CustomerCommsCrew
```

### Network Analytics

```text
Which region had the most CRITICAL network outages?
```

Expected result: **Midwest — 6 outages**

Route:

```text
NetworkAnalytics → CustomerCommsCrew
```

### Network Diagnostics

```text
What is the current status of tower TX-512?
Include its latest performance metrics and any open incidents.
```

Route:

```text
NetworkDiagnosticsADK → CustomerCommsCrew
```

### Billing Resolution

```text
Customer CUST-10002 says they were charged twice for
Unlimited Plus in September 2026. Investigate and resolve it.
```

Expected result:

```text
Duplicate: CHG-50022
Credit: $65.99
Status: PENDING_APPROVAL
Balance: $131.98
```

Route:

```text
BillingResolutionADK → CustomerCommsCrew
```

### Multi-Agent SLA Analysis

```text
A CRITICAL network outage in the Midwest on September 12,
2026 lasted 6 hours and affected 8,500 customers.
Analyze the outage and determine whether it met the SLA.
```

Route:

```text
NetworkAnalytics → PolicyRAG → CustomerCommsCrew
```

## Testing

Run all required scenarios:

```powershell
python -m scripts.test_langgraph_scenarios
```

Final validated result:

```text
Passed: 5/5
```

## Final Status

The project successfully integrates **LlamaIndex, Google ADK, A2A, LangGraph, CrewAI, SQLite, and Streamlit** into an end-to-end telecom AI Operations Center.
