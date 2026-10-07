# Telecom AI Operations Hub

An AI assistant for telecom support teams. Ask a question in plain English and the system works out which specialist agents are needed, gathers verified facts from policy documents and a live database, and writes a customer-ready reply.

> **Credits:** Based on a project by `<friend's GitHub username>`, used and modified with their permission.

---

## Features

- **Smart routing:** a LangGraph supervisor decides which specialist runs next, one at a time, and never repeats a specialist.
- **Policy answers:** searches policy and FAQ documents (roaming, SLA rules, billing disputes, device upgrades, 5G).
- **Network analytics:** turns questions like "Which region had the most critical outages?" into SQL and answers from real data.
- **Tower diagnostics:** a separate agent service reports tower status, performance metrics and open incidents.
- **Billing resolution:** a separate agent service investigates duplicate charges and proposes credits.
- **Grounded replies:** a writer and a reviewer agent produce the final message and remove anything not backed by the gathered facts.
- **Transparent results:** the interface shows the final reply, the agents that ran, their outputs, and the response time.

## How a Request Flows

```text
1. User submits an inquiry in the Streamlit app
2. Supervisor reads it and picks the first specialist
3. Specialist runs and adds its findings to a shared context
4. Supervisor picks the next specialist, or finishes if none are needed
5. Communications Specialist drafts a reply from the findings
6. Quality Reviewer removes any unsupported claims
7. App shows the final reply and the agent trace
```

## Tech Stack

| Layer                  | Technology                         |
| ---------------------- | ---------------------------------- |
| Interface              | Streamlit                          |
| Workflow orchestration | LangGraph                          |
| Policy document search | LlamaIndex (vector index)          |
| Text-to-SQL analytics  | LlamaIndex + SQLAlchemy            |
| Remote agents          | Google ADK with A2A protocol       |
| Reply writing          | CrewAI                             |
| Database               | SQLite                             |
| Language model         | Anthropic Claude                   |
| Embeddings             | all-MiniLM-L6-v2                   |

## Folder Overview

| Folder / file     | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| `ui/`             | Streamlit application (`app.py`)                     |
| `orchestration/`  | Supervisor graph, shared state, CrewAI nodes         |
| `llamaindex_rag/` | Policy document index and semantic SQL search        |
| `adk-services/`   | Network diagnostics and billing resolution agents    |
| `data/`           | Policy documents, vector index, SQLite database      |
| `sql/`            | Database schema and seed scripts                     |
| `scripts/`        | Setup, validation and scenario test scripts          |

## Getting Started

### 1. Prerequisites
- Python 3.11
- An Anthropic API key

### 2. Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure
Copy `.env.example` to `.env` and fill in your values:
```text
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```
Never commit your `.env` file.

### 4. Prepare the data
```powershell
python scripts/setup_database.py
python scripts/validate_database.py
python -m llamaindex_rag.document_rag
```

### 5. Start everything
Use three separate terminals.

```powershell
# Terminal 1: network diagnostics agent
cd "adk-services\network_diagnostics"
uvicorn agent:a2a_app --host 0.0.0.0 --port 8001
```
```powershell
# Terminal 2: billing resolution agent
cd "adk-services\billing_resolution"
uvicorn agent:a2a_app --host 0.0.0.0 --port 8002
```
```powershell
# Terminal 3: web interface (from the project root)
streamlit run ui/app.py
```

The sidebar shows whether both agent services are online.

## Example Inquiries

| Category          | Try asking                                                                                          | Specialists used                  |
| ----------------- | --------------------------------------------------------------------------------------------------- | --------------------------------- |
| Policy            | What are the roaming charges in Japan?                                                              | Policy                            |
| Analytics         | Which region had the most CRITICAL network outages?                                                 | Network analytics                 |
| Tower status      | What is the current status of tower TX-512? Include its latest performance metrics and open incidents. | Network diagnostics            |
| Billing           | Customer CUST-10002 says they were charged twice for Unlimited Plus in September 2026. Investigate and resolve it. | Billing resolution |
| SLA analysis      | A CRITICAL outage in the Midwest on September 12, 2026 lasted 6 hours and affected 8,500 customers. Did it meet the SLA? | Network analytics, then Policy |

The sidebar in the app has one-click buttons for each of these.

## Testing

```powershell
python -m scripts.test_langgraph_scenarios
```
Runs the five scenarios above and reports how many passed.

## Troubleshooting

| Problem | What to check |
| ------- | ------------- |
| Sidebar shows a service as offline | Make sure its uvicorn terminal is running on port 8001 or 8002 |
| `ANTHROPIC_API_KEY was not found` | Check that `.env` exists in the project root and has the key |
| Policy answers fail or are empty | Rebuild the index with `python -m llamaindex_rag.document_rag` |
| `SQLite database not found` | Run `python scripts/setup_database.py` |
| First question is slow | The embedding model loads on first use; later questions are faster |

## Design Notes

- Specialists only report facts; the final wording is written separately, so the customer reply never includes invented policies, promises or diagnoses.
- Credits that are pending approval are described as pending, never as already applied.
- The supervisor has a loop limit so a routing mistake cannot run forever.
- The SQL query engine is created once and reused, which avoids reloading the embedding model for each question.

## Future Improvements

- Chat history so follow-up questions keep their context
- Charts for outage and performance data in the interface
- Authentication and role-based access for billing actions
- Docker setup to start all three services with one command
- Automated tests that run without live API calls
