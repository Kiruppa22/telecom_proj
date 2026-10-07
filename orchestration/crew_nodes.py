import os
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001",
)

# Low temperature keeps replies factual and consistent
CREW_TEMPERATURE = 0.1


if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY was not found in .env")


# Shared language model used by both crew agents
llm = LLM(
    model=f"anthropic/{ANTHROPIC_MODEL}",
    api_key=ANTHROPIC_API_KEY,
    temperature=CREW_TEMPERATURE,
)


# Agent 1: writes the first draft of the customer reply
communications_specialist = Agent(
    role="Customer Communications Specialist",
    goal=(
        "Turn verified telecom operations findings into a clear, "
        "accurate, professional, customer-ready reply without "
        "inventing any facts."
    ),
    backstory=(
        "You write customer replies for a telecom operations team. "
        "You receive verified findings from policy retrieval, network "
        "analytics, network diagnostics and billing resolution "
        "systems, and you explain them clearly to the customer. You "
        "never invent operational facts, policies, diagnoses, "
        "promises, actions or timelines that are not contained in the "
        "supplied findings."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


# Agent 2: checks the draft and removes anything unsupported
quality_reviewer = Agent(
    role="Quality Reviewer",
    goal=(
        "Review customer replies for factual accuracy, grounding, "
        "clarity, policy safety, billing safety and professional tone."
    ),
    backstory=(
        "You are the last check before a reply reaches the customer. "
        "You verify every statement against the supplied findings, "
        "remove unsupported assumptions, promises, judgments, "
        "diagnoses and invented next steps, keep verified facts "
        "exactly as they are, and return only the final reply."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
)


def run_customer_comms(query: str, agent_context: str) -> str:
    """Drafts and reviews the customer reply, returning the final text."""

    # Basic input checks
    if not query or not query.strip():
        return (
            "Unable to prepare a customer response "
            "because the original inquiry is empty."
        )

    if not agent_context or not agent_context.strip():
        return (
            "Unable to prepare a verified customer "
            "response because no specialist context "
            "was provided."
        )

    # Task 1: write the draft
    draft_task = Task(
        description="""
Prepare a professional customer-facing response using
ONLY the original customer inquiry and the verified
specialist context supplied below.

ORIGINAL CUSTOMER INQUIRY:

{query}

VERIFIED SPECIALIST CONTEXT:

{agent_context}

Follow these rules strictly:

1. Answer the customer's actual question directly.

2. Treat the specialist context as the only factual
   source for the response.

3. Do not invent facts, policies, account information,
   network information, incident details, diagnoses,
   thresholds, actions, timelines, or promises.

4. Preserve important identifiers exactly when they are
   relevant, including customer IDs, tower IDs, incident
   IDs, charge IDs, credit IDs, and outage IDs.

5. Preserve numeric facts exactly, including balances,
   charges, credits, dates, durations, customer counts,
   latency, packet loss, throughput, signal strength,
   and other operational metrics.

6. For billing cases, clearly distinguish between:
   - APPLIED credits
   - PENDING_APPROVAL credits

7. A PENDING_APPROVAL credit has NOT yet been applied.
   Do not reduce the current balance unless the
   specialist context explicitly confirms that the
   credit status is APPLIED and the balance was updated.

8. Do not promise that a pending credit will be approved,
   automatically applied, refunded, processed by a
   particular time, or followed by a notification unless
   the specialist context explicitly confirms this.

9. Do not tell the customer that no action is required
   unless the specialist context explicitly establishes
   that fact.

10. For network cases, distinguish verified observations
    from hypotheses.

11. Do not call a metric high, low, elevated, degraded,
    abnormal, acceptable, unacceptable, healthy, or
    unhealthy unless that classification is explicitly
    supported by the specialist context.

12. Do not invent performance thresholds or compare
    measurements against unspecified thresholds.

13. If the specialist context contains a suspected cause
    or possible diagnosis, clearly label it as suspected,
    possible, or a hypothesis. Never present it as a
    confirmed root cause.

14. Do not claim that a team is actively taking an action
    unless that action is explicitly supported by the
    specialist context.

15. For policy and SLA questions, state only the policy
    rules supported by the specialist context.

16. Do not invent compensation, credits, penalties,
    escalation procedures, incident reviews, preventive
    measures, or remediation activities.

17. Clearly distinguish:
    - current verified facts
    - historical facts
    - pending actions
    - possible future outcomes

18. If information needed to make a claim is missing,
    omit the claim rather than guessing.

19. Do not mention internal implementation details such
    as LangGraph, CrewAI, LlamaIndex, Google ADK, A2A,
    SQLite, SQL queries, vector stores, embeddings,
    agents, tools, or specialist context.

20. Use a professional and concise telecom customer
    service tone.

21. Do not add generic promises such as:
    - "we will notify you"
    - "we are working to resolve this"
    - "we will keep you updated"
    - "no action is required"
    unless explicitly supported by the specialist
    context.

Produce a polished draft response for the customer.
""",
        expected_output=(
            "A factual, grounded, professional "
            "customer-facing response based only on the "
            "provided specialist context."
        ),
        agent=communications_specialist,
    )

    # Task 2: review the draft and strip anything unsupported
    review_task = Task(
        description="""
Review the drafted customer response against BOTH the
original inquiry and the verified specialist context.

ORIGINAL CUSTOMER INQUIRY:

{query}

VERIFIED SPECIALIST CONTEXT:

{agent_context}

Perform a strict factual grounding review.

Check every statement in the draft against the
specialist context.

Apply all of the following rules:

1. Remove any statement that cannot be supported by the
   specialist context.

2. Preserve verified identifiers, dates, monetary
   amounts, balances, metrics, statuses, durations,
   customer counts, and policy values accurately.

3. Do not introduce new information during the review.

4. Do not invent operational thresholds.

5. Do not classify a network metric as acceptable,
   unacceptable, elevated, degraded, healthy, abnormal,
   poor, good, or otherwise qualitative unless that
   classification is explicitly present in the
   specialist context.

6. Do not convert a suspected cause or hypothesis into
   a confirmed diagnosis or root cause.

7. If a cause is described as suspected or possible,
   preserve that uncertainty explicitly.

8. Do not promise future notifications, follow-ups,
   approvals, refunds, timelines, investigations,
   preventive actions, repairs, remediation, or other
   future activities unless explicitly supported by the
   specialist context.

9. Do not tell the customer that no action is required
   unless the specialist context explicitly establishes
   that fact.

10. For billing responses, verify the exact credit
    status.

11. If a credit is PENDING_APPROVAL:
    - state that it is pending approval
    - state that it has not yet been applied
    - preserve the current balance
    - do not describe the balance as already reduced

12. Do not state that a pending credit will definitely
    be approved or automatically applied unless that is
    explicitly established by the specialist context.

13. For APPLIED credits, only state that the balance was
    reduced when the specialist context confirms the
    balance update.

14. Do not treat historical PAID charges as current
    OPEN charges.

15. Do not describe a legitimate charge as a duplicate
    merely because another charge has the same amount.

16. For network responses, preserve the difference
    between:
    - tower operational status
    - measured performance metrics
    - open incident status
    - suspected causes
    - confirmed root causes

17. For policy and SLA responses, use only policy rules
    explicitly supported by the specialist context.

18. Do not invent incident review procedures,
    compensation, SLA penalties, preventive measures,
    infrastructure improvements, or escalation actions.

19. Do not expose internal implementation details such
    as agents, CrewAI, LangGraph, LlamaIndex, Google ADK,
    A2A, SQL, SQLite, embeddings, vector retrieval, or
    internal specialist context.

20. Remove unnecessary speculation.

21. Keep the answer focused on the original inquiry.

22. Maintain a professional, clear, customer-ready tone.

23. If the specialist context does not contain enough
    information to support a statement, remove that
    statement rather than filling the gap with general
    knowledge.

24. Return ONLY the final customer-facing response.

Do not include:
- review notes
- critique
- scoring
- explanations of edits
- internal reasoning
- statements such as "the response has been reviewed"
""",
        expected_output=(
            "Only the final factual and customer-ready "
            "response, with all unsupported claims "
            "removed."
        ),
        agent=quality_reviewer,
        context=[draft_task],
    )

    # The two tasks run one after the other: draft first, then review
    crew = Crew(
        agents=[communications_specialist, quality_reviewer],
        tasks=[draft_task, review_task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff(
            inputs={
                "query": query.strip(),
                "agent_context": agent_context.strip(),
            }
        )

        # Newer CrewAI versions return an object with a .raw text field
        if hasattr(result, "raw"):
            final_response = result.raw
        else:
            final_response = str(result)

        if not final_response:
            return (
                "The communications workflow completed "
                "without producing a final response."
            )

        return str(final_response).strip()

    except Exception as error:
        return (
            "Customer communications could not be "
            f"completed. {type(error).__name__}: {error}"
        )