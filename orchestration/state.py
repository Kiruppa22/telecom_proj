from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[str], add]

    next: str

    user_query: str

    agent_context: str

    completed_workers: list[str]

    execution_trace: list[dict[str, str]]

    final_response: str

    error: str