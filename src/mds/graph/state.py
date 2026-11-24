from typing import Literal, TypedDict, Annotated
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import BaseMessage, SystemMessage

class AgentState(MessagesState):
    approval_status: Literal["pending", "approved", "rejected"] | None

class AgentContext(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_number: str
    user_id: str
    department: str
    approval_status: Literal["pending", "approved", "rejected"] | None
