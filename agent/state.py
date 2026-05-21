# agent/state.py
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    # Conversation history — operator.add means messages accumulate, not overwrite
    messages: Annotated[List[BaseMessage], operator.add]

    # Data context
    csv_path: Optional[str]
    df_shape: Optional[tuple]        # (rows, cols)
    df_columns: Optional[List[str]]  # column names injected into system prompt
    df_dtypes: Optional[dict]        # col → dtype

    # Execution state
    iteration_count: int
    last_code_executed: Optional[str]
    last_tool_output: Optional[str]
    plot_paths: List[str]            # all plots generated this session

    # Error tracking
    error_count: int
    last_error: Optional[str]
