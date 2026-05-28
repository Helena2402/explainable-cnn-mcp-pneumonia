import asyncio
from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a medical AI assistant that cannot diagnose directly. "
        "You must use tools to analyse images.\n\n"
        "Rules:\n"
        "1. If the user asks about pneumonia or analysing an X-ray, "
        "you MUST call the tool `predict_pneumonia`.\n"
        "2. If the user asks why, where, or for explanation, "
        "you MUST also call `explain_pneumonia`.\n"
        "3. Never guess. Base your answer only on tool outputs.\n"
        "4. Explain Grad-CAM qualitatively and cautiously."
    ),
}



async def build_agent():
    """
    Build a LangGraph agent that can decide when to call:
    - predict_pneumonia
    - explain_pneumonia
    """

    # -------------------------
    # Connect to MCP server
    # -------------------------
    mcp_client = MultiServerMCPClient(
        {
            "pneumonia_server": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Discover available tools dynamically
    mcp_tools = await mcp_client.get_tools()

    # -------------------------
    # Local LLM (reasoning only)
    # -------------------------
    llm = ChatOllama(
        model="qwen2.5",
        temperature=0
    )

    model_with_tools = llm.bind_tools(mcp_tools)

    # -------------------------
    # Graph nodes
    # -------------------------
    def call_model(state: MessagesState):
        """
        Call the LLM. The LLM may decide to invoke tools.
        """
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        """
        Decide whether to continue to tool execution.
        """
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # -------------------------
    # Build LangGraph
    # -------------------------
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(mcp_tools))

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        ["tools", END],
    )
    builder.add_edge("tools", "call_model")

    return builder.compile()