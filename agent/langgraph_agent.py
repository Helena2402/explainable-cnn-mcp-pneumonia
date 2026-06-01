from typing import Optional, List, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from mcp_server.server import predict_pneumonia, explain_pneumonia


# ============================================================
# State
# ============================================================

class AgentState(MessagesState):
    image: Optional[List]
    heatmap: Optional[List]


# ============================================================
# System prompt (DECISION ONLY)
# ============================================================

SYSTEM_PROMPT = AIMessage(
    content=(
        "You are Pia, a medical AI assistant.\n"
        "You cannot see images.\n"
        "You must NEVER describe image features.\n\n"

        "Decide your action based on the user's request:\n"
        "- CHAT: for general questions or conversation\n"
        "- PREDICT: if the user explicitly asks to analyse or predict from an uploaded X-ray\n"
        "- EXPLAIN: if the user asks to explain a model prediction\n\n"

        "Respond with EXACTLY one of:\n"
        "DECISION: CHAT\n"
        "DECISION: PREDICT\n"
        "DECISION: EXPLAIN\n\n"

        "Do not write anything else."
    )
)
CHAT_SYSTEM_PROMPT = AIMessage(
    content=(
        "You are Pia, a medical AI assistant developed for an academic project.\n"
        "You are not Qwen, and you are not affiliated with Alibaba Cloud.\n"
        "Do not mention model names, companies, or training origins.\n"
        "Answer clearly, politely, and accurately.\n"
        "For medical topics, provide general information and avoid diagnoses."
    )
)

# ============================================================
# Decision node
# ============================================================

def decide(state: AgentState):
    llm = ChatOllama(model="qwen2.5", temperature=0)

    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)

    decision = response.content.strip().splitlines()[0]

    return {
        "messages": state["messages"] + [AIMessage(content=decision)]
    }


# ============================================================
# Chat node (THIS WAS MISSING)
# ============================================================

def run_chat(state: AgentState):
    llm = ChatOllama(model="qwen2.5", temperature=0)

    # Remove decision messages
    chat_messages = [
        m for m in state["messages"]
        if not (isinstance(m, AIMessage) and m.content.startswith("DECISION:"))
    ]

    # Inject chat system prompt
    messages = [CHAT_SYSTEM_PROMPT] + chat_messages

    response = llm.invoke(messages)

    return {
        "messages": state["messages"] + [
            AIMessage(content=response.content)
        ]
    }



# ============================================================
# Router
# ============================================================

def route(state: AgentState) -> Literal["chat", "predict", "explain"]:
    decision = state["messages"][-1].content.upper()

    if decision == "DECISION: CHAT":
        return "chat"
    if decision == "DECISION: PREDICT":
        return "predict"
    if decision == "DECISION: EXPLAIN":
        return "explain"

    return "chat"


# ============================================================
# Prediction node
# ============================================================

def run_prediction(state: AgentState):
    image = state.get("image")

    if image is None:
        return {
            "messages": state["messages"] + [
                AIMessage(content="Please upload a chest X-ray image first.")
            ]
        }

    result = predict_pneumonia(image=image)

    label = result["prediction"]
    label_text = "Pneumonia detected" if label == 1 else "Healthy lungs (no pneumonia)"

    return {
        "messages": state["messages"] + [
            AIMessage(
                content=(
                    f"Prediction result: {label_text}\n"
                    f"(Label: {label}, where 0 = healthy lungs, 1 = pneumonia)\n"
                    f"Probability: {result['probability']:.3f}"
                )
            )
        ]
    }


# ============================================================
# Explanation node
# ============================================================

def run_explanation(state: AgentState):
    image = state.get("image")

    if image is None:
        return {"messages": state["messages"]}

    result = explain_pneumonia(image=image)

    return {
        "messages": state["messages"] + [
            AIMessage(
                content=(
                    "The Grad‑CAM heatmap is shown next to the uploaded X‑ray. "
                    "Cool colours (blue) indicate regions with little influence on the prediction, "
                    "while warmer colours (yellow to red) highlight areas that contributed most "
                    "strongly to the model’s decision."
                )
            )
        ],
        "heatmap": result["heatmap"],
    }


# ============================================================
# Build graph
# ============================================================

async def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("decide", decide)
    builder.add_node("chat", run_chat)
    builder.add_node("predict", run_prediction)
    builder.add_node("explain", run_explanation)

    builder.add_edge(START, "decide")
    builder.add_conditional_edges(
        "decide",
        route,
        ["chat", "predict", "explain"]
    )

    builder.add_edge("chat", END)
    builder.add_edge("predict", END)
    builder.add_edge("explain", END)

    return builder.compile()
