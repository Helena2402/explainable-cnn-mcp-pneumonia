from typing import Optional, List, Literal, Dict, Any
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from rag.rag import ProjectRAG
from mcp_server.server import predict_pneumonia, explain_pneumonia


# ============================================================
# State
# ============================================================

class AgentState(MessagesState):
    image: Optional[List]
    heatmap: Optional[List]
    retrieved_context: Optional[str]


# ============================================================
# Global components
# ============================================================

llm = ChatOllama(model="qwen2.5", temperature=0)
rag = None


# ============================================================
# Decision node
# ============================================================

DECISION_PROMPT = """You are routing user requests in a medical AI system.

Choose ONE label:

- CHAT:
  General questions, explanations, or questions about pneumonia, X-rays, or how the system works.
- PREDICT:
  User asks to analyse or predict from an uploaded X-ray image.
- EXPLAIN:
  ONLY if user explicitly asks to explain a PREVIOUS prediction result or heatmap.

Important:
- General "how does it work" questions are CHAT


Return ONLY one word.
"""


def decide(state: AgentState) -> Dict[str, Any]:
    user_text = state["messages"][-1].content

    response = llm.invoke([
        HumanMessage(content=DECISION_PROMPT + "\n\nUser: " + user_text)
    ])

    decision = response.content.strip().upper()

    return {
        "messages": [AIMessage(content=f"DECISION: {decision}")]
    }


def route(state: AgentState) -> Literal["chat", "predict", "explain"]:
    last = state["messages"][-1].content
    image = state.get("image")

    if "PREDICT" in last:
        return "predict"
    elif "EXPLAIN" in last:

        if image is None:
            return "chat"
        return "explain"

    return "chat"


# ============================================================
# RAG node
# ============================================================

def retrieve_context(state: AgentState):
    global rag

    if rag is None:
        rag = ProjectRAG()

    query = state["messages"][-2].content  # user msg before decision
    context = rag.retrieve(query)

    return {
        "retrieved_context": context[:2000]  # prevent prompt explosion
    }


# ============================================================
# Chat node
# ============================================================

def run_chat(state: AgentState):
    history = [
        m for m in state["messages"]
        if not (isinstance(m, AIMessage) and m.content.startswith("DECISION:"))
    ]

    user_query = history[-1].content
    context = state.get("retrieved_context", "")

    prompt = f"""
You are a Pia, medical AI assistant for predicting pneumonia.

Use the following context if relevant.
If not relevant, ignore it.

Context:
{context}

User:
{user_query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}


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

    heatmap = result["heatmap"]

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
        "heatmap": heatmap
    }



# ============================================================
# Build graph
# ============================================================

async def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("decide", decide)
    builder.add_node("retrieve", retrieve_context)
    builder.add_node("chat", run_chat)
    builder.add_node("predict", run_prediction)
    builder.add_node("explain", run_explanation)

    builder.set_entry_point("decide")

    builder.add_conditional_edges(
        "decide",
        route,
        {
            "chat": "retrieve",
            "predict": "predict",
            "explain": "explain",
        },
    )

    builder.add_edge("retrieve", "chat")

    builder.add_edge("chat", END)
    builder.add_edge("predict", END)
    builder.add_edge("explain", END)

    return builder.compile()
