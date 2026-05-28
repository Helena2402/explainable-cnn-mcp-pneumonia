import asyncio
from agent.langgraph_agent import build_agent


async def main():
    # Build agent
    graph = await build_agent()

    # Simulated user request
    user_message = {
        "role": "user",
        "content": (
            "Analyse the chest X-ray image for pneumonia. "
            "Use the prediction tool, then explain the result."
        ),
    }

    # Run agent
    result = await graph.ainvoke(
        {"messages": [user_message]}
    )

    # Print final response
    final_message = result["messages"][-1]
    print("\nAGENT RESPONSE:\n")
    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())