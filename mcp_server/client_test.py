import asyncio
import numpy as np
import tensorflow_datasets as tfds
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    """
    Minimal client to test the MCP server tool:
    - loads one PneumoniaMNIST test image
    - sends it to the MCP server
    - prints prediction and Grad-CAM info
    """

    # -------------------------
    # Load one test image
    # -------------------------
    ds_test = tfds.load(
        "pneumonia_mnist",
        split="test",
        as_supervised=True
    )

    image, label = next(iter(ds_test.take(1)))
    image_np = image.numpy().tolist()

    print("True label:", int(label.numpy()))

    # -------------------------
    # Connect to MCP server
    # -------------------------
    client = MultiServerMCPClient(
        {
            "pneumonia_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Fetch available tools
    tools = await client.get_tools()
    tool_names = [tool.name for tool in tools]

    print("Available MCP tools:", tool_names)

    # -------------------------
    # Call predict_pneumonia tool
    # -------------------------
    tool = next(tool for tool in tools if tool.name == "predict_pneumonia")

    results = await tool.ainvoke(
        {
            "image": image_np
        }
    )


    results = await tool.ainvoke(
        {
            "image": image_np
        }
    )

    # MCP returns a list of messages; take the first one
    message = results[0]

    # Parse JSON payload
    payload = json.loads(message["text"])

    # -------------------------
    # Print results
    # -------------------------
    print("Prediction:", payload["prediction"])
    print("Probability:", payload["probability"])
    print(
        "Heatmap size:",
        len(payload["heatmap"]),
        "x",
        len(payload["heatmap"][0]),
    )


if __name__ == "__main__":
    asyncio.run(main())