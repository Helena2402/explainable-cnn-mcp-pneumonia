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
    tools = await client.get_tools()
    print("Available MCP tools:", [tool.name for tool in tools])

    tool = next(tool for tool in tools if tool.name == "predict_pneumonia")

    results = await tool.ainvoke(
        {
            "image": image_np
        }
    )

    # MCP returns a list of messages; take the first one
    message = results[0]

    # Prediction
    predict_tool = next(t for t in tools if t.name == "predict_pneumonia")
    pred_msg = (await predict_tool.ainvoke({"image": image_np}))[0]
    pred = json.loads(pred_msg["text"])

    print("Probability:", pred["probability"])

    # Explanation
    explain_tool = next(t for t in tools if t.name == "explain_pneumonia")
    exp_msg = (await explain_tool.ainvoke({"image": image_np}))[0]
    exp = json.loads(exp_msg["text"])

    print(
        "Heatmap size:",
        len(exp["heatmap"]),
        "x",
        len(exp["heatmap"][0]),
    )


if __name__ == "__main__":
    asyncio.run(main())