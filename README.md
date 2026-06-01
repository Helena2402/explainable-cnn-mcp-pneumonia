# Pia – Explainable Multimodal AI for Pneumonia Detection

This repository contains the implementation of **Pia**, a multimodal and explainable AI system for pneumonia detection from chest X‑ray images.
The system combines a convolutional neural network (CNN) for image‑based prediction with Grad‑CAM for visual explanation, exposed through a FastMCP server and orchestrated by a LangGraph agent. A Streamlit application provides the user interface.

---

## ✨ Features

- **Pneumonia Prediction (CNN)**  
  Binary classification of chest X‑ray images (normal vs pneumonia).

- **Visual Explanation (Grad‑CAM)**  
  Heatmaps highlighting image regions that influenced the model’s prediction.

- **FastMCP Server**  
  Prediction and explanation exposed as callable MCP tools.

- **LangGraph Agent**  
  Decision‑based routing between chat, prediction, and explanation tools.

- **Streamlit Interface**  
  Image upload, chat interaction, and Grad‑CAM visualisation in a single UI.

- **Multimodal Interaction**  
  Supports both image input and text‑based queries within one system.

---

## 🧰 Prerequisites

- Python 3.9+
- Ollama (local LLM inference)
- Streamlit
- TensorFlow
- LangGraph, LangChain, FastMCP  
(All dependencies are listed in `requirements.txt`)

---

## ⚙️ Installation

```bash
git clone https://github.com/Helena2402/explainable-cnn-mcp-pneumonia.git
cd explainable-cnn-mcp-pneumonia
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧠 Ollama Setup

The agent uses a small local language model to run on a personal laptop.

```bash
ollama pull qwen2.5
ollama serve
```

---

## 📁 Project Structure

```
explainable-cnn-mcp-pneumonia/
├── agent/
│   └── langgraph_agent.py       # LangGraph agent logic
├── app/
│   └── streamlit_app.py         # Streamlit UI
├── mcp_server/
│   └── server.py                # FastMCP server exposing tools
├── model/
│   ├── cnn_functional.py        # CNN (Functional API)
│   ├── gradcam.py               # Grad-CAM implementation
│   ├── predict_and_explain.py   # Model service wrapper
│   └── pneumonia_cnn.keras      # Trained model
├── main.py                      # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

### 1. Start the FastMCP Server

```bash
python mcp_server/server.py
```

The server exposes prediction and explanation tools over MCP.

---

### 2. Launch the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Open the application in your browser at:

```
http://localhost:8501
```

---

### 3. Using the System

- Upload a chest X‑ray image (single image at a time)
- Ask a question in the chat interface
- The agent will:
  - respond directly (chat), or
  - call the prediction tool, or
  - call the explanation tool
- Grad‑CAM heatmaps are displayed when explanation is requested

---

## ℹ️ Notes

- The system is designed as an academic prototype.
- The CNN is trained on the PneumoniaMNIST dataset (28 × 28 grayscale images).
- Grad‑CAM explanations are qualitative due to the low image resolution.
- The language model is intentionally small to support local execution.

---

This project was developed for academic purposes.
