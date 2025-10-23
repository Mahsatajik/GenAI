# Automating Clinical Trial Eligibility: Building a Smart Pipeline with LangGraph, ChromaDB, and Gemini

Clinical trials are vital for advancing medical research, yet identifying eligible patients is a slow and error-prone process. This project introduces an **AI-driven pipeline** that automates patient eligibility screening using **NLP**, **vector databases**, and **LLMs** to bridge the gap between clinical data and trial criteria.

---

## 🧠 Project Overview

The **Clinical Trial Eligibility Pipeline** processes clinical discharge notes from the **MIMIC-IV dataset**, summarizes them, matches them against trial inclusion/exclusion criteria, and evaluates patient eligibility.

Built with **Python**, **LangGraph**, **ChromaDB**, and **Google’s Gemini API**, the system streamlines the entire eligibility workflow through a modular, agent-based design and an interactive **Gradio UI**.

### Key Features

* **Summarization** — Condenses lengthy clinical notes into concise, structured summaries.
* **Criteria Processing** — Converts trial conditions into structured LLM prompts.
* **Chunking & Embedding** — Stores summary vectors in a ChromaDB collection for retrieval.
* **Eligibility Evaluation** — Uses Gemini and chain-of-thought (CoT) reasoning to assess patient-trial fit.
* **Evaluation Metrics** — ROUGE, cosine similarity, and fuzzy matching validate results.
* **Interactive UI** — A Gradio dashboard visualizes results and database contents.

---

## ⚙️ Workflow Overview

The workflow is orchestrated using **LangGraph**, a stateful framework managing data flow between each pipeline component.
Each node represents a distinct processing step—from summarization to evaluation—within a controlled execution graph.

![Workflow Diagram](https://storm-jacket-4ec.notion.site/image/attachment%3A96e2ead4-bad1-4037-937f-3a1ddd46f684%3AWorkflow-blogpost.jpg?table=block\&id=1dbd34e9-9126-8030-91f0-e7abe3ef1314\&spaceId=7a2dddc0-0aad-4b5a-9b6d-f2a7630610e1\&width=2000\&userId=\&cache=v2)

### Workflow Steps

1. **Summarize Clinical Data:**
   Fine-tune a Gemini model to summarize MIMIC-IV discharge notes into concise summaries capturing key diagnoses and outcomes.

2. **Database Storage:**
   Save raw and summarized texts into a local SQLite database for downstream tasks.

3. **Criteria Processing:**
   Read inclusion/exclusion criteria from CSVs, convert to JSON, and generate structured CoT prompts.

4. **Chunking & Embedding:**
   Split summaries into 450-character chunks, embed them using Google’s `text-embedding-004`, and store vectors in ChromaDB.

5. **Eligibility Evaluation:**
   Retrieve relevant chunks and evaluate eligibility using Gemini reasoning (avoiding inference from symptoms).

6. **Result Evaluation:**
   Validate Gemini’s decisions via ROUGE and similarity-based metrics; save CSV reports for each trial.

7. **Visualization (Gradio UI):**
   Display eligibility results, database content, embeddings, and logs in an interactive web interface.

---

## 🧩 Key Components

| Component           | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| **LangGraph Agent** | Orchestrates pipeline execution and manages state between nodes. |
| **Gemini Models**   | Summarization and reasoning (Gemini 1.5 Flash).                  |
| **ChromaDB**        | Vector database for chunk retrieval and similarity search.       |
| **SQLite**          | Stores summaries and trial metadata.                             |
| **Gradio UI**       | Frontend interface for results visualization and exploration.    |

---

## 🧪 Evaluation and Impact

The pipeline demonstrates how AI can reduce manual screening workload and accelerate patient-trial matching, increasing recruitment efficiency while maintaining interpretability and compliance.

**Future Directions:**

* Integrate real-time eligibility visualization.
* Expand support to multimodal data (e.g., imaging and lab reports).
* Deploy as a clinical decision-support tool for hospital systems.

---

## 👥 Authors

Developed by:

* [Mehrnoosh Azizi](https://www.linkedin.com/in/mehrnoosh-azizi/)
* [Reza Rezaeian](https://www.linkedin.com/in/reza-rezaeian/)
* [Mahsa Tajik](https://www.linkedin.com/in/mahsa-tajik/)

---

This repository showcases how **LangGraph**, **ChromaDB**, and **LLMs** like **Gemini** can collaborate in a modular agent architecture to make clinical trial screening faster, more accurate, and scalable.
