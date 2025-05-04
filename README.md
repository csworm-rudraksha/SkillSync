# SkillSync – AI-Powered Team Suggestion System

SkillSync is a smart employee suggestion dashboard that helps project managers form effective teams from benched or partially billable employees. It combines IBM Watsonx AI's Granite model with a local vector database (ChromaDB) and sentence embeddings (Mini-LLM) to identify top team configurations based on skills and bio-level semantic similarity.

---

## 🚀 Features

- 🔐 **Login System** for role-based access (`manager`, `employee`)
- 📁 **CSV-based Employee Management** with bios and skillsets
- 🤖 **Watsonx Granite Model Integration** for extracting skills from project descriptions
- 🔍 **Semantic Search with ChromaDB + Mini-LLM (mxbai-embed-large-v1)** to find matching employees
- 🧠 **Team Grouping and Similarity Scoring** using vector embeddings
- ⚠️ **Outlier Detection** in groups based on lowest pairwise similarity
- 📊 **Interactive UI** with TailwindCSS, including:
  - Manager Dashboard
  - Suggested Teams
  - Approved Teams
  - Employee View
- ✅ **Select and Approve a Team** with animated confirmation

---

## 🛠️ Technologies Used

| Layer       | Tech                                       |
|-------------|--------------------------------------------|
| Backend     | FastAPI, ChromaDB, Sentence Transformers   |
| LLM         | IBM Watsonx Granite (`text-generation`)    |
| Embeddings  | `mxbai-embed-large-v1` via SentenceTransformer |
| Frontend    | TailwindCSS, HTML, Vanilla JS              |
| Data Store  | Local CSV + ChromaDB (DuckDB + Parquet)    |

---



### Please add in your own API key and Project ID to make it functional.
