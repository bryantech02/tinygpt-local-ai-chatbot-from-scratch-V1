# TinyGPT + RAG — Local AI Chatbot Prototype

A locally running educational AI chatbot project built from scratch using Python and PyTorch, combining a custom TinyGPT Transformer architecture with Retrieval-Augmented Generation (RAG), semantic retrieval, query normalization, grounding validation, and ambiguity handling.

This project represents my ongoing journey into AI engineering, machine learning, neural networks, Large Language Models (LLMs), and AI systems.

---

## 🎥 Working Demonstration

### Live Testing Video

A video demonstration is available showing the chatbot running locally in the terminal and responding to different types of user prompts.

The demonstration includes:

- Normal questions
- Different ways of asking the same question
- Poorly structured / fragmented questions
- Semantic retrieval
- RAG-grounded answers
- Ambiguous questions
- Out-of-knowledge questions
- Grounding-based refusal
- Final evaluation results

**▶️ Watch the chatbot demonstration:**

[Google Drive — TinyGPT + RAG Chatbot Demonstration](SOON TO UPLOAD)

---

# 🤖 Project Overview

This project is an educational local AI chatbot prototype designed to explore how modern AI systems can combine:

- Neural networks
- Transformer architecture
- Tokenization
- Semantic embeddings
- Vector similarity
- Information retrieval
- Retrieval-Augmented Generation (RAG)
- Query normalization
- Lexical matching
- Topic detection
- Intent detection
- Grounding validation
- Ambiguity handling
- Model evaluation

The system runs locally and does not require a cloud-hosted chatbot API for its core operation.

---

# 🧠 How the System Works

The chatbot follows a retrieval-and-grounding pipeline.

You:
                    USER
                      │
                      ▼
                User Question
                      │
                      ▼
             Query Normalization
                      │
                      ▼
              Semantic Embedding
                      │
                      ▼
            Knowledge Retrieval
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
     Semantic Similarity   Lexical Matching
            │                   │
            └─────────┬─────────┘
                      ▼
                Topic Detection
                      │
                      ▼
                Intent Detection
                      │
                      ▼
             Result Ranking
                      │
                      ▼
            Grounding Validation
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Valid Knowledge      Invalid /
             │              Ambiguous
             ▼                 │
       Grounded Answer         ▼
                          Safe Refusal
