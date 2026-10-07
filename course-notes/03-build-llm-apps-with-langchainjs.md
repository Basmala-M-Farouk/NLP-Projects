# Build LLM Apps with LangChain.js — My Notes

**Provider:** DeepLearning.AI · **Instructor:** Jacob Lee (LangChain.js lead maintainer) · **Level:** Intermediate · **Length:** ~1 h 4 min, 8 video lessons, 6 code examples
**Applied in:** [Project 3 — HR Chatbot REST API](../03-HR-Chatbot-API-LangChainJS)

## What the course covers

Building LLM-powered web backends in JavaScript/TypeScript with LangChain.js: the same RAG ideas as the Python courses, plus shipping the result as a web API.

## Lesson by lesson

**1. Building blocks**
- Text LLMs take a string and return a string; chat models take a list of messages and return a message. Chat models are the default for most apps.
- **LCEL** (LangChain Expression Language) connects components into pipelines.
- Output parsers give structured output; **streaming** returns partial results as they're generated; **batching** sends many requests at once.

**2. Loading and splitting data**
- `PDFLoader` for PDFs, `TextLoader` for text, `DirectoryLoader` for whole folders.
- `RecursiveCharacterTextSplitter` keeps meaning better than plain character splitting and is the best default.

**3. Embeddings and vector stores**
- `OpenAIEmbeddings` turns text into vectors.
- `MemoryVectorStore` lives in RAM (good for demos); FAISS saves locally; Pinecone and Weaviate are cloud options for millions of documents.

**4. Question answering**
- The retriever finds relevant chunks; the LLM answers from them.

**5. Conversational Q&A**
- Rephrase follow-up questions into standalone questions using chat history, then retrieve.
- In-memory history is fine for demos; production apps store history in a database (e.g. Redis, Postgres).

**6. Shipping as a web API**
- Wrap the chain in an HTTP endpoint. `Deno.serve()` is quick for testing; Express or FastAPI are common in production.

## The full pipeline

1. Load docs → 2. Split (`RecursiveCharacterTextSplitter`) → 3. Embed (`OpenAIEmbeddings`) → 4. Store (`MemoryVectorStore` / FAISS / Pinecone) → 5. Retriever (`.asRetriever()`) → 6. Chain with a chat model → 7. Memory per session → 8. Serve over HTTP

## How I applied it

I rebuilt the Project 2 HR assistant in TypeScript on Deno: documents and Q&A pairs loaded from `./data`, a rephrase chain plus retrieval chain composed with `RunnableSequence`, `RunnableWithMessageHistory` keyed by `session_id`, and `Deno.serve()` exposing a `POST` endpoint that returns a JSON reply.
