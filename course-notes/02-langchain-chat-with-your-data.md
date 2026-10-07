# LangChain: Chat with Your Data — My Notes

**Provider:** DeepLearning.AI · **Instructor:** Harrison Chase · **Level:** Beginner · **Format:** 8 video lessons, 6 code examples
**Applied in:** [Project 2 — HR Policy RAG Chatbot](../02-HR-Policy-RAG-Chatbot)

## What the course covers

How to build chatbots that answer from **your own documents** instead of only the model's training data, using **Retrieval-Augmented Generation (RAG)**: retrieve the relevant text, then let the LLM answer from it.

## Key concepts

- **Document loading:** loaders for PDF, Word, TXT and even YouTube transcripts bring external content in.
- **Document splitting:** break large documents into smaller, meaningful, overlapping chunks so embeddings and retrieval are more accurate.
- **Embeddings and vector stores:** turn chunks into vectors and index them (FAISS, Chroma) for semantic similarity search.
- **Retrieval:** fetch the most relevant chunks for a question.
- **Question answering:** pass those chunks plus the question to the LLM for a grounded, context-aware answer.
- **Conversational memory:** keep chat history so follow-up questions are understood in context.

## The RAG pipeline

```
Documents → Loader → Splitter → Embeddings → Vector store
                                                   ↓
User question → (rephrase with chat history) → Retriever → relevant chunks
                                                   ↓
                                    LLM + prompt → grounded answer
```

## Skills

Data ingestion with loaders · chunking strategies · embedding generation and vector indexing · retrieval pipelines · contextual Q&A with LLMs · conversational interfaces over private data

## Real-world uses

Company knowledge assistants · policy and compliance chatbots · research assistants over private data · educational Q&A · document-aware support bots

## How I applied it

I built an internal HR policy chatbot over three synthetic policy files (PDF, DOCX, TXT): loaders per format → small overlapping chunks → OpenAI embeddings → Chroma → a conversational retrieval chain with strict "don't invent policies" prompt rules, plus an LLM classifier that routes non-HR questions to a polite fallback.
