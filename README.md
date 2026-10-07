# NLP Projects — LLM Applications with LangChain

Three projects I built during my **AI Internship (NLP track) at DataScience Middle East** (Jul – Sep 2025). Each project follows one DeepLearning.AI course and applies it to a realistic business problem: answering customer FAQs, answering employee HR questions from company documents, and serving that HR assistant as a web API.

| # | Project | What it does | Built with | Course |
|---|---------|--------------|------------|--------|
| 1 | [**FAQ Assistant**](./01-FAQ-Assistant) | Customer-support chatbot that answers from an FAQ knowledge base, remembers the conversation, and grades its own answers | Python · LangChain · OpenAI · in-memory vector search · LLM-assisted evaluation | LangChain for LLM Application Development |
| 2 | [**HR Policy RAG Chatbot**](./02-HR-Policy-RAG-Chatbot) | Internal HR assistant that answers employee questions from policy documents (PDF, Word, text) and politely handles off-topic questions | Python · LangChain · OpenAI embeddings · Chroma · RAG | LangChain: Chat with Your Data |
| 3 | [**HR Chatbot REST API**](./03-HR-Chatbot-API-LangChainJS) | The HR assistant rebuilt in TypeScript and served over HTTP, with separate chat history per user session | TypeScript · LangChain.js · Deno · REST API | Build LLM Apps with LangChain.js |

## What these projects show

- **Retrieval-Augmented Generation (RAG):** load documents → split into chunks → embed → store in a vector database → retrieve the most relevant chunks → answer with an LLM grounded in them.
- **Conversation memory:** follow-up questions ("what about remote work?") are rewritten into standalone questions using the chat history.
- **Guardrails:** prompts forbid inventing policies or numbers; off-topic questions get a short, professional fallback instead of a made-up answer.
- **Evaluation:** Project 1 uses an LLM as a judge to grade each answer against the expected FAQ answer.
- **Shipping:** Project 3 exposes the assistant as an HTTP endpoint other apps can call.

## Repository structure

```
NLP-Projects/
├── 01-FAQ-Assistant/              # Project 1 (Python)
├── 02-HR-Policy-RAG-Chatbot/      # Project 2 (Python)
├── 03-HR-Chatbot-API-LangChainJS/ # Project 3 (TypeScript / Deno)
└── course-notes/                  # My study notes from the three courses
```

## Running the projects

All three projects call the OpenAI API, so you need your own key. Create a `.env` file (see [`.env.example`](./.env.example)) or set `OPENAI_API_KEY` in your environment. **Never commit your real key.**

Each project folder has its own README with setup and run steps.

> These projects were written in 2025 against the LangChain versions used in the courses. Newer LangChain releases have renamed some imports, so pin the versions listed in each project's requirements.

## Courses

1. [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — DeepLearning.AI
2. [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — DeepLearning.AI
3. [Build LLM Apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) — DeepLearning.AI

My notes from each course are in [`course-notes/`](./course-notes). The course lab notebooks themselves belong to DeepLearning.AI and aren't included here.

## Author

**Basmala Mohamed Farouk** — AI Engineer · B.Sc. Artificial Intelligence, AASTMT (2026)
[LinkedIn](https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/) · [GitHub](https://github.com/Basmala-M-Farouk)
