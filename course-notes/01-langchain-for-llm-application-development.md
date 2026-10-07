# LangChain for LLM Application Development — My Notes

**Provider:** DeepLearning.AI · **Instructors:** Harrison Chase (creator of LangChain) & Andrew Ng · **Level:** Beginner · **Length:** ~1 h 38 min, 8 video lessons, 6 code examples
**Applied in:** [Project 1 — FAQ Assistant](../01-FAQ-Assistant)

## What the course covers

A practical introduction to building LLM-powered applications with LangChain: structuring prompts and parsing outputs, keeping conversation memory, chaining steps together, answering questions over your own documents, building agents that use tools, and evaluating LLM answers.

**Skills:** LangChain fundamentals · prompt engineering · conversational memory · chain construction · agents and tool use · LLM output evaluation

---

## Lesson 1 — Models, Prompts and Parsers

**Flow:**
1. **Create the model:** `ChatOpenAI(model_name="gpt-3.5-turbo")`, the LLM that generates responses.
2. **Define the output fields:** `ResponseSchema(name="gift", description="Was the item purchased as a gift?")`.
3. **Build a parser:** `StructuredOutputParser.from_response_schemas([...])`.
4. **Get format instructions:** `output_parser.get_format_instructions()` tells the model to reply in a parseable (JSON-like) format.
5. **Build the prompt:** `PromptTemplate` / `ChatPromptTemplate.from_template(...)` with placeholders plus the format instructions.
6. **Call the model** with the formatted messages.
7. **Parse the reply:** `output_parser.parse(response)` turns the text into a Python dict.

**Useful pieces**
- `load_dotenv()` / `find_dotenv()`: load API keys from a `.env` file.
- `format_messages()`: turns a prompt template into chat messages.
- `LLMChain(llm=llm, prompt=prompt)` then `chain.run(inputs)`: fill the template → call the model → get the response.

## Lesson 2 — Memory

```python
llm = ChatOpenAI(temperature=0.0)      # temperature 0 = predictable answers
memory = ConversationBufferMemory()    # or Window / Token / Summary
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
conversation.predict(input="Hello, I am Basmallah")
conversation.predict(input="What is my name?")
memory.load_memory_variables({})       # inspect what is stored
```

| Memory type | What it keeps | Pros | Cons |
|---|---|---|---|
| Buffer | All history | Full memory | Gets long and expensive |
| BufferWindow (k) | Last k exchanges | Focus on recent turns | Forgets older context completely |
| TokenBuffer (limit) | Up to N tokens | Fits model token limits | Old info dropped |
| SummaryBuffer (limit) | Summary + latest turns | Effectively unlimited, compressed | Depends on summary quality |

Others: **VectorStoreMemory** (embeddings for semantic recall) and **EntityMemory** (facts about people and things, for personalised assistants).

**Notes:** `verbose=True` prints what happens under the hood. `OpenAI` is the older completion API (text in → text out); `ChatOpenAI` is the chat API (messages in → messages out) and is what we normally use.

## Lesson 3 — Chains

- **LLMChain:** the simplest chain, one prompt + one LLM. Example: `{product}` → "What is the best name for a company that makes {product}?"
- **SimpleSequentialChain:** chains run one after another; each step's single output becomes the next step's input (product → company name → company description).
- **SequentialChain:** the flexible version with multiple inputs and outputs. You declare `input_variables`, `output_variables`, and an `output_key` per step. Example: Review → English translation → summary; Review → language; summary + language → follow-up message.
- **Router chain (`MultiPromptChain`):** a router LLM reads the question and sends it to the best destination chain (physics, math, history, computer science) or a `default_chain`.

> Think of chains as train tracks: LLMChain is a single short track, SequentialChain passes through several stations, and a RouterChain is the switch that picks the track.

## Lesson 4 — Question Answering over Documents

**Mental model:** Data → Embeddings → Vector DB → Retriever → LLM → Answer

```python
docs = loader.load()
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
```

| Chain type | Flow | Good for | Weakness |
|---|---|---|---|
| Stuff | All docs in one LLM call | Small docs, fast | Context-window limit |
| Map Reduce | Answer per doc → combine | Large document sets | More calls, may lose nuance |
| Refine | First answer → improve doc by doc | Layered, thorough answers | Order matters |
| Map Rerank | Answer + score per doc → pick best | Single strongest source | Ignores other docs |

Start with **Stuff**; switch to **Map Reduce** when documents get large.

## Lesson 5 — Evaluation

| Approach | How | Pros | Cons |
|---|---|---|---|
| Manual | You check each answer yourself | Simple, full control | Slow, subjective, doesn't scale |
| LLM-assisted | An LLM grades answers (`QAEvalChain`) | Fast, consistent, scalable | LLM bias, API cost, needs spot checks |
| LangChain evaluation platform | Tracks runs, metrics and comparisons | Built for teams and production | More setup |

**Pipeline:** question → retriever → QA chain → LLM → answer → compare with ground-truth Q&A examples → metrics.

> Manual = hand-grading homework · LLM-assisted = a teaching assistant grades it · Platform = a full grading system with reports.

## Lesson 6 — Agents

- **LLM = brain:** reasons and decides what to do.
- **Tools = hands:** built-in (`llm-math`, Wikipedia, search, Python REPL) or custom functions wrapped with `@tool`.
- **Agent = orchestrator:** `initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)`.

**Flow:** user query ("What is 25% of 300?") → agent decides a tool is needed → calculator runs → result (75) → LLM writes the final answer.
