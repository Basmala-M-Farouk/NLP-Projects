import os
import datetime
import warnings
from dotenv import load_dotenv, find_dotenv
import openai
import textwrap

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.docstore.document import Document
from langchain.document_loaders import JSONLoader
from langchain.evaluation.qa import QAEvalChain

warnings.filterwarnings("ignore")

# --- Load API Key ---
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# --- Model Selection ---
current_date = datetime.datetime.now().date()
llm_model = "gpt-3.5-turbo" if current_date > datetime.date(2024, 6, 12) else "gpt-3.5-turbo-0301"

# --- Load FAQ Data ---
json_loader = JSONLoader(file_path="fake.json", jq_schema=".[]", text_content=False)
json_docs = json_loader.load()

# Convert dicts to string format for LLM
for doc in json_docs:
    content = doc.page_content
    if isinstance(content, dict):
        doc.page_content = f"Q: {content.get('question','')}\nA: {content.get('answer','')}"

docs = []
docs.extend(json_docs)

# --- Initialize LLM and embeddings ---
llm = ChatOpenAI(model=llm_model, temperature=0.0)
embeddings = OpenAIEmbeddings()

# --- Vector Index ---
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_documents(docs)

# --- Memory ---
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    input_key="question",
    output_key="answer"
)

# --- Prompt Templates ---
answer_template = """
You are a helpful customer support assistant.
Use the FAQ database to answer questions.
If the answer is not found, say: "I'm not sure, please contact customer support."

Context:
{context}

Question:
{question}

Helpful Answer:
"""
answer_prompt = PromptTemplate(input_variables=["context", "question"], template=answer_template)

condense_template = """
Given the following conversation and a follow-up input, rephrase the follow-up so it becomes
a standalone question in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""
condense_prompt = PromptTemplate.from_template(condense_template)

# --- Build QA chain once ---
retriever = index.vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs={"prompt": answer_prompt},
    return_source_documents=True,
    verbose=False
)

# --- Evaluation chain (LLM-assisted) ---
eval_chain = QAEvalChain.from_llm(llm)

# --- Chat Loop ---
def chat_loop():
    print("Welcome to the FAQ Assistant! Type 'exit' to quit.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        # Ask the assistant
        response = qa_chain({"question": user_input, "chat_history": chat_history})
        answer = response["answer"]
        sources = response.get("source_documents", [])

        # --- Handle off-topic questions ---
        off_topic = False
        if not sources or len(sources) == 0:
            answer = "I'm not sure, please contact customer support."
            off_topic = True

        # Add to memory
        chat_history.append((user_input, answer))

        # Show response
        wrapped_answer = textwrap.fill(answer, width=100, break_long_words=False, break_on_hyphens=False)
        print(f"Bot: {wrapped_answer}\n")

        # --- Evaluate this response ---
        if off_topic:
            print("Eval: CORRECT (off-topic handled properly)\n")
        else:
            faq_answer = sources[0].page_content.split("A:")[-1].strip()
            eval_result = eval_chain.evaluate(
                [{"query": user_input, "answer": faq_answer}],
                [{"query": user_input, "answer": faq_answer, "result": answer}]
            )
            print(f"Eval: {eval_result[0]['text']}\n")


if __name__ == "__main__":
    chat_loop()
