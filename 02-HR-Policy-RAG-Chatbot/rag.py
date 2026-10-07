from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import textwrap
import os
import openai

# --- Set API key ---
openai.api_key = os.environ['OPENAI_API_KEY']

# -----------------------------
# --- Document & Vectorstore ---
# -----------------------------
def load_documents(paths):
    """Load all documents from paths."""
    loaders = []
    for path in paths:
        if path.endswith(".pdf"):
            loaders.append(PyPDFLoader(path))
        elif path.endswith(".txt"):
            loaders.append(TextLoader(path))
        elif path.endswith(".docx"):
            loaders.append(Docx2txtLoader(path))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=200, chunk_overlap=50):
    """Split documents into chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vectorstore(splits):
    """Create vectorstore from document splits."""
    embedding = OpenAIEmbeddings()
    return Chroma.from_documents(splits, embedding)

# -----------------------------
# --- QA Chain & LLM Setup ---
# -----------------------------
def create_qa_chain(vectordb):
    """Build conversational retrieval chain with memory and custom prompt."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""
You are HR AssistantGPT, a professional and friendly HR chatbot.
Only answer questions about company HR policies, benefits, or leave.
If the question is not related to HR policies, respond briefly and professionally without repeating HR policy content.

Rules:
1. Grounded Answers: Use knowledge from documents for HR questions.
2. General/Off-Topic: For greetings, acknowledgements, or non-HR questions, respond politely, briefly, and professionally. Do NOT ask personal questions.
3. Human-Like: Respond as a professional HR colleague.
4. No Hallucinations: Never invent policies or numbers not in the documents.
5. Personalization: Use user's name if provided.
6. Avoid Repetition: Do not repeat previous answers unnecessarily.
7. Professional Follow-Ups: Suggest HR-related next steps or topics at the end. Never casual or personal follow-ups.
8. Clarity & Helpfulness: Keep answers short, clear, actionable.

Chat history:
{chat_history}

Company Policy Context:
{context}

User question:
{question}

Answer concisely below and include a professional follow-up suggestion:
"""
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        get_chat_history=lambda h: h
    )

    return qa_chain, llm

# -----------------------------
# --- Dynamic HR Classification ---
# -----------------------------
def is_hr_question_dynamic(llm, query):
    """Ask the LLM if a query is HR-related."""
    prompt = f"""
You are an expert classifier.
Answer only "Yes" or "No".
Is the following question about HR policies, employee benefits, leave, or workplace rules?

Question: {query}
"""
    response = llm.predict(prompt)
    return "yes" in response.lower()

# -----------------------------
# --- Professional fallback ---
# -----------------------------
def professional_fallback(llm, query):
    """Provide a professional fallback for non-HR or HR-adjacent queries."""
    return llm.predict(
        f"""
The user asked: "{query}"
You are a professional HR assistant. Provide guidance only based on company HR policies,
employee benefits, or workplace rules. Do not give personal opinions or recommendations.
If the question is HR-adjacent but unclear, explain relevant HR policy information
and suggest next steps (like checking a policy or contacting HR).
Keep the response concise and professional.
"""
    )

# -----------------------------
# --- Chatbot Loop ---
# -----------------------------
def run_hr_chatbot(document_paths):
    """Run the professional HR chatbot loop."""
    docs = load_documents(document_paths)
    splits = split_documents(docs)
    vectordb = build_vectorstore(splits)
    qa_chain, llm = create_qa_chain(vectordb)

    greeted = False
    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye 👋")
            break

        # Handle first greeting
        if not greeted and query.lower() in ["hi", "hello", "hey"]:
            print("Bot: Hello! How can I assist you today?\n")
            greeted = True
            continue

        # Dynamic HR check
        if is_hr_question_dynamic(llm, query):
            result = qa_chain({"question": query})
            answer = result["answer"]
        else:
            answer = professional_fallback(llm, query)

        # Wrap for readability
        wrapped = textwrap.fill(answer, width=100, break_long_words=False, break_on_hyphens=False)
        print(f"Bot: {wrapped}\n")

# -----------------------------
# --- Example usage ---
# -----------------------------
if __name__ == "__main__":
    document_files = [
        "leave_policy.pdf",
        "remote_work_policy.txt",
        "employee_benefits.docx"
    ]
    run_hr_chatbot(document_files)
