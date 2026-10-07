// lib/helpers.ts
// Helpers for loading documents, splitting, creating chains, and fallback prompt.

import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

/* ------------------------- 1. Load documents from ./data ------------------------- */
export async function loadDocumentsFromDataDir(): Promise<Document[]> {
  const docs: Document[] = [];
  const dataDir = "./data";

  for await (const dirEntry of Deno.readDir(dataDir)) {
    const name = dirEntry.name;
    const path = `${dataDir}/${name}`;

    if (name.endsWith(".pdf")) {
      const loader = new PDFLoader(path);
      const loaded = await loader.load();
      docs.push(...loaded);
    } else if (name.endsWith(".txt")) {
      const txt = await Deno.readTextFile(path);
      docs.push(new Document({ pageContent: txt, metadata: { source: name } }));
    } else if (name.endsWith(".json")) {
      try {
        const raw = await Deno.readTextFile(path);
        const parsed = JSON.parse(raw);

        if (Array.isArray(parsed)) {
          for (const item of parsed) {
            let content = "";
            if (item.question && item.answer) {
              content = `Q: ${item.question}\nA: ${item.answer}`;
            } else if (item.content) {
              content = item.content;
            } else {
              content = JSON.stringify(item);
            }
            docs.push(new Document({ pageContent: content, metadata: { source: name } }));
          }
        } else {
          docs.push(new Document({ pageContent: JSON.stringify(parsed), metadata: { source: name } }));
        }
      } catch (e) {
        console.warn("Failed to parse json file:", path, e);
      }
    }
  }
  return docs;
}

/* ------------------------- 2. Split documents ------------------------- */
export async function splitDocuments(
  docs: Document[],
  chunkSize = 1000,
  chunkOverlap = 128
) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
    separators: ["\n\n", "\n", ".", " ", ""],
  });
  return await splitter.splitDocuments(docs);
}

/* ------------------------- 3. Initialize vectorstore ------------------------- */
export async function initializeVectorstoreWithDocuments({
  documents,
}: {
  documents: Document[];
}) {
  const embeddings = new OpenAIEmbeddings();
  const vectorstore = new MemoryVectorStore(embeddings);
  await vectorstore.addDocuments(documents);
  return vectorstore;
}

/* ------------------------- 4. Rephrase question chain ------------------------- */
export function createRephraseQuestionChain() {
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("history"),
    ["human", "Rephrase the following question as a standalone question:\n{question}"],
  ]);

  return RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);
}

/* ------------------------- 5. Document retrieval chain ------------------------- */
export function createDocumentRetrievalChain(retriever: any) {
  const convertDocsToString = (documents: Document[]): string =>
    documents.map((doc) => `<doc>\n${doc.pageContent}\n</doc>`).join("\n");

  return RunnableSequence.from([
    (input: any) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);
}

/* ------------------------- 6. HR Answer Prompt ------------------------- */
export function createHRAnswerPrompt() {
  const HR_SYSTEM_PROMPT = `You are HR AssistantGPT, a professional and friendly HR chatbot. Only answer questions about company HR policies, benefits, or leave.
Rules:
1. Grounded Answers: Use knowledge from documents for HR questions.
2. General/Off-Topic: For greetings, acknowledgements, or non-HR questions, respond politely, briefly, and professionally. Do NOT ask personal questions.
3. Human-Like: Respond as a professional HR colleague.
4. No Hallucinations: Never invent policies or numbers not in the documents.
5. Personalization: Use user's name if provided.
6. Avoid Repetition: Do not repeat previous answers unnecessarily.
7. Professional Follow-Ups: Suggest HR-related next steps or topics at the end. Never casual or personal follow-ups.
8. Clarity & Helpfulness: Keep answers short, clear, actionable.`;

  return ChatPromptTemplate.fromMessages([
    ["system", HR_SYSTEM_PROMPT],
    new MessagesPlaceholder("history"),
    [
      "human",
      "Now, answer this question using the previous context and chat history: {standalone_question}\nContext: {context}",
    ],
  ]);
}

/* ------------------------- 7. Fallback Chain ------------------------- */
export function createProfessionalFallbackChain() {
  const FALLBACK_SYSTEM_PROMPT = `You are a professional HR assistant. Provide guidance only based on company HR policies, employee benefits, or workplace rules. Do not give personal opinions or recommendations. If the question is HR-adjacent but unclear, explain relevant HR policy information and suggest next steps (like checking a policy or contacting HR). Keep the response concise and professional.`;

  const fallbackPrompt = ChatPromptTemplate.fromMessages([
    ["system", FALLBACK_SYSTEM_PROMPT],
    new MessagesPlaceholder("history"),
    ["human", "{question}"],
  ]);

  return RunnableSequence.from([
    fallbackPrompt,
    new ChatOpenAI({ temperature: 0, modelName: "gpt-3.5-turbo-1106" }),
    new StringOutputParser(),
  ]);
}
