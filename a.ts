import dotenv from "dotenv";
dotenv.config();
import { ChatGroq } from "@langchain/groq";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
import { concat } from "@langchain/core/utils/stream";
import { StateGraph } from "@langchain/langgraph";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.3-70b-versatile",
  temperature: 0,
});

const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
  apiKey: process.env.MISTRAL_API_KEY,
});

// const vectorStore = new MemoryVectorStore(embeddings);

const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await splitter.splitDocuments(docs);
console.log(allSplits.length);

const pinecone = new PineconeClient();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX!);

const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
  // Maximum number of batch requests to allow at once. Each batch is 1000 vectors.
  maxConcurrency: 5,
  // You can pass a namespace here too
  // namespace: "foo",
});

await vectorStore.addDocuments(allSplits);

const retriever = vectorStore.asRetriever({
  k: 5, // number of results
});

// const res = await retriever.invoke("What is Task Decomposition?");
// console.log(res);

const customTemplate = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Answer:`;

const customRagPrompt = PromptTemplate.fromTemplate(customTemplate);

const customRagChain = await createStuffDocumentsChain({
  llm: llm,
  prompt: customRagPrompt,
  outputParser: new StringOutputParser(), // output result as string
});

const userQuery = "What is Task Decomposition?";

const context = await retriever.invoke(userQuery);

const res = await customRagChain.invoke({
  question: userQuery,
  context,
});

console.log("res", res);

// const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// // Example:
// const example_prompt = await promptTemplate.invoke({
//   context: "(context goes here)",
//   question: "(question goes here)",
// });
// const example_messages = example_prompt.messages;

// console.assert(example_messages.length === 1);

// const InputStateAnnotation = Annotation.Root({
//   question: Annotation<string>,
// });

// const StateAnnotation = Annotation.Root({
//   question: Annotation<string>,
//   context: Annotation<Document[]>,
//   answer: Annotation<string>,
// });

// const retrieve = async (state: typeof InputStateAnnotation.State) => {
//   const retrievedDocs = await vectorStore.similaritySearch(state.question);
//   return { context: retrievedDocs };
// };

// const generate = async (state: typeof StateAnnotation.State) => {
//   const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
//   const messages = await promptTemplate.invoke({
//     question: state.question,
//     context: docsContent,
//   });
//   const response = await llm.invoke(messages);
//   return { answer: response.content };
// };

// const graph = new StateGraph(StateAnnotation)
//   .addNode("retrieve", retrieve)
//   .addNode("generate", generate)
//   .addEdge("__start__", "retrieve")
//   .addEdge("retrieve", "generate")
//   .addEdge("generate", "__end__")
//   .compile();

// // Input test
// let inputs = { question: "What is Task Decomposition?" };

// const stream = await graph.stream(inputs, { streamMode: "messages" });

// // other type of output operation can be used by replacing the below code
// for await (const [message, _metadata] of stream) {
//   process.stdout.write(message.content + "|");
// }
