// Documentation: https://betterprogramming.pub/build-a-chatbot-on-your-csv-data-with-langchain-and-openai-ed121f85f0cd

import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { JSONLoader } from "langchain/document_loaders/fs/json";

import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain, loadQARefineChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

//3. Import env variables
import * as dotenv from 'dotenv';

//4. Import others
import fs from 'fs';

// Load Environment Variables
dotenv.config()

// Load local files such as .json and .txt from ./docs
const loader = new DirectoryLoader("./docs", {
  ".json": (path) => new JSONLoader(path),
  ".txt": (path) => new TextLoader(path)
})
const VECTOR_STORE_PATH = "Documents.index";


const normalizeDocuments = (docs) => {
  return docs.map((doc) => {
    if (typeof doc.pageContent === "string") {
      return doc.pageContent;
    } else if (Array.isArray(doc.pageContent)) {
      return doc.pageContent.join("\n");
    }
  });
}

// Main Function
export const run = async (params) => {
  const prompt = params[0]
  console.log('Prompt:', prompt)

  console.log("Loading docs...")
  const docs = await loader.load();

  console.log('Processing...')
  const model = new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });

  let vectorStore;

  console.log('Creating new vector store...')
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });
  const normalizedDocs = normalizeDocuments(docs);
  const splitDocs = await textSplitter.createDocuments(normalizedDocs);

  vectorStore = await HNSWLib.fromDocuments(
    splitDocs,
    new OpenAIEmbeddings()
  );

  await vectorStore.save(VECTOR_STORE_PATH);
  console.log("Vector store created.")

  console.log("Creating retrieval chain...")
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever())

  // const chain = new RetrievalQAChain({
  //   combineDocumentsChain: loadQARefineChain(model),
  //   retriever: vectorStore.asRetriever(),
  // });

  console.log("Querying chain...")
  const res = await chain.call({ query: prompt })

  console.log(res)
}

run(process.argv.slice(2))