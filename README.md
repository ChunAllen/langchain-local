# Process local files with Langchain

Sample Node.js app that uses OpenAI to process local data using Langchain

Medium Article: https://medium.com/@allenchun/feed-local-data-to-llm-using-langchain-node-js-fd7ac44093e9


### How it works
- Langchain processes it by loading documents inside docs/ (In this case, we have a sample data.txt)
- It works by taking big source of data, take for example a 50-page PDF and breaking it down into chunks
- These chunks are then embedded into a Vector Store which serves as a local database and can be used for data processing

### Prerequisites
- Node.js <= 18

### Setting Up
- npm install
- mv .env.copy .env
- Replace `OPEN-API-KEY` in .env with your actual API Key from OpenAPI


### Running prompts 
- Asking questions related to the document
```
$ node index.js "Describe this applicant's employment history"
{
  text: ' This applicant has 5+ years of experience in IT, with experience in System Administration, Network Configuration, Software Installation, Troubleshooting, Windows Environment, Customer Service, and Technical Support. They worked as a Senior IT Specialist at XYZ Global from 2018-Present, an IT Support Specialist at Zero Web from 2015-2017, and a Junior Desktop Support Engineer at Calumcoro Medical from 2014-2015.'
}
```
- Asking questions not related to the document
```
$ node index.js "What is 1+1?"
{ text: " I don't know." }
```

### What if we want to reference Langchain using our local data and OpenAI LLM
```
const chain = new RetrievalQAChain({
  combineDocumentsChain: loadQARefineChain(model),
  retriever: vectorStore.asRetriever(),
});
```