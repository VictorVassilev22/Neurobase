from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from app.config import get_vectorstore_dir, EMBEDDING_MODEL, LLM_ENDPOINT
from langchain_core.runnables import RunnableMap, RunnableLambda



# 📁 Vector DB directory
vectorstore_dir = get_vectorstore_dir("./")
# print(f"📂 Vectorstore directory: {vectorstore_dir}")

# 🧠 Load LLM endpoint
llm = HuggingFaceEndpoint(
    endpoint_url=LLM_ENDPOINT,
    max_new_tokens=768,
    top_k=30,
    top_p=0.7,
    temperature=0.2,
    repetition_penalty=1.4,
)

# 🧠 Vector DB + Retriever setup
db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    # collection_metadata={"hnsw:space": "cosine"}
)

# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 8}
# )

retriever = db.as_retriever(
    search_type="mmr", # "similarity" or "hnsw"
     search_kwargs={"k": 6,  # Number of documents to retrieve from the vectorstore.
                   "lambda_mult": 0.7} # precision vs recall tradeoff
)

# 🧠 MAP (per-document) prompt
map_prompt = PromptTemplate.from_template("""
You are a highly skilled AI assistant helping developers understand their codebase. Analyze the following file snippet and answer the developer's question based on the content provided.

Context:
{context}

Developer Question:
{question}

Instructions:
- If the answer is explicitly found in the snippet, quote the relevant lines and explain what they do.
- Extract key elements (functions, classes, constants, logic) that help answer the question.
- If the answer can be logically inferred, explain how.
- If the snippet doesn't contain relevant info, respond with: "Not found in this snippet."

Respond concisely and with technical clarity.
""")

# 🧠 REDUCE (merge answers) prompt
reduce_prompt = PromptTemplate.from_template("""
Developer Question:
{question}

Answers from File Snippets:
{context}

Instructions:
- Given these *Answers from File Snippets* produce the *Final Answer* which best answers the *Developer Question*
- Respond in a clear, structured, and developer-friendly format.

Final Answer:
""")

map_chain = map_prompt | llm
reduce_chain = reduce_prompt | llm

combine_chain = (
    RunnableMap({
        "question": lambda x: x["question"],
        "input_documents": lambda x: x["documents"],
    })
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "summaries": [
            map_chain.invoke({"context": doc.page_content, "question": x["question"]})
            for doc in x["input_documents"]
        ],
        "input_documents": x["input_documents"]
    })
    | RunnableLambda(lambda x: {
        "result": reduce_chain.invoke({
            "question": x["question"],
            "context": "\n\n".join(x["summaries"]),
        }),
        "source_documents": x["input_documents"]
    })
)



# 🎯 Main runner
# 🎯 Main Entry Point
def run_query(question: str):
    print(f"🔎 Question: {question}")
    docs = retriever.invoke(question)
    result = combine_chain.invoke({"question": question, "documents": docs})

    print(f"💬 Answer: {result['result']}\n")
    print(f"📁 Files used:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', '[no source]')}")