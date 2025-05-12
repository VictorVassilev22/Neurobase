from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from app.config import get_vectorstore_dir, EMBEDDING_MODEL, LLM_ENDPOINT
from collections import Counter



vectorstore_dir = get_vectorstore_dir("./")

# 🧠 Initialize LLM
llm = HuggingFaceEndpoint(
    endpoint_url=LLM_ENDPOINT,
    max_new_tokens=256, # The maximum number of tokens to generate in the output.
    top_k=30, # The number of top most probable tokens the model samples from at each generation step.
    top_p=0.8, # The cumulative probability of parameter highest probability tokens to keep for generation. restrict overly creative outputs
    temperature=0.2, # The value used to modulate the next token probabilities.  LOWER for factual, deterministic output
    repetition_penalty=1.5 # The parameter for repetition penalty. 1.0 means no penalty. increase if repeating function names
)


# 🧠 Initialize Vector Store + Retriever
db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    # collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
    search_type="mmr", # "similarity" or "hnsw"
     search_kwargs={"k": 22,  # Number of documents to retrieve from the vectorstore.
                   "lambda_mult": 0.7} # precision vs recall tradeoff
)

#  Prompt templates
initial_prompt = PromptTemplate.from_template("""
You are an expert developer assistant. Your job is to answer questions about a codebase.
You are given:
                                              
- The Developer Request
- A portion of the codebase which is the Context                                             

Your job is to serve the Developer Request clearly using the Context provided.
                                              
------------------                                              

Developer Request:
{question}

Context:
{context}

Instructions:
- Use only the Context shown above to Answer.
- If the Context does not contain enough information to Answer the Developer Request, state clearly that you don't know the Answer.
- Quote exact variable names, values, files or lines if applicable.
- Always point the developer to the file where the information is found.
- Do NOT mention anything about your Context, NOT you previous iterations and conclusions or how you are instruncted to Answer or changing your Answer.
- Keep the Answer informative, concise and maximally relevant to the Developer Request.

Answer:                                            
""")

refine_prompt = PromptTemplate.from_template("""
You are refining an earlier answer based on new information from the codebase.

-----------------

Developer Request:
{question}

Previous Answer:
{existing_answer}

New Context:
{context}

Instructions:
- If the New Context contains information related to Developer Request, and the New Context is also valueble for the Developer Request, then improve your Previous Answer with information from the New Context.
- If the Previous Answer is not valuable for the Developer Request clearly, and the New Context is not valuable for the Developer Request, state clearly that you dont know the Answer.
- If the Previous Answer is correct, and the New Context is valuable for the Developer Request, then base your Refined Answer on the New Context only.
- Quote exact variable names, values, files or lines if applicable.
- Always point the developer to the file where the information is found.                                             
- Do NOT mention anything about your Context, NOT you previous iterations and conclusions or how you are instruncted to Answer or changing your Answer.
- Keep the Answer informative, concise and maximally relevant to the Developer Request.                                            
- Output your improved Answer in Refined Answer pretending this is the only Answer.

Refined Answer:                                                                                    
""")


# 🔗 Build QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine",
    return_source_documents=True,
    chain_type_kwargs={
        "question_prompt": initial_prompt,
        "refine_prompt": refine_prompt,
        "document_variable_name": "context",
        # "verbose": True
    }
)


# 🎯 Main Entry Point
def run_query(question: str):
    print(f"🔎 Question: {question}")
    result = qa_chain.invoke({"query": question})
    print(f"💬 Answer: {result['result']}")

    source_counter = Counter(
     doc.metadata.get("source", "[no source]") for doc in result["source_documents"]
    )
    print(f"\n📁 Files used:")
    for source, count in source_counter.items():
        print(f"- {source} (referenced {count} time{'s' if count > 1 else ''})")