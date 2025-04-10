import os

data_dir = "data"
thumbnail_dir = os.path.join(data_dir, "thumbnails")
pdf_dir = os.path.join(data_dir, "downloaded_papers")
rag_dir = os.path.join(data_dir, "rag_outputs")

finetuning_output_dir = os.path.join("finetuning", "output")

# Create the directories if not yet exist
for dir in [data_dir, thumbnail_dir, pdf_dir, rag_dir, finetuning_output_dir]:
    os.makedirs(dir, exist_ok=True)


QUERY_AGENT_CONFIG = {
    "use_groq": False,
    "model_cache_dir": "./models", 
    "embed_model": "BAAI/bge-small-en-v1.5", #"openai"
    "llm_model": "unsloth/gemma-3-4b-it-unsloth-bnb-4bit", # "gemma2-9b-it"
    "max_seq_length": 2048,
    "dtype": None,
    "load_in_4bit": True,
    "chromadb_dir": "./chroma_db", 
    "chroma_collection": "arxiv_papers",
    "reranker_model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "top_k_retrieval": 5,
    "top_k_rerank": 3,
    "system_prompt": """
        You are an expert research assistant specializing in machine learning and artificial intelligence. 
        Your task involves processing a collection of recent research paper abstracts. Perform the following:

        1. Provide a coherent and comprehensive summary of these abstracts tailored to the user's specific query.
        2. Identify and summarize the key themes, methodologies, and implications of the research findings, highlighting any significant trends or innovations.
    """,
    "user_prompt": """ 
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        ---------------------
        Query: {query_str}
        ---------------------
        Answer: 
    """,
}

CODE_AGENT_CONFIG = {
    "use_groq": False,
    "llm_model": "google/codegemma-7b-it",
    "max_seq_length": 2048,
    "dtype": None,
    "load_in_4bit": True,
    "system_prompt": """
        You are a code assistant powered by a large language model. Your task is to help users implement algorithms from research papers, solve programming problems, provide code examples, explain programming concepts, and debug code. Perform the following:

        1. Provide a detailed code implementation or code-based answer tailored to the user's specific query and based on the original research papers (if any).
        2. Provide a concise, intuitive explanation as well as information about the original source of your code implementation (if any).
    """,
    "user_prompt": """ 
        Write Python code to answer the question below.
        ---------------------
        Query: {query_str}
        ---------------------
        Answer: 
    """,
}

ROUTER_AGENT_CONFIG = {
    "use_groq": True,
    "llm_model": "gemma2-9b-it",
    "max_seq_length": 2048,
    "dtype": None,
    "load_in_4bit": True,
}


