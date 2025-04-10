import torch
import chromadb
from llama_index.core import (
    VectorStoreIndex, Settings, StorageContext,
    ChatPromptTemplate, PromptTemplate
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_hf_model, get_groq_model, get_openai_model
from config import QUERY_AGENT_CONFIG  


class QueryAgent:
    def __init__(self, model_name=None):
        self.model_name = model_name or QUERY_AGENT_CONFIG["llm_model"]
        self.use_groq = True if self.model_name in ["gemma2-9b-it"] else False #QUERY_AGENT_CONFIG["use_groq"]
        self.use_openai = self.model_name.startswith("gpt-")

        # Setup LLM
        Settings.llm = None
        self.llm_model = self._load_model()

        # Setup embed model
        device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = OpenAIEmbedding() if QUERY_AGENT_CONFIG["embed_model"] == "openai" else HuggingFaceEmbedding(
                                        model_name=QUERY_AGENT_CONFIG["embed_model"],
                                        cache_folder=QUERY_AGENT_CONFIG["model_cache_dir"],
                                        device=device_type
                                    ) 

        # Setup vector store index
        self.index = self._load_index()

        # Set up reranker (optional)
        self.reranker = SentenceTransformerRerank(
            model=QUERY_AGENT_CONFIG["reranker_model"],
            top_n=QUERY_AGENT_CONFIG["top_k_rerank"],
            keep_retrieval_score=True
        )

        # Setup prompts
        self.prompt_template = self._setup_prompt_template()

        # Setup query engine
        self.engine = self.index.as_query_engine(
            similarity_top_k=QUERY_AGENT_CONFIG["top_k_retrieval"],
            llm=self.llm_model,
            #node_postprocessors=[self.reranker]
            streaming=True, 
        )
        self.engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.prompt_template})

    def _load_model(self):
        if self.use_groq:
            groq_llm = get_groq_model(self.model_name)
            Settings.llm = groq_llm
            return groq_llm
        elif self.use_openai:  
            openai_llm = get_openai_model(self.model_name)
            Settings.llm = openai_llm
            return openai_llm
        return get_hf_model(
            model_name=self.model_name,
            max_seq_length=QUERY_AGENT_CONFIG["max_seq_length"],
            dtype=QUERY_AGENT_CONFIG["dtype"],
            load_in_4bit=QUERY_AGENT_CONFIG["load_in_4bit"]
        )

    def _load_index(self):
        chroma_client = chromadb.PersistentClient(path=QUERY_AGENT_CONFIG["chromadb_dir"])
        chroma_collection = chroma_client.get_or_create_collection(QUERY_AGENT_CONFIG["chroma_collection"])
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )

    def _setup_prompt_template(self):
        system_prompt = QUERY_AGENT_CONFIG["system_prompt"]
        user_prompt = QUERY_AGENT_CONFIG["user_prompt"]

        message_template = [
            ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
            ChatMessage(content=user_prompt, role=MessageRole.USER)
        ]
        
        return ChatPromptTemplate(message_template)

    def query(self, query_str: str):
        return self.engine.query(query_str)

    def retrieve(self, query_str: str):
        return self.engine.retrieve(query_str)


