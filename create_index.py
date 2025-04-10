from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import torch
import os
import pandas as pd

# GET AND CLEAN DATA
def clean_text(x):
    # Replace newline characters with a space
    new_text = " ".join([c.strip() for c in x.replace("\n", "").split()])
    # Remove leading and trailing spaces
    new_text = new_text.strip()
    
    return new_text

# CREATE INDEX FROM DATA
Settings.llm = None

# Create embed model
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder="./models", device=device_type)
# TODO: Try: embed_model = OpenAIEmbedding()

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("arxiv_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Check if collection is empty
if chroma_collection.count() == 0:
    print("Creating new index...")

    data_dir = "data/"
    processed_data_path=os.path.join(data_dir, "arxiv_processed.csv")

    if not os.path.exists(processed_data_path):
        print("Processed data file not found!")

    print("Loading processed data...")
    df_data = pd.read_csv(processed_data_path, dtype={'id': str})  

    df_data['title'] = df_data['title'].apply(clean_text)
    df_data['abstract'] = df_data['abstract'].apply(clean_text)
    df_data['prepared_text'] = df_data['title'] + '\n ' + df_data['abstract']

    arxiv_documents = [Document(text=prepared_text, doc_id=id) 
                       for prepared_text, id in list(zip(df_data['prepared_text'], df_data['id']))]

    # Create storage context and index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        arxiv_documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )

else:
    print("Index already exists!")

