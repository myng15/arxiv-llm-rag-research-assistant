import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import pdf_dir, rag_dir
from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from base64 import b64decode
import json

load_dotenv()  # Looks for .env in current directory by default
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class MultimodalRAGAgent:
    def __init__(self, arxiv_id: str):
        self.arxiv_id = arxiv_id
        self.pdf_path = os.path.join(pdf_dir, arxiv_id + ".pdf")
        self.retriever = None
        self._build_retriever()

    def _build_retriever(self):
        chunks = partition_pdf(
            filename=self.pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"], # Extract image of figures and tables in the paper
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )

        self.texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
        self.images = self._extract_images(self.texts)
        
        summary_dir = os.path.join(rag_dir, "generated_summaries")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"{self.arxiv_id}.json")

        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summaries = json.load(f)
                self.text_summaries = summaries.get("text_summaries", [])
                self.image_summaries = summaries.get("image_summaries", [])
        else:
            self.text_summaries = self._summarize_texts(self.texts)
            self.image_summaries = self._summarize_images(self.images)

            with open(summary_path, "w") as f:
                json.dump({
                    "text_summaries": self.text_summaries,
                    "image_summaries": self.image_summaries
                }, f, indent=2)

        self.retriever = self._build_vector_retriever()

    # Get the images from the CompositeElement objects
    def _extract_images(self, chunks):
        images_b64 = []
        for chunk in chunks:
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)) or "Table" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
        return images_b64

    def _summarize_texts(self, texts):
        prompt = ChatPromptTemplate.from_template("""
            You are an assistant tasked with summarizing texts from ArXiv papers.
            Give a concise summary of the texts.
            Respond only with the summary, no additional comment like "Here is a summary...".
            
            The text chunk to be summarized: {element}
        """)

        model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        return summarize_chain.batch(texts, {"max_concurrency": 1})

    def _summarize_images(self, images):
        prompt_template = """Describe the image in detail. For context,
        the image is part of an ArXiv research paper in the field of AI and machine learning.
        Be specific about graphs, plots, diagrams and tables."""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

        return chain.batch(images) if len(images) > 0 else []

    def _build_vector_retriever(self):
        vectorstore = Chroma(collection_name="arxiv_multimodal_rag", embedding_function=OpenAIEmbeddings())
        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

        # Add text summaries
        doc_ids = [str(uuid.uuid4()) for _ in self.texts]
        text_summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(self.text_summaries)
        ]
        retriever.vectorstore.add_documents(text_summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, self.texts)))

        # Add image summaries
        img_ids = [str(uuid.uuid4()) for _ in self.images]
        img_summary_docs = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(self.image_summaries)
        ]
        retriever.vectorstore.add_documents(img_summary_docs)
        retriever.docstore.mset(list(zip(img_ids, self.images)))

        return retriever

    def _parse_docs(self, docs):
            b64, text = [], []
            for doc in docs:
                try:
                    b64decode(doc)
                    b64.append(doc)
                except Exception:
                    text.append(doc)
            return {"images": b64, "texts": text}

    def _build_prompt(self, kwargs):
        """Build prompt with context (including images)"""
        docs_by_type = kwargs["context"]
        user_query = kwargs["query"]

        context_text = "".join([t.text for t in docs_by_type["texts"]])
        
        prompt_template = f"""
        Answer the query based only on the following context, which can include text, tables, and the following images.

        Context: {context_text}
        Query: {user_query}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]
        
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })

        return ChatPromptTemplate.from_messages([
                HumanMessage(content=prompt_content),
            ])

    def query(self, query_str: str, return_sources=True):
        chain = {
            "context": self.retriever | RunnableLambda(self._parse_docs),
            "query": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | ChatOpenAI(model="gpt-4o-mini")
                | StrOutputParser()
            )
        )

        result = chain.invoke(query_str)
        return result if return_sources else result['response']


# TEST
# import utils
# from config import rag_dir

# paper_url = "https://arxiv.org/abs/1706.03762"
# paper_id = paper_url.split("/")[-1]
# paper_url = utils.get_url(paper_id) 

# import base64
# from PIL import Image
# from io import BytesIO

# def save_base64_image(base64_code, image_path):
#     # Decode the base64 string to binary
#     image_data = base64.b64decode(base64_code)
#     # Convert binary data to an image
#     image = Image.open(BytesIO(image_data))
#     # Save the image to the specified path
#     image.save(image_path)


# multimodal_query = "What is the attention mechanism?"
# multimodal_rag_agent = MultimodalRAGAgent(paper_id)
# result = multimodal_rag_agent.query(multimodal_query, return_sources=True)
# #Debug
# print(result)

# print("\n\nResponse of chain_with_sources.invoke: ", result['response'])

# print("\n\nContext:\n\n")
# for text in result['context']['texts']:
#     print(text.text)
#     print("Page number: ", text.metadata.page_number)
#     print("\n" + "-"*50 + "\n")
# for i, image in enumerate(result['context']['images']):
#     image_path = os.path.join(rag_dir, paper_id + f"-context_image_{i}.png")
#     save_base64_image(image, image_path)
