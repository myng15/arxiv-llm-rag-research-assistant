import streamlit as st
import os
import time 

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from llama_index.core.schema import NodeRelationship

from agents.query_agent import QueryAgent  
from agents.code_agent import CodeAgent  
from agents.router_agent import RouterAgent  
from agents.multimodal_rag_agent import MultimodalRAGAgent 
import utils
import constants

from multiprocessing import Pool

import base64
from PIL import Image
from io import BytesIO

# Explicitly set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit config
st.set_page_config(
    page_title="AI ArXiv Research Assistant",
    page_icon= "https://www.svgrepo.com/show/469627/research.svg",
    layout="wide", 
    initial_sidebar_state="auto",
)

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio(
        "Select Task",
        ["General Text-based RAG Query", "Multimodal RAG Query"],
        index=0,
        help="Choose between general text-based search or paper-specific multimodal analysis"
    )
    
# ------------------------ MAIN CONTENT ------------------------

st.title("AI ArXiv Research Assistant")

## ------------------------ GENERAL QUERY SECTION ------------------------
if app_mode == "General Text-based RAG Query":
    st.header("General Text-based RAG Query")
    
    with st.container(border=True):
        # General search query input
        query = st.text_input(
            "Search the AI Research Landscape", 
            placeholder="Enter your AI-related question...",
            key="general_query"
        )

        # Model selection
        model_option = st.selectbox(
            "Select a Large Language Model for answer generation",
            options=[
                "OpenAI GPT-4o",
                "Gemma 3 (4B) Instruct - 4-bit quantized",
                "Gemma 3 (4B) Instruct",
                "Gemma 2 (9B) Instruct",
                "Llama 3.1 (8B) Instruct - 4-bit quantized",
                "Llama 3.1 (8B) Instruct",
                "Gemma 3 (4B) Instruct - Fine-tuned for Q&A about ArXiv papers",
                "Gemma 3 (4B) Instruct - Fine-tuned for Q&A in German",
                "Llama 3.1 (8B) Instruct - Fine-tuned for Q&A about ArXiv papers",
                "Llama 3.1 (8B) Instruct - Fine-tuned for Q&A in German",
            ]
        )

    # Use reranker
    #use_reranker = st.checkbox("Use a reranker", value=False)

    if query:
        with st.spinner("Generating response..."):
            start_time = time.time()

            # Generate answer
            query_agent = QueryAgent(model_name=constants.LLM_MODELS[model_option])
            code_agent = CodeAgent()
            agents = {
                "query_agent": query_agent,
                "code_agent": code_agent,
            }
            router_agent = RouterAgent(agents=agents)
            response = router_agent.query(query)
            source_nodes = response.source_nodes

            end_time = time.time()

            # Iterate over rows to multiprocess the get_thumbnail function
            if source_nodes:
                pool = Pool(len(source_nodes))
                processes = []
                for node in source_nodes: 
                    paper_id = node.node.relationships.get(NodeRelationship.SOURCE).node_id
                    processes.append(pool.apply_async(utils.get_thumbnail, (utils.get_url(paper_id),)))
                pool.close()  # no more tasks

            # Answer section
            with st.container(border=True):
                st.markdown("### Answer")

                # Display collapsible top-k source papers
                with st.expander("🔍 Retrieved Context Papers"):
                    if not source_nodes:
                        st.info("No context papers retrieved for this query.")
                    else:
                        i = 0
                        for node in source_nodes: 
                            paper_id = node.node.relationships.get(NodeRelationship.SOURCE).node_id
                            paper_url = utils.get_url(paper_id)

                            paper_title, paper_abstract = node.text.split('\n', 1)
                            st.markdown(f'#### {paper_title}')
                            #st.markdown(f'[Read Paper]({paper_url})')
                            st.markdown(f"""
                                <div style="
                                    display: inline-block;
                                    padding: 0.6em 1.2em;
                                    border-radius: 25px;
                                    background-color: #1E88E5;
                                    color: white !important;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                    transition: all 0.3s;
                                    margin: 5px;
                                    cursor: pointer;
                                "
                                onmouseover="this.style.backgroundColor='#1565C0'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.3)';"
                                onmouseout="this.style.backgroundColor='#1E88E5'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.2)';"
                                >
                                    <a href="{paper_url}" target="_blank" style="color: white; text-decoration: none;">
                                        📄 Read Paper
                                    </a>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(paper_abstract)

                            # TODO: Display paper's "authors" ("authors_parsed") and "categories"
                            
                            # Display thumbnails of context papers
                            thumbnail = processes[i].get()  # get the result from the process
                            i += 1
                            if thumbnail:
                                st.image(thumbnail, use_container_width=True)
                                st.caption("Paper preview")

                            st.markdown("---")
                
                # Display generated answer
                st.markdown(response, unsafe_allow_html=True)

            
            st.success(f"Answer generation completed in {end_time - start_time:.2f} seconds")

## ------------------------ MULTIMODAL RAG SECTION ------------------------
elif app_mode == "Multimodal RAG Query":
    st.header("Multimodal RAG Query")

    with st.container(border=True):
        # Paper URL input
        paper_url = st.text_input(
            "Paper URL", 
            placeholder="Enter paper URL...",
            key="paper_url")

        # Paper-specific search query
        multimodal_query = st.text_input(
            "Ask about paper", 
            placeholder="Enter your question about the paper's content...",
            key="multimodal_query")
               
                
        if paper_url and multimodal_query:
            with st.spinner("Generating response..."):
                # Re-generate the URL in case the user enters the abstract URL instead of the PDF URL
                paper_id = paper_url.split("/")[-1]
                paper_url = utils.get_url(paper_id)

                start_time = time.time()
                multimodal_rag_agent = MultimodalRAGAgent(paper_id)
                result = multimodal_rag_agent.query(multimodal_query, return_sources=True)
                end_time = time.time()
                
                # Display the context images in the result
                def display_base64_image(base64_code, caption=None, width=300): 
                    # Decode the base64 string and convert to image
                    image_data = base64.b64decode(base64_code) # base64 -> binary
                    image = Image.open(BytesIO(image_data)) # binary -> image
                    
                    st.image(image, caption=caption, width=width)

                # Answer section
                with st.container(border=True):
                    thumbnail = utils.get_thumbnail(paper_url)
                    if thumbnail:
                        st.image(thumbnail, use_container_width=True)
                        st.caption("Paper preview")

                    st.markdown("### Answer")

                    # Display generated answer
                    st.markdown(result["response"], unsafe_allow_html=True)
                    
                    # Display context images
                    context_images = result["context"]["images"]
                    if context_images:
                        for img in context_images:
                            display_base64_image(img)
                    else:
                        st.info("There are no context images for this answer.")

                st.success(f"Answer generation completed in {end_time - start_time:.2f} seconds")

# ------------------------ END ------------------------

st.markdown("---")
st.caption("This is a prototype AI research assistant using Large Language Models, multimodal RAG and Streamlit.")



