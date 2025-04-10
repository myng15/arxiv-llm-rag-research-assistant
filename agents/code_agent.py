from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Response

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_hf_model, get_groq_model
from config import CODE_AGENT_CONFIG

class CodeAgent:
    def __init__(self):
        self.llm = self._load_model()
        self.prompt_template = self._setup_prompt_template() 
        
    def _load_model(self):
        if CODE_AGENT_CONFIG["use_groq"]:
            return get_groq_model(model_name=CODE_AGENT_CONFIG["llm_model"])
        return get_hf_model(
            model_name=CODE_AGENT_CONFIG["llm_model"],
            max_seq_length=CODE_AGENT_CONFIG["max_seq_length"],
            dtype=CODE_AGENT_CONFIG["dtype"],
            load_in_4bit=CODE_AGENT_CONFIG["load_in_4bit"]
        )
    
    def _setup_prompt_template(self):
        system_prompt = CODE_AGENT_CONFIG["system_prompt"]
        user_prompt = CODE_AGENT_CONFIG["user_prompt"]

        message_template = [
            ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
            ChatMessage(content=user_prompt, role=MessageRole.USER)
        ]

        return ChatPromptTemplate(message_template)

    def query(self, query_str: str):
        formatted_prompt = self.prompt_template.format(query_str=query_str)
        response = self.llm.complete(formatted_prompt)
        return Response(response=str(response), metadata={}) #return str(response)

# TEST
# code_query_agent = CodeAgent() 
# response = code_query_agent.query("How to code a simple video diffusion model?")
# print(response)