from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
# from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.groq import Groq
# from unsloth import FastLanguageModel
import os
from typing import Any, Dict, List, Optional, Sequence, cast

from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.output_parsers.base import StructuredOutput
from llama_index.core.output_parsers.selection import Answer, SelectionOutputParser
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors.prompts import (
    DEFAULT_MULTI_SELECT_PROMPT_TMPL,
    DEFAULT_SINGLE_SELECT_PROMPT_TMPL,
    MultiSelectPrompt,
    SingleSelectPrompt,
)
from llama_index.core.service_context import ServiceContext
# from llama_index.core.service_context_elements.llm_predictor import (
#     LLMPredictorType,
# )
from llama_index.core.llms import LLM
from llama_index.core import Settings #, llm_from_settings_or_context
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.types import BaseOutputParser

from dotenv import load_dotenv

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import get_hf_model, get_groq_model
from config import ROUTER_AGENT_CONFIG 


load_dotenv()  # Looks for .env in current directory by default
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Verify token exists
if not hf_token:
    raise ValueError("Hugging Face token not found in .env file")

if not groq_api_key:
    raise ValueError("Groq API token not found in .env file")


def _build_choices_text(choices: Sequence[ToolMetadata]) -> str:
    """Convert sequence of metadata to enumeration text."""
    texts: List[str] = []
    for ind, choice in enumerate(choices):
        text = " ".join(choice.description.splitlines())
        text = f"({ind + 1}) {text}"  # to one indexing
        texts.append(text)
    return "\n\n".join(texts)


def _structured_output_to_selector_result(output: Any) -> SelectorResult:
    """Convert structured output to selector result."""
    structured_output = cast(StructuredOutput, output)
    answers = cast(List[Answer], structured_output.parsed_output)

    selections = []
    for answer in answers:
        if answer.choice is None:
            print(f"Warning: No valid choice selected. Reason: {answer.reason}")
            continue  
        
        # Adjust for zero indexing
        selections.append(SingleSelection(index=answer.choice - 1, reason=answer.reason))

    # Fall back to default option when no choice is selected
    if not selections:
        default_reason = "No confident selection made. Falling back to Query Engine as default."
        selections.append(SingleSelection(index=0, reason=default_reason))

    return SelectorResult(selections=selections)


class LLMSingleSelectorCustom(BaseSelector):
    """LLM single selector.

    LLM-based selector that chooses one out of many options.

    Args:
        LLM (LLM): An LLM.
        prompt (SingleSelectPrompt): A LLM prompt for selecting one out of many options.
    """

    def __init__(
        self,
        llm: LLM, #LLMPredictorType,
        prompt: SingleSelectPrompt,
    ) -> None:
        self._llm = llm 
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLM] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMSingleSelector":
        # optionally initialize defaults
        llm = llm #or Settings.llm 
        if llm is None:
            raise ValueError("No LLM provided or found in Settings")
    
        prompt_template_str = prompt_template_str or DEFAULT_SINGLE_SELECT_PROMPT_TMPL
        output_parser = output_parser or SelectionOutputParser()

        # construct prompt
        prompt = SingleSelectPrompt(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SINGLE_SELECT,
        )
        return cls(llm, prompt)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction = self._llm.predict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )
        print(prediction)

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parse)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        choices_text = _build_choices_text(choices)

        # predict
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            num_choices=len(choices),
            context_list=choices_text,
            query_str=query.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_selector_result(parse)


class RouterAgent:
    def __init__(self, agents: dict): #, llm: Optional[LLM] = None
        #self.llm_model = llm or Settings.llm

        self.model_name = ROUTER_AGENT_CONFIG["llm_model"]
        self.use_groq = ROUTER_AGENT_CONFIG["use_groq"]

        # Setup LLM
        Settings.llm = None
        self.llm_model = self._load_model()

        if self.llm_model is None:
            raise ValueError("No LLM provided and none found in Settings")

        self.query_agent = agents["query_agent"]
        self.code_agent = agents["code_agent"]

        # Create tools from agents
        self.query_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_agent,
            description="Useful for searching papers and answering questions related to the ArXiv papers and relationships between them",
        )

        self.code_tool = QueryEngineTool.from_defaults(
            query_engine=self.code_agent,
            description="Useful for answering coding questions",
        )

        # Custom LLM selector with the provided LLM
        selector = LLMSingleSelectorCustom.from_defaults(llm=self.llm_model)

        # Router Query Engine
        self.engine = RouterQueryEngine(
            selector=selector,
            query_engine_tools=[
                self.query_tool, 
                self.code_tool
            ],
            verbose=True,
        )
    
    def _load_model(self):
        if self.use_groq:
            groq_llm = get_groq_model(self.model_name)
            Settings.llm = groq_llm
            return groq_llm
        return get_hf_model(
            model_name=self.model_name,
            max_seq_length=ROUTER_AGENT_CONFIG["max_seq_length"],
            dtype=ROUTER_AGENT_CONFIG["dtype"],
            load_in_4bit=ROUTER_AGENT_CONFIG["load_in_4bit"]
        )

    def query(self, query_str: str):
        return self.engine.query(query_str)

#TEST
# router_agent = RouterAgent()
# query = "How to do research for anomaly detection?"
# response = router_agent.query(query)

# # Show response to query 
# print(f"\nResponse: {response}")

# # Show retrieval result of the query engine
# print("\nRetrieved Contexts:")
# for node in response.source_nodes:
#     print(node.text)



