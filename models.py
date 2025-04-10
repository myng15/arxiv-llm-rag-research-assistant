from unsloth import FastLanguageModel, FastModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()  # Looks for .env in current directory by default
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not hf_token:
    raise ValueError("Hugging Face token not found in .env file")

if not groq_api_key:
    raise ValueError("Groq API token not found in .env file")

if not openai_api_key:
    raise ValueError("OpenAI API token not found in .env file")

def get_hf_model(model_name, max_seq_length, dtype, load_in_4bit):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, #"unsloth/gemma-3-4b-it", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        #token = hf_token, # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # When Unsloth returns a wrapper object with tokenizer inside it
    if model_name.startswith("unsloth/gemma-3-4b-it"): #"unsloth/gemma-3-4b-it-unsloth-bnb-4bit", "unsloth/gemma-3-4b-it"
    #if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        tokenizer.name_or_path = model_name

    if model_name.startswith("mieng155/gemma-3-4b-it"):
        base_model_id = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    if "codegemma" in model_name: #"google/codegemma-7b-it"
        context_window = 2048
    else:
        context_window = 8192
    

    return HuggingFaceLLM(model=model, tokenizer=tokenizer, context_window=context_window, max_new_tokens=max_seq_length,
                        # Tokenizer-specific settings
                        # tokenizer_kwargs={
                        #     "model_input_names": ["input_ids", "attention_mask"],  
                        #     "add_special_tokens": False, 
                        # },
                        # # Generation-specific settings
                        # generate_kwargs={
                        #     "pad_token_id": tokenizer.eos_token_id, 
                        # },
                        )

def get_groq_model(model_name):
    return Groq(model=model_name, api_key=groq_api_key)


def get_openai_model(model_name):
    return OpenAI(model=model_name, api_key=openai_api_key)
