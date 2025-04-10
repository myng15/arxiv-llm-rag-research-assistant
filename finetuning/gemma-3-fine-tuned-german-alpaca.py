from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import logging
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import data_dir, finetuning_output_dir

from dotenv import load_dotenv

load_dotenv()  # Looks for .env in current directory by default
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Verify token exists
if not hf_token:
    raise ValueError("Hugging Face token not found in .env file")

if not groq_api_key:
    raise ValueError("Groq API token not found in .env file")


###############
# Setup logging
###############
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

#####################################
# Load pretrained model and tokenizer
#####################################
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048, 
    load_in_4bit = True,  
    load_in_8bit = False, # whether to use 2x memory (a bit more accurate)
    full_finetuning = False, # whether to do full finetuning
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text
    finetune_language_layers   = True,  # Should leave on
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

###############
# Load datasets
###############
raw_data_path = os.path.join(data_dir, "german_alpaca.json")
processed_data_path = os.path.join(data_dir, "german_alpaca_processed.json")
if os.path.exists(processed_data_path):
    print(f"Loading preprocessed dataset from {processed_data_path}")
    with open(processed_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)

else:
    print(f"Preprocessing dataset from {raw_data_path}")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def create_conversation(example):
        prompt = example["instruction"]
        if example["input"]:
            prompt += f"\n\n{example['input']}"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    # Apply the transformation
    conversations = [create_conversation(item) for item in raw_data]
    dataset = Dataset.from_list(conversations)

    def apply_chat_template(examples):
        texts = tokenizer.apply_chat_template(examples["messages"])
        return { "text" : texts }

    dataset = dataset.map(apply_chat_template, batched=True)

    # Save dataset to disk
    dataset.to_json(processed_data_path, orient="records")
    print(f"Dataset saved to {processed_data_path}")


####################
# Initialize Trainer
####################
training_args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use gradient accumulation to mimic batch size
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "tensorboard", #"none", "wandb"
        output_dir = os.path.join(finetuning_output_dir, "gemma-3-4b-it-german-alpaca-qlora"),
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, 
    args = training_args,
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
#Debug
# print("Masked instruction:\n ", tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
# print("Masked answer:\n ", tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))


###############
# Training loop
###############
def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

# Check for last checkpoint
last_checkpoint = get_checkpoint(training_args)
if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
# log metrics
metrics = train_result.metrics

metrics['train_samples'] = len(dataset)
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()


# Inference
print("Running inference...")
messages = [{
    "role": "user",
    "content": [{"type" : "text", 
                 "text" : "Wie können wir die Luftverschmutzung verringern?",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048, # = max_seq_length
    # Recommended Gemma-3 settings:
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

##################################
# Save model and create model card
##################################
logger.info('*** Save model ***')

# Restore k,v cache for fast inference
trainer.model.config.use_cache = True
model.save_pretrained(training_args.output_dir)  # Local saving
tokenizer.save_pretrained(training_args.output_dir)
logger.info(f'Model and tokenizer saved to {training_args.output_dir}')

trainer.create_model_card({'tags': ['sft', 'gemma-3', 'german-alpaca']})

# model.push_to_hub("mieng155/gemma-3-4b-it-german-alpaca-qlora", token=hf_token) # Online saving
# tokenizer.push_to_hub("mieng155/gemma-3-4b-it-german-alpaca-qlora", token=hf_token) # Online saving
logger.info('Pushing to hub...')
trainer.push_to_hub()

logger.info('*** Training complete! ***')

##################################
# Load saved LoRA adapters for inference
##################################
load_saved_adapters = True
if load_saved_adapters:
    model, tokenizer = FastModel.from_pretrained(
        model_name = "mieng155/gemma-3-4b-it-german-alpaca-qlora", 
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", 
                 "text" : "Diesen Code überarbeiten und eine neue Version erstellen: def faktorialisieren (num): \nfaktoriell = 1 \nfür i in der Reichweite (1, num): \nfaktoriell * = i \n\nfaktoriell zurückliefern",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048, # = max_seq_length
    # Recommended Gemma-3 settings:
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)