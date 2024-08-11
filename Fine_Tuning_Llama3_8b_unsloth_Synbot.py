# -*- coding: utf-8 -*-
"""Copy of Fine-Tuning LLama 3 8B.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VDweKBzifAMOesPlSzB5Ni9AYzgptRPf

# SYNTALIX CHATBOT CREATION

First we check the GPU version available in the environment and install specific dependencies that are compatible with the detected GPU to prevent version conflicts.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import torch
# major_version, minor_version = torch.cuda.get_device_capability()
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# if major_version >= 8:
#     !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
# else:
#     !pip install --no-deps xformers trl peft accelerate bitsandbytes
# pass
# #%%capture
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# #!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

"""Next we need to prepare to load a range of quantized language models, including a new 15 trillion token LLama-3 model, optimized for memory efficiency with 4-bit quantization.

"""

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! Llama 3 is up to 8k
dtype = None
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", # Llama-3 70b also works (just change the model name)
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""---

Next, we integrate LoRA adapters into our model, which allows us to efficiently update just a fraction of the model's parameters, enhancing training speed and reducing computational load.
"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

"""<a name="Data"></a>
### Data Prep
We now use the Alpaca dataset from [yahma](https://huggingface.co/datasets/yahma/alpaca-cleaned), which is a filtered version of 52K of the original [Alpaca dataset](https://crfm.stanford.edu/2023/03/13/alpaca.html). You can replace this code section with your own data prep.

Then, we define a system prompt that formats tasks into instructions, inputs, and responses, and apply it to a dataset to prepare our inputs and outputs for the model, with an EOS token to signal completion.

"""

# this is basically the system prompt
SynBot_prompt = """You are a Syntalix.AI consultancy chatbot. Below is a scenario describing a visitor's query, paired with some context. Write a response that appropriately addresses the visitor's needs.

### Scenario:
{}

### Context:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # do not forget this part!
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = SynBot_prompt.format(instruction, input, output) + EOS_TOKEN # without this token generation goes on forever!
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
#dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = load_dataset("json", data_files="syntalixai_finetuning_data.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

"""<a name="Train"></a>
### Train the model
- We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
- At this stage, we're configuring our model's training setup, where we define things like batch size and learning rate, to teach our model effectively with the data we have prepared.
"""

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # increase this to make the model learn "better"
        num_train_epochs=10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# We're now kicking off the actual training of our model, which will spit out some statistics showing us how well it learns
trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""<a name="Inference"></a>
### Inference
Let's run the model! You can change the instruction and input - leave the output blank!
"""

FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    SynBot_prompt.format(
        "Can you name the team members of your company?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 200, use_cache = True)
tokenizer.batch_decode(outputs)

""" CHAT WITH THE TRAINED BOT"""

from transformers import TextStreamer
import io
import sys

# Define a function to suppress output
class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

FastLanguageModel.for_inference(model)
print("\n---WELCOME SYNTALIX CHATBOT EXPERIENCE---\n (press q to exit)\n")

while True:
    Query = input("YOU <*><*> : ")
    if Query == 'q':
      break
    Input = input("Wanna add some EXTRAS (this is highly EXPERIMENTAL, leave blank if no or q to exit, generally for more specific query, like some document or pdf upload query):\n ")
    if Input == 'q':
        break
    inputs = tokenizer(
        [
            SynBot_prompt.format(
                f"{Query}",  # instruction or Query
                f"{Input}",  # input or
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    with SuppressOutput():  # Suppress the output from model.generate()
        text_streamer = TextStreamer(tokenizer)
        generated_output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=200)

    # Convert generated tokens back to text
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    # Extract the Response part
    response_start = generated_text.find("### Response:") + len("### Response:")
    response_text = generated_text[response_start:].strip()

    print("\nSynBot:<`><`>: ",response_text,"\n")
    ch = input("Continue? (y/n): ")
    if ch.lower() == "n":
        break

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

#model.save_pretrained("SynBot_Model") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
from huggingface_hub import notebook_login
from google.colab import userdata
key = userdata.get('HugTok')
notebook_login(key)
model.push_to_hub("GenDey/SynBot_Model", token = key)

!pip install transformers

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("GenDey/SynBot_Model", low_cpu_mem_usage=False)