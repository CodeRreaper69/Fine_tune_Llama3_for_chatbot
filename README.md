# SynBot: Fine-Tuning LLama 3 8B for production level chatbot creation

## Overview

This project demonstrates the process of fine-tuning a LLama 3 8B language model with LoRA adapters to create a chatbot, SynBot, optimized for consultancy services. The workflow includes setting up the environment, training the model on a custom dataset, and running inference to interact with the chatbot.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Testing the Model](#testing-the-model)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.x
- GPU with CUDA support (optional but recommended for training)
- Internet access for downloading models and datasets

## Setup and Installation

1. **Install Dependencies:**
   - The following dependencies are required for this project. Use the provided installation commands to set up your environment:
     ```bash
     
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
      ( # the above lines are for testing your system compatibility and installing the best resources ad optimizing it if using in colab )
     
     pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
     pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
     ```

2. **Environment Configuration:**
   - Ensure that you have the necessary libraries for working with the model:
     ```bash
     pip install transformers
     ```

## Data Preparation

1. **Define the Prompt Template:**
   - Create a prompt template used for training the chatbot:
     ```python
     SynBot_prompt = """You are a Syntalix.AI consultancy chatbot. Below is a scenario describing a visitor's query, paired with some context. Write a response that appropriately addresses the visitor's needs.

     ### Scenario:
     {}

     ### Context:
     {}

     ### Response:
     """
     ```

2. **Load and Format Dataset:**
   - Load your custom dataset and format it according to the prompt template. The dataset should be in JSON format.
     ```python
     from datasets import load_dataset
     dataset = load_dataset("json", data_files="syntalixai_finetuning_data.json", split="train")
     dataset = dataset.map(formatting_prompts_func, batched=True)
     ```

## Training the Model

1. **Load the Base Model:**
   - Use the quantized LLama 3 8B model:
     ```python
     from unsloth import FastLanguageModel
     model, tokenizer = FastLanguageModel.from_pretrained(
         model_name="unsloth/llama-3-8b-bnb-4bit",
         max_seq_length=2048,
         load_in_4bit=True
     )
     ```

2. **Integrate LoRA Adapters:**
   - Apply LoRA adapters to optimize the training process:
     ```python
     model = FastLanguageModel.get_peft_model(
         model,
         r=16,
         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
         lora_alpha=16,
         lora_dropout=0,
         bias="none"
     )
     ```

3. **Configure and Run Training:**
   - Set up the training parameters and start training:
     ```python
     from trl import SFTTrainer
     from transformers import TrainingArguments

     trainer = SFTTrainer(
         model=model,
         tokenizer=tokenizer,
         train_dataset=dataset,
         dataset_text_field="text",
         max_seq_length=2048,
         args=TrainingArguments(
             per_device_train_batch_size=2,
             gradient_accumulation_steps=4,
             warmup_steps=5,
             max_steps=100,
             num_train_epochs=10,
             learning_rate=2e-4,
             fp16=True,
             logging_steps=1,
             optim="adamw_8bit",
             weight_decay=0.01,
             lr_scheduler_type="linear",
             seed=3407,
             output_dir="outputs"
         )
     )
     trainer.train()
     ```

## Running Inference

1. **Load the Fine-Tuned Model:**
   - After training, load the model for inference:
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     from peft import PeftModel

     base_model_name = "unsloth/llama-3-8b-bnb-4bit"
     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
     base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
     model_name = "GenDey/SynBot_Model" # this is the fine tuned Lora_adapters for the model I uploaded on hugging face link - https://huggingface.co/GenDey/SynBot_Model/tree/main 
     model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
     ```

2. **Interact with the Chatbot:**
   - Implement a loop to interact with the chatbot:
     ```python
     FastLanguageModel.for_inference(model)
     print("\n---WELCOME SYNTALIX CHATBOT SERVICES---\n (Enter q to exit)\n")

     while True:
         Query = input("YOU <*><*>      : ")
         if Query == 'q':
           break

         inputs = tokenizer(
             [
                 SynBot_prompt.format(
                     f"{Query}",
                     "",
                     ""
                 )
             ], return_tensors="pt").to("cuda")

         text_streamer = TextStreamer(tokenizer)
         generated_output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=200, use_cache=True)

         generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
         response_start = generated_text.find("### Response:") + len("### Response:")
         response_text = generated_text[response_start:].strip()

         print("\nSynBot <`><`>   :", response_text, "\n")
     ```

## Saving and Loading the Model

1. **Save the Model:**
   - Save the fine-tuned model to Hugging Face's hub or locally:
     ```python
     from huggingface_hub import notebook_login
     notebook_login("YOUR_HUGGINGFACE_TOKEN")
     model.push_to_hub("GenDey/SynBot_Model", token="YOUR_HUGGINGFACE_TOKEN")
     ```

2. **Load the Saved Model:**
   - Load the model for further use or deployment:
     ```python
     from transformers import AutoModelForCausalLM
     model = AutoModelForCausalLM.from_pretrained("GenDey/SynBot_Model", low_cpu_mem_usage=False)
     ```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any sections as needed based on your specific requirements or additional features.
