import torch
import subprocess
import os
import requests
from pathlib import Path
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# === CONFIG ===
PDF_DIR = "pdf_docs"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-finetuned"
MERGED_DIR = "./tinyllama-merged"
GGUF_OUTPUT = "./tinyllama-chat.gguf"
MAX_LENGTH = 512

# === STEP 1: Load data from JSON file ===
def load_json_dataset(json_file_path):
    """
    Load conversation data from a JSON file containing chat messages.
    Expected format: List of dictionaries with 'role' and 'content' keys.
    """
    import json
    from pathlib import Path
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} messages from {json_file_path}")
    return data

# === STEP 2: Format conversations for training ===
def format_conversations_for_training(json_data):
    """
    Convert JSON chat data into training format.
    Groups consecutive user-assistant pairs into conversation threads.
    """
    formatted_texts = []
    current_conversation = []
    
    for message in json_data:
        if message["role"] == "user":
            # If we have a complete conversation, save it
            if current_conversation:
                formatted_text = format_conversation_thread(current_conversation)
                if formatted_text:
                    formatted_texts.append({"text": formatted_text})
            # Start new conversation
            current_conversation = [message]
        elif message["role"] == "assistant" and current_conversation:
            # Add assistant response to current conversation
            current_conversation.append(message)
    
    # Don't forget the last conversation
    if current_conversation:
        formatted_text = format_conversation_thread(current_conversation)
        if formatted_text:
            formatted_texts.append({"text": formatted_text})
    
    return formatted_texts

def format_conversation_thread(conversation):
    """
    Format a single conversation thread into training format.
    """
    if len(conversation) < 2:
        return None
    
    formatted_parts = []
    for i in range(0, len(conversation), 2):
        if i + 1 < len(conversation):
            user_msg = conversation[i]
            assistant_msg = conversation[i + 1]
            
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                formatted_parts.append(f"<|user|>\n{user_msg['content'].strip()}")
                formatted_parts.append(f"<|assistant|>\n{assistant_msg['content'].strip()}")
    
    if formatted_parts:
        return "\n".join(formatted_parts) + "\n<|endoftext|>"
    return None

# === STEP 3: Alternative format for single message pairs ===
def format_individual_messages(json_data):
    """
    Format each user-assistant pair as individual training examples.
    """
    formatted_texts = []
    
    for i in range(0, len(json_data) - 1, 2):
        if (i + 1 < len(json_data) and 
            json_data[i]["role"] == "user" and 
            json_data[i + 1]["role"] == "assistant"):
            
            user_content = json_data[i]["content"].strip()
            assistant_content = json_data[i + 1]["content"].strip()
            
            formatted_text = f"<|user|>\n{user_content}\n<|assistant|>\n{assistant_content}\n<|endoftext|>"
            formatted_texts.append({"text": formatted_text})
    
    return formatted_texts

# === STEP 4: Prepare Dataset ===
def prepare_dataset(text_data):
    """
    Convert formatted text data into a Dataset object.
    """
    from datasets import Dataset
    return Dataset.from_list(text_data)

# === STEP 5: Main function to process JSON file ===
def process_json_to_dataset(json_file_path, format_type="conversations"):
    """
    Complete pipeline to process JSON file into training dataset.
    
    Args:
        json_file_path: Path to JSON file containing chat data
        format_type: "conversations" for multi-turn or "individual" for single pairs
    """
    # Load JSON data
    json_data = load_json_dataset(json_file_path)
    
    # Format for training
    if format_type == "conversations":
        formatted_data = format_conversations_for_training(json_data)
    else:
        formatted_data = format_individual_messages(json_data)
    
    print(f"Created {len(formatted_data)} training examples")
    
    # Create dataset
    dataset = prepare_dataset(formatted_data)
    
    return dataset

# === STEP 6: Tokenization Function ===
def tokenize_function(example, tokenizer):
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer(
        example["text"], 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length",
        return_tensors=None
    )

# === STEP 7: Merge LoRA weights ===
def merge_lora_weights(base_model_name, lora_path, output_path):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Determine device and dtype
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # Use float32 for better MPS compatibility
        device_map=None if device == "mps" else "auto"
    )
    
    # Move to device if MPS
    if device == "mps":
        base_model = base_model.to(device)
    
    print("Loading LoRA model...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Move to CPU for saving
    merged_model = merged_model.cpu()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

# === STEP 8: Convert to GGUF ===
def convert_to_gguf(model_path, output_file):
    print(f"Converting {model_path} to GGUF format...")
    
    # Try different possible locations for the conversion script
    possible_scripts = [
        "convert_hf_to_gguf.py",
        "/opt/homebrew/bin/convert_hf_to_gguf.py",
        "./llama.cpp/convert_hf_to_gguf.py"
    ]
    
    script_path = None
    for script in possible_scripts:
        if os.path.exists(script):
            script_path = script
            break
    
    if not script_path:
        print("Converting using uv run (assuming llama.cpp is available)...")
        cmd = [
            "uv", "run", "python", "convert_hf_to_gguf.py",
            model_path,
            "--outfile", output_file,
            "--outtype", "f16"  # f16 is better for most use cases
        ]
    else:
        cmd = [
            "python", script_path,
            model_path,
            "--outfile", output_file,
            "--outtype", "f16"
        ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("GGUF conversion successful!")
        print(f"GGUF file saved to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"GGUF conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

# === MAIN ===
def load_dataset(json_file):
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    # Check for MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) on Mac Mini M4")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        device_map=device if device != "mps" else None  # MPS doesn't support device_map
    )
    
    # Move model to MPS if needed
    if device == "mps":
        model = model.to(device)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
       
    dataset = process_json_to_dataset(json_file, format_type="conversations") # individual
    print(f"Dataset created with {len(dataset)} examples")
    
    tokenized_dataset = dataset.map(
        lambda e: tokenize_function(e, tokenizer), 
        batched=True,
        remove_columns=dataset.column_names
    )
    return model, tokenizer, tokenized_dataset

def main(model, tokenizer, tokenized_dataset):
    # Apply PEFT with LoRA
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,  # Increased rank for better performance
        lora_alpha=32,  # Increased alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More comprehensive targeting
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Define training args - MPS optimized
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # M4 can handle larger batches
        gradient_accumulation_steps=2,  # Added for effective larger batch size
        num_train_epochs=3,
        learning_rate=2e-4,  # Explicit learning rate
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=100,  # Added warmup
        fp16=False,  # Disable fp16 for MPS compatibility
        bf16=False,  # Disable bf16 for MPS compatibility  
        dataloader_drop_last=False,
        report_to="none",
        remove_unused_columns=False,
        # Remove use_mps_device as it's not a valid parameter
    )
    
    # Data collator for Causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,  # Fixed: use processing_class instead of tokenizer
        data_collator=data_collator
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save LoRA adapter
    print("Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Merge LoRA weights with base model
    print("Merging LoRA weights with base model...")
    merge_lora_weights(MODEL_NAME, OUTPUT_DIR, MERGED_DIR)
    
    # Convert to GGUF for OpenWebUI
    print("Converting to GGUF format...")
    success = convert_to_gguf(MERGED_DIR, GGUF_OUTPUT)
    
    if success:
        print(f"\n🎉 Success! Your model is ready for OpenWebUI:")
        print(f"📁 GGUF file location: {os.path.abspath(GGUF_OUTPUT)}")
        print(f"\n📋 To use in OpenWebUI:")
        print(f"1. Copy {GGUF_OUTPUT} to your OpenWebUI models directory")
        print(f"2. Restart OpenWebUI")
        print(f"3. Select 'tinyllama-chat' from the model dropdown")
    else:
        print(f"\n⚠️  Training completed but GGUF conversion failed.")
        print(f"You can manually convert using:")
        print(f"python convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_OUTPUT} --outtype f16")

if __name__ == "__main__":
    model, tokenizer, tokenized_dataset = load_dataset("chat_output1.json")
    main(model, tokenizer, tokenized_dataset)