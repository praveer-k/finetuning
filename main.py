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

def download_pdfs():
    pdf_links = [
        "https://www.oecd.org/content/dam/oecd/en/publications/reports/2022/01/oecd-transfer-pricing-guidelines-for-multinational-enterprises-and-tax-administrations-2022_57104b3a/0e655865-en.pdf",
        "https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"
    ]
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    for idx, url in enumerate(pdf_links, start=1):
        try:
            response = requests.get(url)
            response.raise_for_status()
            filename = url.split("/")[-1] or f"document_{idx}.pdf"
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            filepath = os.path.join(PDF_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

# === STEP 1: Extract text from PDFs ===
def extract_text_from_pdfs(pdf_folder):
    texts = []
    for pdf_file in Path(pdf_folder).glob("*.pdf"):
        print(f"Extracting {pdf_file}")
        text = extract_text(str(pdf_file))
        if text.strip():
            # Format as chat template for better fine-tuning
            formatted_text = f"<|user|>\nWhat can you tell me about this document?\n<|assistant|>\n{text.strip()}\n<|endoftext|>"
            texts.append({"text": formatted_text})
    return texts

# === STEP 2: Prepare Dataset ===
def prepare_dataset(text_data):
    return Dataset.from_list(text_data)

# === STEP 3: Tokenization Function ===
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

# === STEP 4: Merge LoRA weights ===
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

# === STEP 5: Convert to GGUF ===
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
def main():
    download_pdfs()
    
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
    
    # Extract and tokenize data
    print("Extracting text from PDFs...")
    raw_data = extract_text_from_pdfs(PDF_DIR)
    
    if not raw_data:
        print("No PDF data found. Please check your PDF_DIR path.")
        return
    
    dataset = prepare_dataset(raw_data)
    print(f"Dataset created with {len(dataset)} examples")
    
    tokenized_dataset = dataset.map(
        lambda e: tokenize_function(e, tokenizer), 
        batched=True,
        remove_columns=dataset.column_names
    )
    
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
    main()