import os
import json
from pathlib import Path
import re
import requests
import pdfplumber
from openai import OpenAI
from tqdm import tqdm
from hashlib import sha256

# === CONFIG ===
PDF_DIR = "pdf_docs"

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

def extract_overlapping_chunks(pdf_path, overlap=1):
    """Extracts 2-page overlapping text chunks."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for i in tqdm(range(0, num_pages - 1)):
            text1 = pdf.pages[i].extract_text() or ""
            text2 = pdf.pages[i + 1].extract_text() or ""
            combined = f"{text1.strip()}\n\n{text2.strip()}"
            chunks.append(combined)
    return chunks

def ask_openai_for_questions(text):
    prompt = (
        "Based on the following document content, generate 3-5 concise, relevant questions that a user might ask to understand it better in json format:\n\n"
        f"{text[:3000]}"
    )
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    pattern = r'```(.*?)```'
    matches = re.findall(pattern, response.choices[0].message.content, re.DOTALL)
    print(matches)
    questions = json.loads(matches[0])
    questions = [q["question"] for q in questions]
    return questions

def ask_openai_to_answer(text, question):
    messages = [
        {"role": "system", "content": "Answer only based on the provided document content."},
        {"role": "user", "content": f"Document:\n{text[:5000]}\n\nQuestion: {question}"}
    ]
    response = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_chat_data(pdf_path):
    chunks = extract_overlapping_chunks(pdf_path)
    seen_questions = set()
    qa_pairs = []

    for chunk_index, text in enumerate(chunks):
        print(f"Processing chunk {chunk_index + 1} of {len(chunks)}...")

        try:
            questions = ask_openai_for_questions(text)
            print(questions)
        except Exception as e:
            print(f"Error generating questions: {e}")
            continue

        for question in questions:
            q_hash = sha256(question.lower().encode()).hexdigest()
            if q_hash in seen_questions:
                continue

            try:
                answer = ask_openai_to_answer(text, question)
            except Exception as e:
                print(f"Error answering question '{question}': {e}")
                continue

            seen_questions.add(q_hash)
            qa_pairs.append({"role": "user", "content": question})
            qa_pairs.append({"role": "assistant", "content": answer})

    return qa_pairs

def download_pdfs(pdf_links):
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    for idx, url in enumerate(pdf_links, start=1):
        try:
            filename = url.split("/")[-1] or f"document_{idx}.pdf"
            pdf_path = os.path.join("./pdf_docs", filename)
            if Path(pdf_path).exists():
                print(f"File already exists {filename}")
                continue
            response = requests.get(url)
            response.raise_for_status()
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            filepath = os.path.join(PDF_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

# Example usage
if __name__ == "__main__":
    pdf_links = [
        "https://www.oecd.org/content/dam/oecd/en/publications/reports/2022/01/oecd-transfer-pricing-guidelines-for-multinational-enterprises-and-tax-administrations-2022_57104b3a/0e655865-en.pdf",
        "https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"
    ]
    download_pdfs(pdf_links)
    for idx, url in enumerate(pdf_links, start=1):
        filename = url.split("/")[-1] or f"document_{idx}.pdf"
        pdf_path = os.path.join("./pdf_docs", filename)
        chat_data = generate_chat_data(pdf_path)
        with open(f"chat_output{idx}.json", "w") as f:
            json.dump(chat_data, f, indent=2)

    print("Done. Output saved to chat_output.json")