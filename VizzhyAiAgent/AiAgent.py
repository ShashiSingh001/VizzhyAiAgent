import os
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import faiss
from urllib.parse import quote, urljoin
from sentence_transformers import SentenceTransformer
import openpyxl  # Ensure Excel compatibility

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc]).strip() or None
    except Exception as e:
        print(f"⚠ Error processing PDF {pdf_path}: {e}")
        return None


def extract_text_from_csv(csv_path):
    """Extract text from a CSV file."""
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", encoding_errors="replace")
        return "\n".join(df.apply(lambda x: ' '.join(x.dropna()), axis=1)).strip() or None
    except Exception as e:
        print(f"⚠ Error processing CSV {csv_path}: {e}")
        return None


def extract_text_from_excel(excel_path):
    """Extract text from an Excel file, handling multiple sheets."""
    try:
        sheets = pd.read_excel(excel_path, sheet_name=None, dtype=str, engine="openpyxl")
        text = "\n".join("\n".join(df.apply(lambda x: ' '.join(x.dropna()), axis=1)) for df in sheets.values()).strip()
        return text or None
    except Exception as e:
        print(f"⚠ Error processing Excel {excel_path}: {e}")
        return None

def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_dataset_link(file_name, file_path, use_cloud=False):
    """Generate a dataset link for local or cloud storage."""
    if use_cloud:
        return urljoin("https://your-cloud-storage.com/datasets/", quote(file_name))
    return f"file:///{quote(os.path.abspath(file_path).replace('\\', '/'))}"

def process_files(data_dir):
    """Process all supported files in the specified directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")
    
    all_files = os.listdir(data_dir)
    print(f"📂 Found {len(all_files)} files in data directory.")
    structured_data = []
    
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"⚠ File not found: {file_path}")
            continue
        
        text, file_type = None, None
        
        if file.lower().endswith(".pdf"):
            text, file_type = extract_text_from_pdf(file_path), "PDF"
        elif file.lower().endswith(".csv"):
            text, file_type = extract_text_from_csv(file_path), "CSV"
        elif file.lower().endswith(".xlsx"):
            text, file_type = extract_text_from_excel(file_path), "Excel"
        else:
            print(f"⚠ Skipping unsupported file type: {file}")
            continue
        
        if not text:
            print(f"⚠ Skipping empty/unreadable file: {file}")
            continue
        
        chunks = split_text_into_chunks(text, chunk_size=500)
        dataset_link = generate_dataset_link(file, file_path, use_cloud=False)
        
        for i, chunk in enumerate(chunks):
            structured_data.append({
                "file_name": file,
                "file_type": file_type,
                "file_link": dataset_link,
                "chunk_index": i,
                "text_chunk": chunk
            })
    
    if not structured_data:
        raise ValueError("⚠ No valid text chunks found! Ensure files contain readable text.")
    
    print(f"✅ Processed {len(set(d['file_name'] for d in structured_data))} files and stored {len(structured_data)} text chunks.")
    
    with open("structured_data.json", "w", encoding="utf-8") as json_file:
        json.dump(structured_data, json_file, indent=4, ensure_ascii=False)
    
    return structured_data

def compute_embeddings(structured_data):
    """Compute text embeddings and store them in a FAISS index."""
    batch_size = 32
    all_embeddings = []
    num_chunks = len(structured_data)
    
    for i in range(0, num_chunks, batch_size):
        batch = [item["text_chunk"] for item in structured_data[i: i + batch_size]]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True).cpu().numpy()
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings).astype("float32")
    
    if embeddings.shape[0] > 0:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "vector_database.index")
        print(f"✅ Successfully stored {num_chunks} text chunks!")
    else:
        raise ValueError("⚠ No embeddings found! Ensure files contain readable text.")

if __name__ == "__main__":
    data_directory = r"C:\Users\My PC\Downloads\VizzhyAiAgent\VizzhyAiAgent\data"
    processed_data = process_files(data_directory)
    compute_embeddings(processed_data)


# In[ ]:


from flask import Flask, request, jsonify, render_template, send_file
import faiss
import json
import os
import numpy as np
import requests
import re
import threading
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
REQUIRED_FILES = ["vector_database.index", "structured_data.json"]
RELEVANCE_THRESHOLD = 1.5
TOP_K_RESULTS = 3
AI_API_KEY = "akm5535x-m60d44cc-39dbca60-fccb8d59"
AI_ENDPOINT_URL = "https://api.us.inc/hanooman/router/v1/chat/completions"
USER_QUERY_HISTORY = {}
QUERY_LOCK = threading.Lock()
DATA_DIR = "C:\\Users\\rishu\\OneDrive\\Desktop\\VizzhyAiAgent\\data"

# Verify required files exist
for file in REQUIRED_FILES:
    if not os.path.exists(file):
        logging.error(f"Required file missing: {file}")
        raise FileNotFoundError(f"Required file missing: {file}")

# Load FAISS index
try:
    index = faiss.read_index("vector_database.index")
    logging.info("FAISS index loaded successfully!")
except Exception as e:
    logging.error(f"Error loading FAISS index: {str(e)}")
    raise e

# Load structured data
try:
    with open("structured_data.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    logging.info(f"Loaded {len(structured_data)} structured text chunks!")
except Exception as e:
    logging.error(f"Error loading structured data: {str(e)}")
    raise e

# Load SentenceTransformer model
try:
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logging.info("Embedding model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading embedding model: {str(e)}")
    raise e


def search_faiss(query, top_k=TOP_K_RESULTS):
    """Search FAISS index for relevant text chunks."""
    try:
        query_vector = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = index.search(query_vector, top_k)
        
        valid_indices = [idx for idx in indices[0] if 0 <= idx < len(structured_data)]
        retrieved_chunks = [structured_data[idx] for idx in valid_indices]
        
        if not retrieved_chunks or distances[0][0] > RELEVANCE_THRESHOLD:
            return "Out of context", []
        
        return "Relevant", retrieved_chunks
    except Exception as e:
        logging.error(f"FAISS search error: {str(e)}")
        return "Error", []


def query_ai(query, retrieved_chunks, user_id, expected_items=20):
    """Query AI model with retrieved context."""
    context = "\n".join(chunk.get("text_chunk", "") for chunk in retrieved_chunks)
    user_history = "\n".join(USER_QUERY_HISTORY.get(user_id, []))
    
    payload = {
        "model": "everest",
        "messages": [
            {"role": "system", "content": f"You are a chatbot providing responses based ONLY on the given context. Ensure your response contains exactly {expected_items} complete items if requested."},
            {"role": "user", "content": f"User History:\n{user_history}\n\nQuery: {query}\n\nGIVEN CONTEXT:\n{context}"}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(AI_ENDPOINT_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: No valid response.")
        
        # Ensure the response contains expected number of items
        if "genes" in query.lower() and len(re.findall(r"\d+\. ", ai_response)) < expected_items:
            logging.warning("AI response is incomplete. Retrying with strict constraints.")
            return query_ai(query, retrieved_chunks, user_id, expected_items)
        
        return ai_response
    except requests.exceptions.RequestException as e:
        logging.error(f"AI service error: {str(e)}")
        return "Error: AI service is unavailable."


@app.route("/")
def home():
    return render_template("Vizzhy.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "default_user")
    
    if not user_message:
        return jsonify({"reply": "Please provide a question or message."})
    
    with QUERY_LOCK:
        USER_QUERY_HISTORY.setdefault(user_id, []).append(user_message)
        if len(USER_QUERY_HISTORY[user_id]) > 10:
            USER_QUERY_HISTORY[user_id] = USER_QUERY_HISTORY[user_id][-10:]
    
    status, retrieved_chunks = search_faiss(user_message)
    if status == "Out of context":
        return jsonify({"reply": "I cannot answer that based on the available data."})
    
    ai_response = query_ai(user_message, retrieved_chunks, user_id, expected_items=20)
    return jsonify({"reply": ai_response})


@app.route("/download/<path:filename>")
def download_file(filename):
    file_path = os.path.join(DATA_DIR, filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) else jsonify({"error": "File not found!"}), 404


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logging.error(f"Flask app failed to start: {str(e)}")

