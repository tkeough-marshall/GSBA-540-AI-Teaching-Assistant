from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tempfile
import os
import openai
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pptx import Presentation

# ==============================
# 🔧 Environment + Config Inputs
# ==============================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model + Retrieval Settings
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_DIM = 3072
TOP_K = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Flask App Setup
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Init Clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY


# ==============================
# 📘 Load System Prompt
# ==============================
def load_system_prompt(filepath="system_message.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return (
            "You are a helpful assistant that answers questions strictly using the provided context. "
            "If the context doesn't include enough information, say so clearly."
        )


SYSTEM_PROMPT = load_system_prompt()


# ==============================
# 🧠 Embedding + Document Ingestion Helpers
# ==============================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join([p.get_text("text") for p in doc if p.get_text("text")])
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    elif ext == ".pptx":
        prs = Presentation(file_path)
        texts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
        return "\n".join(texts)
    else:
        return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        yield " ".join(words)
        return
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        yield " ".join(chunk)
        if i + chunk_size >= len(words):
            break


def embed_text(text):
    resp = openai.embeddings.create(input=[text], model=EMBED_MODEL)
    return resp.data[0].embedding


def process_and_store(file_path, file_name):
    text = extract_text(file_path)
    if not text.strip():
        print(f"⚠️ No extractable text in {file_name}")
        return {"status": "no_text"}

    chunks = list(chunk_text(text))
    total = 0
    for i, chunk in enumerate(chunks):
        try:
            emb = embed_text(chunk)
            meta = {"source_file": file_name, "chunk_index": i, "file_type": os.path.splitext(file_name)[1]}
            supabase.table("documents").insert({"content": chunk, "embedding": emb, "metadata": meta}).execute()
            total += 1
        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")

    return {"status": "success", "chunks": total}


# ==============================
# 🌐 Routes
# ==============================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")


@app.route("/manage")
def manage_page():
    return send_from_directory("templates", "manage.html")


# ==============================
# 📁 File Management API
# ==============================
@app.route("/api/files")
def list_files():
    data = supabase.table("files").select("*").order("uploaded_at", desc=True).execute()
    return jsonify(data.data)


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload file → Supabase storage + table + embeddings"""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    path = f"materials/{filename}"

    tmp = tempfile.NamedTemporaryFile(delete=False)
    file.save(tmp.name)

    # Upload to Supabase Storage bucket
    supabase.storage.from_("materials").upload(filename, open(tmp.name, "rb"))

    # Log record in 'files' table
    supabase.table("files").insert({
        "file_name": filename,
        "file_type": os.path.splitext(filename)[1],
        "source_path": path
    }).execute()

    # Embed + store in documents
    result = process_and_store(tmp.name, filename)

    return jsonify({
        "message": f"{filename} uploaded and embedded successfully",
        "embedding_result": result
    })


@app.route("/delete/<file_name>", methods=["DELETE"])
def delete_file(file_name):
    """Delete file from Supabase Storage, 'files', and 'documents'"""
    supabase.storage.from_("materials").remove([file_name])
    supabase.table("files").delete().eq("file_name", file_name).execute()
    supabase.table("documents").delete().filter("metadata->>source_file", "eq", file_name).execute()
    return jsonify({"message": f"{file_name} deleted successfully"})


# ==============================
# 💬 Chat API
# ==============================
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Step 1: Embed Query
        embed_resp = openai.embeddings.create(model=EMBED_MODEL, input=[user_input])
        query_vector = embed_resp.data[0].embedding

        # Step 2: Retrieve Matches
        response = supabase.rpc("match_documents", {"query_embedding": query_vector, "match_count": TOP_K}).execute()

        if getattr(response, "error", None):
            return jsonify({"error": str(response.error)}), 500

        matches = getattr(response, "data", [])
        if not matches:
            return jsonify({
                "response": (
                    "The provided materials do not include information on this topic.\n"
                    "**Sources:** _No relevant documents found._"
                )
            })

        # Step 3: Build Context
        context_blocks, sources = [], set()
        for m in matches:
            content = m.get("content", "").strip()
            src = m.get("source_file", "Unknown")
            if content:
                context_blocks.append(f"[source_file: {src}]\n{content}")
                sources.add(src)
        context = "\n\n".join(context_blocks)
        formatted_sources = "**Sources:** " + ", ".join(f"`{s}`" for s in sorted(sources))

        # Step 4: Generate Response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]
        chat_resp = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
        answer = chat_resp.choices[0].message.content.strip()

        return jsonify({"response": f"{answer}\n\n{formatted_sources}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔌 Run Server
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)