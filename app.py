import os, sys, tempfile, fitz, openai, traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from docx import Document as DocxDocument
from pptx import Presentation

# Always flush logs for Render
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(msg)
    sys.stdout.flush()

# ==============================
# 🔧 Environment + Config Inputs
# ==============================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_DIM = 3072
TOP_K = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY


# ==============================
# 📘 System Prompt
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
# 🧠 Embedding + Ingestion
# ==============================
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            return "\n".join(p.get_text("text") for p in doc)
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text)
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
    except Exception as e:
        log(f"❌ extract_text() failed for {file_path}: {e}")
        traceback.print_exc()
        return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        yield " ".join(words)
        return
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+chunk_size])
        if i + chunk_size >= len(words):
            break


def embed_text(text):
    resp = openai.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def process_and_store(file_path, file_name):
    log(f"⚙️ process_and_store triggered for {file_name}")
    try:
        text = extract_text(file_path)
        log(f"📄 Extracted {len(text)} characters")
        if not text.strip():
            log(f"⚠️ No extractable text in {file_name}")
            return {"status": "no_text"}

        chunks = list(chunk_text(text))
        log(f"🧩 Split into {len(chunks)} chunks")

        total = 0
        for i, chunk in enumerate(chunks):
            try:
                log(f"Embedding chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                emb = embed_text(chunk)
                meta = {
                    "source_file": file_name,
                    "chunk_index": i,
                    "file_type": os.path.splitext(file_name)[1],
                }
                supabase.table("documents").insert({
                    "content": chunk,
                    "embedding": emb,
                    "metadata": meta,
                }).execute()
                total += 1
            except Exception as e:
                log(f"❌ Error embedding chunk {i}: {e}")
                traceback.print_exc()
        log(f"✅ Finished embedding {total} chunks for {file_name}")
        return {"status": "success", "chunks": total}
    except Exception as e:
        log(f"❌ Fatal error in process_and_store: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


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
# 📁 File Management
# ==============================
@app.route("/api/files")
def list_files():
    data = supabase.table("files").select("*").order("uploaded_at", desc=True).execute()
    return jsonify(data.data)


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    path = f"materials/{filename}"
    tmp = tempfile.NamedTemporaryFile(delete=False)
    file.save(tmp.name)
    log(f"✅ Upload received: {filename}")

    try:
        supabase.storage.from_("materials").upload(filename, open(tmp.name, "rb"))
        supabase.table("files").insert({
            "file_name": filename,
            "file_type": os.path.splitext(filename)[1],
            "source_path": path
        }).execute()
        log(f"📦 Logged file: {filename}")
    except Exception as e:
        log(f"❌ Upload/log error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    result = process_and_store(tmp.name, filename)
    log(f"🧠 Embedding result: {result}")
    return jsonify({"message": f"{filename} uploaded and embedding attempted", "embedding_result": result})


@app.route("/delete/<file_name>", methods=["DELETE"])
def delete_file(file_name):
    try:
        supabase.storage.from_("materials").remove([file_name])
        supabase.table("files").delete().eq("file_name", file_name).execute()
        supabase.table("documents").delete().filter("metadata->>source_file", "eq", file_name).execute()
        log(f"🗑️ Deleted {file_name}")
        return jsonify({"message": f"{file_name} deleted successfully"})
    except Exception as e:
        log(f"❌ Delete error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==============================
# 💬 Chat API
# ==============================
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    try:
        embed_resp = openai.embeddings.create(model=EMBED_MODEL, input=[user_input])
        query_vector = embed_resp.data[0].embedding

        response = supabase.rpc("match_documents", {"query_embedding": query_vector, "match_count": TOP_K}).execute()
        if getattr(response, "error", None):
            return jsonify({"error": str(response.error)}), 500

        matches = getattr(response, "data", [])
        if not matches:
            return jsonify({
                "response": "The materials do not include this topic.\n**Sources:** None"
            })

        context_blocks, sources = [], set()
        for m in matches:
            content = m.get("content", "").strip()
            src = m.get("source_file", "Unknown")
            if content:
                context_blocks.append(f"[source_file: {src}]\n{content}")
                sources.add(src)
        context = "\n\n".join(context_blocks)
        formatted_sources = "**Sources:** " + ", ".join(f"`{s}`" for s in sorted(sources))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]
        chat_resp = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
        answer = chat_resp.choices[0].message.content.strip()
        return jsonify({"response": f"{answer}\n\n{formatted_sources}"})
    except Exception as e:
        log(f"❌ Chat error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔌 Run Server
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)