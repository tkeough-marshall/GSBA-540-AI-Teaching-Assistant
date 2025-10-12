import os, sys, tempfile, traceback, re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from zipfile import ZipFile

# Parsers
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pptx import Presentation
from openai import OpenAI

# -------- logging (flush for Render) --------
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def log(msg: str):
    print(msg, flush=True)

# -------- env --------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4o-mini"
VECTOR_DIM  = 3072
TOP_K       = 10
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 100

# -------- app/clients --------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
oai = OpenAI(api_key=OPENAI_API_KEY)

# -------- system prompt --------
def load_system_prompt(filepath="system_message.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ("You are a helpful assistant that answers questions strictly using the provided context. "
                "If the context doesn't include enough information, say so clearly.")
SYSTEM_PROMPT = load_system_prompt()

# ==============================
# Text extraction helpers
# ==============================
def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        txt = page.get_text("text")
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def extract_docx_text(path: str) -> str:
    out = []
    try:
        doc = DocxDocument(path)
        # paragraphs
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                out.append(t)
        # tables
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    out.append(" | ".join(cells))
        if out:
            return "\n".join(out)
    except Exception as e:
        log(f"❌ python-docx failed: {e}")
        traceback.print_exc()

    # ---- fallback XML parse ----
    try:
        with ZipFile(path, "r") as z:
            files = z.namelist()
            if "word/document.xml" in files:
                xml_data = z.read("word/document.xml").decode("utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", "", xml_data)
                log(f"⚙️ fallback xml extracted {len(text)} chars")
                return text
            else:
                log("⚠️ no document.xml found in docx")
    except Exception as e:
        log(f"❌ fallback unzip failed: {e}")
        traceback.print_exc()
    return ""

def extract_pptx_text(path: str) -> str:
    prs = Presentation(path)
    out = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        out.append(line)
            if hasattr(shape, "has_table") and shape.has_table:
                tbl = shape.table
                for r in tbl.rows:
                    cells = [c.text.strip() for c in r.cells if c.text.strip()]
                    if cells:
                        out.append(" | ".join(cells))
            if hasattr(shape, "text"):
                t = getattr(shape, "text", "").strip()
                if t:
                    out.append(t)
    return "\n".join(out)

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return extract_pdf_text(file_path)
        if ext == ".docx":
            return extract_docx_text(file_path)
        if ext == ".pptx":
            return extract_pptx_text(file_path)
        return ""
    except Exception as e:
        log(f"❌ extract_text failed for {file_path}: {e}")
        traceback.print_exc()
        return ""

# ==============================
# Chunking + Embedding
# ==============================
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return
    if len(words) <= chunk_size:
        yield " ".join(words)
        return
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]
        yield " ".join(chunk)
        if i + chunk_size >= len(words):
            break

def embed_text(text: str):
    resp = oai.embeddings.create(model=EMBED_MODEL, input=[text])
    emb = resp.data[0].embedding
    if len(emb) != VECTOR_DIM:
        log(f"⚠️ embedding dims {len(emb)} != expected {VECTOR_DIM}")
    return emb

def process_and_store(file_path: str, file_name: str):
    log(f"⚙️ process_and_store for {file_name}")
    text = extract_text(file_path)
    log(f"📄 Extracted {len(text)} characters")
    if not text.strip():
        return {"status": "no_text"}

    chunks = list(chunk_text(text))
    log(f"🧩 {len(chunks)} chunks")
    total = 0
    for i, chunk in enumerate(chunks):
        try:
            emb = embed_text(chunk)
            meta = {
                "source_file": file_name,
                "chunk_index": i,
                "file_type": os.path.splitext(file_name)[1].lower(),
            }
            supabase.table("documents").insert({
                "content": chunk,
                "embedding": emb,
                "metadata": meta
            }).execute()
            total += 1
        except Exception as e:
            log(f"❌ insert chunk {i}: {e}")
            traceback.print_exc()
    log(f"✅ embedded {total} chunks for {file_name}")
    return {"status": "success", "chunks": total}

# ==============================
# Routes
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

# ---------- File management ----------
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
    storage_path = filename
    tmp = tempfile.NamedTemporaryFile(delete=False)
    file.save(tmp.name)
    log(f"✅ Upload received: {filename}")

    # file integrity check
    size = os.path.getsize(tmp.name)
    with open(tmp.name, "rb") as f:
        sig = f.read(2)
    log(f"📏 File size: {size} bytes | 🔍 Signature: {sig}")

    try:
        with open(tmp.name, "rb") as fh:
            supabase.storage.from_("materials").upload(storage_path, fh)
        supabase.table("files").insert({
            "file_name": filename,
            "file_type": os.path.splitext(filename)[1].lower(),
            "source_path": storage_path
        }).execute()
        log(f"📦 Logged file: {filename}")
    except Exception as e:
        log(f"❌ upload/log error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    result = process_and_store(tmp.name, filename)
    log(f"🧠 Embedding result: {result}")
    return jsonify({"message": f"{filename} uploaded", "embedding_result": result})

@app.route("/delete/<file_name>", methods=["DELETE"])
def delete_file(file_name):
    try:
        supabase.storage.from_("materials").remove([file_name])
        supabase.table("files").delete().eq("file_name", file_name).execute()
        supabase.table("documents").delete().filter("metadata->>source_file", "eq", file_name).execute()
        log(f"🗑️ Deleted {file_name}")
        return jsonify({"message": f"{file_name} deleted"})
    except Exception as e:
        log(f"❌ delete error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------- Chat ----------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    try:
        emb = oai.embeddings.create(model=EMBED_MODEL, input=[user_input]).data[0].embedding
        resp = supabase.rpc("match_documents", {"query_embedding": emb, "match_count": TOP_K}).execute()
        if getattr(resp, "error", None):
            return jsonify({"error": str(resp.error)}), 500

        matches = getattr(resp, "data", []) or []
        if not matches:
            return jsonify({"response": "No relevant materials.\n**Sources:** None"})

        blocks, sources = [], set()
        for m in matches:
            content = m.get("content", "").strip()
            src = m.get("source_file", "Unknown")
            if content:
                blocks.append(f"[source_file: {src}]\n{content}")
                sources.add(src)
        context = "\n\n".join(blocks)
        srcs = "**Sources:** " + ", ".join(f"`{s}`" for s in sorted(sources))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Answer using only this context:\n\n{context}\n\nQuestion: {user_input}"}
        ]
        ans = oai.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)\
              .choices[0].message.content.strip()
        return jsonify({"response": f"{ans}\n\n{srcs}"})
    except Exception as e:
        log(f"❌ chat error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------- run --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
