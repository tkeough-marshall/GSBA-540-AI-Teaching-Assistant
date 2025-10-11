from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
import os
import openai

# ==============================
# 🔧 Environment Setup
# ==============================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================
# ⚙️ Model & Vector Config
# ==============================
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"  # compact + fast
VECTOR_DIM = 3072
TOP_K = 15  # number of chunks to retrieve from Supabase

# ==============================
# 🚀 Flask App Setup
# ==============================
app = Flask(__name__, static_folder="static")
CORS(app)

# Init clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

# ==============================
# 📘 Load System Prompt
# ==============================
def load_system_prompt(filepath="system_message.txt"):
    """Reads the system message from an external file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return (
            "You are a helpful assistant that answers questions based "
            "strictly using the provided context. "
            "If the context doesn't contain enough information, say so clearly."
        )

system_prompt = load_system_prompt()

# ==============================
# 🌐 Routes
# ==============================

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static assets like images, CSS, etc."""
    return send_from_directory("static", filename)

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint."""
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # =======================================
        # 🧠 Step 1: Embed the user’s query
        # =======================================
        embed_resp = openai.embeddings.create(
            input=[user_input],
            model=EMBED_MODEL
        )
        query_vector = embed_resp.data[0].embedding

        # =======================================
        # 📡 Step 2: Query Supabase via RPC
        # =======================================
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_count": TOP_K
        }).execute()

        if getattr(response, "error", None):
            return jsonify({"error": str(response.error)}), 500

        # =======================================
        # 🧩 Step 3: Build context from matches
        # =======================================
        matches = getattr(response, "data", [])
        if not matches:
            return jsonify({"response": "No relevant documents found in Supabase."})

        context = "\n\n".join([doc["content"] for doc in matches if doc.get("content")])

        # =======================================
        # 💬 Step 4: Generate answer with GPT
        # =======================================
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]

        chat_resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = chat_resp.choices[0].message.content.strip()
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔌 Render Port Binding
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
