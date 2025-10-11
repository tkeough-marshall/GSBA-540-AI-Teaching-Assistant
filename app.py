from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
import os
import openai
import re

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
CHAT_MODEL = "gpt-4o-mini"  # fast + strong reasoning
VECTOR_DIM = 3072
TOP_K = 15  # number of chunks to retrieve

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
    """Read the system message from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return (
            "You are a helpful assistant that answers questions based "
            "strictly using the provided context. "
            "If the context doesn't contain enough information, say so clearly."
        )

SYSTEM_PROMPT = load_system_prompt()


# ==============================
# 🌐 Routes
# ==============================

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static assets like images or CSS."""
    return send_from_directory("static", filename)


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint — handles user query and returns grounded answer."""
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # =======================================
        # 🧠 Step 1: Embed the user's query
        # =======================================
        embed_resp = openai.embeddings.create(
            input=[user_input],
            model=EMBED_MODEL
        )
        query_vector = embed_resp.data[0].embedding

        # =======================================
        # 📡 Step 2: Query Supabase (match_documents RPC)
        # =======================================
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_count": TOP_K
        }).execute()

        if getattr(response, "error", None):
            return jsonify({"error": str(response.error)}), 500

        matches = getattr(response, "data", [])
        if not matches:
            return jsonify({
                "response": "No relevant course materials found.",
                "sources": "_No course documents were relevant to this response._"
            })

        # =======================================
        # 🧩 Step 3: Build context with metadata
        # =======================================
        context_blocks = []
        valid_sources = set()

        for doc in matches:
            content = doc.get("content", "").strip()
            source = doc.get("source_file", "Unknown")
            file_type = doc.get("file_type", "Unknown")
            chunk_idx = doc.get("chunk_index", "N/A")

            if content:
                block = f"[Source: {source} | Type: {file_type} | Chunk: {chunk_idx}]\n{content}"
                context_blocks.append(block)
                valid_sources.add(source)

        context = "\n\n".join(context_blocks)

        # =======================================
        # 💬 Step 4: Construct messages
        # =======================================
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Answer the question strictly and exclusively using the provided context.\n"
                    "If the context lacks relevant information, clearly say so.\n\n"
                    "---\nCONTEXT START\n"
                    f"{context}\n"
                    "CONTEXT END\n---\n\n"
                    f"QUESTION: {user_input}\n\n"
                    "Instructions:\n"
                    "- Only reference information explicitly present in the context.\n"
                    "- List source filenames exactly as seen above (deduplicated) at the end of your response.\n"
                    "- Never invent or infer filenames or data.\n"
                    "- If no relevant content exists, respond:\n"
                    "  'The provided materials do not include information on this topic.'\n"
                    "  **Sources:** _No course documents were relevant to this response._"
                )
            }
        ]

        # =======================================
        # 🤖 Step 5: Generate model response
        # =======================================
        chat_resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )

        answer = chat_resp.choices[0].message.content.strip()

        # =======================================
        # 🔒 Step 6: Validate source citations
        # =======================================
        mentioned = set(re.findall(r"`([^`]+)`", answer))
        invalid = mentioned - valid_sources
        if invalid:
            for bad in invalid:
                answer = answer.replace(f"`{bad}`", "")
            answer += "\n\n_Note: Invalid or hallucinated source names were removed._"

        # =======================================
        # ✅ Step 7: Return clean response
        # =======================================
        return jsonify({
            "response": answer,
            "sources": ", ".join(sorted(valid_sources))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔌 Render Port Binding
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
