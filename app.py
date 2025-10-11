from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
import os
import openai

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
TOP_K = 10  # Number of chunks to retrieve

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
    """Reads the system message from an external file."""
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
# 🌐 Routes
# ==============================

@app.route("/")
def index():
    """Serve main HTML page."""
    return send_from_directory(".", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static assets such as images or CSS."""
    return send_from_directory("static", filename)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint: embeds query, retrieves chunks, generates grounded answer."""
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # =======================================
        # 🧠 Step 1: Embed User Query
        # =======================================
        embed_resp = openai.embeddings.create(
            model=EMBED_MODEL,
            input=[user_input]
        )
        query_vector = embed_resp.data[0].embedding

        # =======================================
        # 📡 Step 2: Retrieve Matching Documents
        # =======================================
        response = supabase.rpc(
            "match_documents",
            {"query_embedding": query_vector, "match_count": TOP_K}
        ).execute()

        if getattr(response, "error", None):
            return jsonify({"error": str(response.error)}), 500

        matches = getattr(response, "data", [])
        if not matches:
            return jsonify({
                "response": (
                    "The provided materials do not include information on this topic.\n"
                    "**Sources:** _No course documents were relevant to this response._"
                )
            })

        # =======================================
        # 🧩 Step 3: Build Context Using Metadata Fields
        # =======================================
        context_blocks = []
        sources = set()  # Track unique filenames

        print("\n--- Retrieved Matches ---")
        for m in matches:
            content = m.get("content", "").strip()
            source = m.get("source_file", "Unknown")
            chunk_index = m.get("chunk_index", "N/A")
            file_type = m.get("file_type", "N/A")
            similarity = m.get("similarity", 0.0)

            print(f"{source} | {file_type} | similarity={similarity:.3f}")

            if content:
                block = (
                    f"[source_file: {source} | file_type: {file_type} | chunk_index: {chunk_index}]\n"
                    f"{content}"
                )
                context_blocks.append(block)
                sources.add(source)

        print("--- End Matches ---\n")

        # Join all chunks into a single context
        context = "\n\n".join(context_blocks)

        # Format Sources for final response
        source_list = sorted(sources)
        formatted_sources = (
            f"**Sources:** " + ", ".join(f"`{s}`" for s in source_list)
            if source_list else "**Sources:** _No course documents were relevant to this response._"
        )

        # =======================================
        # 💬 Step 4: Construct Chat Messages
        # =======================================
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Answer the question using only the context below.\n\n"
                    f"---\n{context}\n---\n\n"
                    f"Question: {user_input}"
                ),
            },
        ]

        # =======================================
        # 🤖 Step 5: Generate Model Response
        # =======================================
        chat_resp = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2
        )

        answer = chat_resp.choices[0].message.content.strip()

        # =======================================
        # ✅ Step 6: Return Final Output
        # =======================================
        return jsonify({
            "response": f"{answer}\n\n{formatted_sources}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔌 Run Server
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
