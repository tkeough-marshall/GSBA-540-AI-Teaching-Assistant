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
CHAT_MODEL = "gpt-4.1-mini"
VECTOR_DIM = 3072
TOP_K = 5

# ==============================
# 🚀 Flask App Setup
# ==============================
app = Flask(__name__)  # no static folder
CORS(app)

# Init clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

# ==============================
# 🌐 Routes
# ==============================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")  # serve from root folder

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # 🔍 Embed user query
        embed_resp = openai.embeddings.create(
            input=[user_input],
            model=EMBED_MODEL
        )
        query_vector = embed_resp.data[0].embedding

        # 📡 Query Supabase
        response = supabase.table("documents").rpc("match_documents", {
            "query_embedding": query_vector,
            "match_count": TOP_K
        }).execute()

        if response.get("error"):
            return jsonify({"error": response["error"]["message"]}), 500

        matches = response["data"]
        context = "\n\n".join([doc["content"] for doc in matches])

        # 💬 GPT Prompt
        system_prompt = "You are a helpful assistant that answers based on the context provided."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
        ]

        chat_resp = openai.ChatCompletion.create(
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
