# app/embedder.py
import os
from supabase import create_client
import openai
from parser import extract_text, chunk_text, embed_text

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_DIM = 3072
EMBED_MODEL = "text-embedding-3-large"

openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_chunk_to_supabase(content: str, file_name: str, chunk_idx: int):
    embedding = embed_text(content)
    metadata = {
        "source_file": file_name,
        "chunk_index": chunk_idx,
        "file_type": os.path.splitext(file_name)[1].lower(),
    }
    result = supabase.table("documents").insert({
        "content": content,
        "embedding": embedding,
        "metadata": metadata
    }).execute()
    if result.get("error"):
        raise RuntimeError(result["error"])

def process_file(filename: str, content: bytes):
    ext = os.path.splitext(filename)[1].lower()
    text = extract_text(content, ext)
    chunks = list(chunk_text(text))
    for i, chunk in enumerate(chunks):
        upload_chunk_to_supabase(chunk, filename, i)
    return len(chunks)