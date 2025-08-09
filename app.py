import streamlit as st
from openai import OpenAI
from typing import List, Dict, Tuple
import math
import json

# =========================
# Configuration
# =========================
DEFAULT_CHAT_MODEL = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K_DEFAULT = 5
ACCEPTED_FILE_TYPES = ["txt", "md", "csv", "json"]

# =========================
# OpenAI Client
# =========================
client = OpenAI()

# =========================
# Utilities
# =========================
def decode_bytes(content_bytes: bytes) -> str:
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            return content_bytes.decode(enc)
        except Exception:
            continue
    return content_bytes.decode("utf-8", errors="ignore")


def read_file_to_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    content = decode_bytes(uploaded_file.getvalue())

    if filename.endswith(".json"):
        try:
            parsed = json.loads(content)
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            # Keep as raw text if not valid JSON
            pass
    # For txt/md/csv: use raw text
    return content


def clean_text(text: str) -> str:
    # Normalize whitespace
    return " ".join(text.replace("\r", "").split())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    if chunk_size <= 0:
        chunk_size = 1000
    if overlap < 0:
        overlap = 0

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    # Embedding API expects a list input
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [d.embedding for d in resp.data]
    return vectors


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def build_index(chunks: List[str], embeddings: List[List[float]]) -> List[Dict]:
    index = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        index.append({"id": i, "text": chunk, "embedding": vec})
    return index


def retrieve_top_k(query: str, index: List[Dict], k: int = TOP_K_DEFAULT) -> List[Tuple[int, str, float]]:
    if not query.strip() or not index:
        return []
    q_vec = embed_texts([query])[0]
    scores = []
    for item in index:
        sim = cosine_similarity(q_vec, item["embedding"])
        scores.append((item["id"], item["text"], sim))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[: max(1, k)]


def format_context(chunks: List[Tuple[int, str, float]]) -> str:
    # Provide clearly separated contexts
    parts = []
    for idx, text, score in chunks:
        parts.append(f"[Chunk {idx} | score={score:.3f}]\n{text}")
    return "\n\n---\n\n".join(parts)


def answer_query_with_rag(query: str, context: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer cannot be found in the context, say you don't know. Be concise and accurate."
    )
    user_content = (
        f"Context:\n{context}\n\n"
        f"User Question:\n{query}\n\n"
        "Instructions: Use the context above to answer. If not present, reply that the context does not contain the answer."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def reset_state():
    keys = ["file_name", "raw_text", "chunks", "index", "model", "top_k"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


# =========================
# Streamlit App
# =========================
def init_session_state():
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_CHAT_MODEL
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_DEFAULT
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index" not in st.session_state:
        st.session_state.index = []


def sidebar():
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox(
        "Chat Model",
        options=["gpt-4", "gpt-3.5-turbo"],
        index=0 if st.session_state.model == "gpt-4" else 1,
        help="Select the OpenAI chat model for answering queries.",
    )
    st.session_state.model = model

    top_k = st.sidebar.slider("Top K Chunks", min_value=1, max_value=10, value=st.session_state.top_k, help="Number of top similar chunks to retrieve.")
    st.session_state.top_k = top_k

    if st.sidebar.button("Clear Session", type="secondary"):
        reset_state()
        st.sidebar.success("Session cleared. Upload a new file to start.")


def main():
    st.set_page_config(page_title="RAG over Uploaded File", page_icon="📄", layout="wide")
    st.title("RAG: Ask Questions About Your File 📄🔎")
    st.write("Upload a text-based file (.txt, .md, .csv, .json) and ask questions. The assistant will answer using only information from your file.")

    init_session_state()
    sidebar()

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=ACCEPTED_FILE_TYPES,
        help="Supported: txt, md, csv, json",
    )

    # Process file
    if uploaded_file is not None:
        filename = uploaded_file.name
        is_new_file = filename != st.session_state.file_name

        if is_new_file:
            with st.spinner("Reading and chunking file..."):
                text = read_file_to_text(uploaded_file)
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

            with st.spinner("Creating embeddings and building index..."):
                vectors = embed_texts(chunks)
                index = build_index(chunks, vectors)

            st.session_state.file_name = filename
            st.session_state.raw_text = text
            st.session_state.chunks = chunks
            st.session_state.index = index

            st.success(f"Loaded '{filename}' with {len(chunks)} chunks.")

        # Show file stats
        col1, col2, col3 = st.columns(3)
        col1.metric("File", st.session_state.file_name or "-")
        col2.metric("Characters", f"{len(st.session_state.raw_text):,}")
        col3.metric("Chunks", f"{len(st.session_state.chunks):,}")

        st.divider()

        # Query interface
        query = st.text_input("Ask a question about the file")
        ask = st.button("Ask", type="primary")

        show_sources = st.checkbox("Show retrieved sources", value=True)

        if ask:
            if not st.session_state.index:
                st.error("No index available. Please re-upload the file.")
            elif not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving relevant chunks..."):
                    retrieved = retrieve_top_k(query, st.session_state.index, st.session_state.top_k)
                    context = format_context(retrieved)

                with st.spinner("Generating answer..."):
                    try:
                        answer = answer_query_with_rag(query, context, st.session_state.model)
                        st.subheader("Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        return

                if show_sources:
                    st.subheader("Retrieved Context")
                    for idx, (cid, ctext, score) in enumerate(retrieved, start=1):
                        with st.expander(f"Source {idx}: Chunk {cid} (score={score:.3f})"):
                            st.write(ctext)
    else:
        st.info("Please upload a file to begin.")


if __name__ == "__main__":
    main()