
import streamlit as st

from ingestion.document_loader import extract_text
from ingestion.chunker import split_text
from embeddings.embedding_model import EmbeddingModel
from database.chroma_store import ChromaStore


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📊",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
    color: #1e293b;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.main-title {
    color: #16325c !important;
    font-size: 32px;
    font-weight: 750;
}

.subtitle {
    color: #64748b !important;
    font-size: 15px;
    margin-bottom: 20px;
}

h1, h2, h3, h4, p, label,
.stMarkdown, .stCaption {
    color: #1e293b;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 12px;
    padding: 18px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #16325c !important;
}

div[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 12px;
    padding: 14px;
}

div[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: none !important;
}

div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section * {
    color: #1e293b !important;
}

div[data-testid="stFileUploader"] button {
    background: #0f766e !important;
    color: #ffffff !important;
    border: 1px solid #0f766e !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #115e59 !important;
}

.stButton > button {
    background: #0f766e !important;
    color: #ffffff !important;
    border: 1px solid #0f766e !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
}

.stButton > button * {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: #115e59 !important;
    border-color: #115e59 !important;
}

div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dbe4f0 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: #ffffff !important;
}

div[data-testid="stExpander"] summary * {
    color: #16325c !important;
}

div[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #dbe4f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD COMPONENTS ----------------
@st.cache_resource
def get_components():
    embedder = EmbeddingModel()
    store = ChromaStore(persist_directory="chroma_db")
    return embedder, store


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">'
    'AI Project Intelligence & Risk Advisor'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Document ingestion and knowledge base construction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- OVERVIEW ----------------
st.subheader("Document Knowledge Base")

st.write(
    "Upload PDF, DOCX, CSV, and TXT files. "
    "The application extracts text, creates chunks, generates embeddings, "
    "and stores them in a persistent ChromaDB database."
)


# ---------------- DATABASE STATUS ----------------
try:
    store = ChromaStore(persist_directory="chroma_db")
    existing_count = store.count_chunks()
    database_status = "Connected"
except Exception:
    existing_count = 0
    database_status = "Unavailable"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Supported Formats", "4")

with col2:
    st.metric("Stored Chunks", existing_count)

with col3:
    st.metric("Database Status", database_status)

st.divider()


# ---------------- FILE UPLOAD ----------------
st.subheader("Upload Project Documents")
st.write("Supported formats: PDF, DOCX, CSV, TXT")

uploaded_files = st.file_uploader(
    "Choose your project documents",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple project documents."
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) selected")

    for file in uploaded_files:
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(f"📄 {file.name}")

        with col2:
            st.caption(f"{file.size / 1024:.1f} KB")

st.divider()


# ---------------- PROCESSING PIPELINE ----------------
if st.button("Start Document Processing", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        try:
            with st.spinner(
                "Loading embedding model and connecting to ChromaDB..."
            ):
                embedder, store = get_components()

            success_count = 0
            failed_count = 0
            total_chunks = 0

            st.subheader("Document Processing Results")

            for uploaded_file in uploaded_files:

                with st.expander(
                    f"📄 {uploaded_file.name}",
                    expanded=True
                ):
                    try:
                        file_bytes = uploaded_file.getvalue()

                        # 1. Extract text
                        extracted_text = extract_text(
                            uploaded_file.name,
                            file_bytes
                        )

                        st.write(
                            f"**Characters extracted:** "
                            f"{len(extracted_text):,}"
                        )

                        # 2. Split text into chunks
                        chunks = split_text(
                        extracted_text,
                        chunk_size=350,
                        chunk_overlap=50
                        )

                        if not chunks:
                            raise ValueError(
                                "No text chunks were created."
                            )

                        st.write(f"**Chunks created:** {len(chunks)}")

                        # 3. Generate embeddings
                        with st.spinner(
                            f"Generating embeddings for {uploaded_file.name}..."
                        ):
                            vectors = embedder.embed_texts(chunks)

                        # 4. Store in ChromaDB
                        stored = store.add_chunks(
                            filename=uploaded_file.name,
                            file_bytes=file_bytes,
                            chunks=chunks,
                            embeddings=vectors
                        )

                        success_count += 1
                        total_chunks += stored

                        st.success(
                            f"Indexed successfully: {stored} chunks"
                        )

                        st.markdown("**Extracted text preview**")

                        st.text_area(
                            "Text preview",
                            value=extracted_text[:2000],
                            height=180,
                            key=f"preview_{uploaded_file.name}"
                        )

                        st.markdown("**First chunk preview**")

                        st.text_area(
                            "Chunk preview",
                            value=chunks[0][:1000],
                            height=120,
                            key=f"chunk_{uploaded_file.name}"
                        )

                    except Exception as error:
                        failed_count += 1
                        st.error(
                            f"Failed to process {uploaded_file.name}: {error}"
                        )

            st.divider()

            # Refresh database count after indexing
            final_count = store.count_chunks()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Successful Documents", success_count)

            with col2:
                st.metric("Failed Documents", failed_count)

            with col3:
                st.metric("Chunks Added or Updated", total_chunks)

            st.metric("Total Chunks in ChromaDB", final_count)

            if failed_count == 0:
                st.success(
                    "All uploaded documents were indexed in ChromaDB."
                )
            else:
                st.warning(
                    "Some documents failed. Review their error messages."
                )

        except Exception as error:
            st.error(f"Pipeline initialization failed: {error}")

# ---------------- DOCUMENT SEARCH & RETRIEVAL ----------------
st.divider()

st.subheader("Search Indexed Documents")

st.write(
    "Ask a question about your uploaded documents to retrieve "
    "the most relevant text chunks from ChromaDB."
)

query = st.text_input(
    "Enter your question",
    placeholder="e.g., What are the project deliverables?",
    key="document_search_query"
)

top_k = st.slider(
    "Number of relevant chunks",
    min_value=1,
    max_value=5,
    value=3
)

if st.button("Search Documents", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching your knowledge base..."):
                # Load the same embedding model used for indexing
                embedder, store = get_components()

                # Check whether documents have been indexed
                if store.count_chunks() == 0:
                    st.warning(
                        "No indexed chunks found. Upload and process "
                        "documents before searching."
                    )

                else:
                    # Convert the question into a vector
                    query_embedding = embedder.embed_query(query)

                    # Retrieve matching chunks from ChromaDB
                    results = store.search(
                        query_embedding=query_embedding,
                        top_k=top_k
                    )

                    documents = results.get("documents", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0]

                    if not documents:
                        st.info("No matching chunks were found.")

                    else:
                        st.success(
                            f"Retrieved {len(documents)} relevant chunk(s)."
                        )

                        for index, document in enumerate(documents):
                            metadata = metadatas[index] or {}
                            distance = distances[index]

                            filename = metadata.get(
                                "filename", "Unknown document"
                            )
                            chunk_number = metadata.get(
                                "chunk_index", index
                            )

                            with st.expander(
                                f"Result {index + 1}: {filename}",
                                expanded=True
                            ):
                                st.caption(
                                    f"Chunk number: {chunk_number + 1}"
                                )

                                st.markdown("**Retrieved text**")
                                st.write(document)

                                st.caption(
                                    f"Cosine distance: {distance:.4f} "
                                    "(lower generally means more similar)"
                                )

        except Exception as error:
            st.error(f"Search failed: {error}")
