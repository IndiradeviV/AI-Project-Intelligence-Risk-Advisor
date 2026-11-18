from agents.project_health_agent import ProjectHealthAgent
from agents.project_intelligence_agent import ProjectIntelligenceAgent
import streamlit as st

from ingestion.document_loader import extract_text
from ingestion.chunker import split_text
from embeddings.embedding_model import EmbeddingModel
from database.chroma_store import ChromaStore

from agents.scope_extraction_agent import ScopeExtractionAgent
from agents.risk_detection_agent import RiskDetectionAgent
from agents.blocker_action_agent import BlockerActionAgent
from agents.documentation_agent import DocumentationAgent


# ---------------- CONFIGURATION ----------------

PROJECT_ID = "ai_project_intelligence_risk_advisor"

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
    'Document ingestion, project analysis and intelligent documentation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- DATABASE STATUS ----------------

st.subheader("Document Knowledge Base")

st.write(
    "Upload PDF, DOCX, CSV, and TXT files. "
    "The application extracts text, creates chunks, generates embeddings, "
    "and stores them in a persistent ChromaDB database."
)

try:
    embedder, store = get_components()
    existing_count = store.count_chunks()
    database_status = "Connected"
except Exception as error:
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


# ---------------- DOCUMENT PROCESSING ----------------

if st.button("Start Document Processing", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one document.")

    else:
        try:
            with st.spinner("Loading embedding model and connecting to ChromaDB..."):
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

                        # 2. Split into chunks
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
                            embeddings=vectors,
                            project_id=PROJECT_ID
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
                    "All uploaded documents were indexed successfully."
                )
            else:
                st.warning(
                    "Some documents failed. Review their error messages."
                )

        except Exception as error:
            st.error(f"Pipeline initialization failed: {error}")


# ---------------- DOCUMENT SEARCH ----------------

st.divider()

st.subheader("Search Indexed Documents")

st.write(
    "Ask questions about your uploaded documents "
    "to retrieve relevant text chunks."
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

                embedder, store = get_components()

                query_embedding = embedder.embed_query(query)

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
                            "filename",
                            "Unknown document"
                        )

                        chunk_number = metadata.get(
                            "chunk_index",
                            index
                        )

                        with st.expander(
                            f"Result {index + 1}: {filename}"
                        ):

                            st.caption(
                                f"Chunk number: {chunk_number + 1}"
                            )

                            st.write(document)

                            st.caption(
                                f"Cosine distance: {distance:.4f}"
                            )

        except Exception as error:
            st.error(f"Search failed: {error}")


# ---------------- SCOPE EXTRACTION AGENT ----------------

st.divider()

st.subheader("Scope and Deliverable Extraction Agent")

st.write(
    "Extract project goals, deliverables, milestones, timelines, "
    "responsibilities, and missing information."
)

scope_query = st.text_input(
    "What project information do you want to extract?",
    value=(
        "Extract the project goals, deliverables, milestones, "
        "timelines, and responsibilities."
    ),
    key="scope_extraction_query"
)

if st.button("Extract Project Scope", key="extract_scope_button"):

    if not scope_query.strip():
        st.warning("Please enter a request.")

    else:
        try:
            with st.spinner("Analyzing project documents..."):

                agent = ScopeExtractionAgent()
                result = agent.extract_scope(scope_query)

            if result.get("status") == "success":

                data = result.get("extracted_data", {})

                st.success("Project scope extraction completed.")

                st.caption(
                    f"Retrieved source chunks: "
                    f"{result.get('source_chunks', 0)}"
                )

                sections = [
                    ("Project Goals", "project_goals"),
                    ("Deliverables", "deliverables"),
                    ("Milestones", "milestones"),
                    ("Timelines", "timelines"),
                    ("Responsibilities", "responsibilities"),
                    ("Missing Information", "missing_information")
                ]

                for title, key in sections:

                    with st.expander(title, expanded=True):

                        items = data.get(key, [])

                        if not items:
                            st.info("No information extracted.")

                        else:
                            for item in items:

                                if isinstance(item, dict):

                                    st.write(
                                        item.get(
                                            "description",
                                            item.get(
                                                "responsibility",
                                                "Not specified"
                                            )
                                        )
                                    )

                                    if item.get("person"):
                                        st.write(
                                            "Person:",
                                            item["person"]
                                        )

                                    if item.get("evidence"):
                                        st.caption(
                                            f"Evidence: {item['evidence']}"
                                        )

                                else:
                                    st.write(item)

            elif result.get("status") == "no_context":
                st.warning(result.get("message", "No context found."))

            else:
                st.error("The agent returned an invalid response.")

                if result.get("raw_response"):
                    st.code(result["raw_response"])

        except Exception as error:
            st.error(f"Scope extraction failed: {error}")


# ---------------- RISK DETECTION AGENT ----------------

st.divider()

st.subheader("Risk Detection and Delivery Forecasting Agent")

st.write(
    "Identify project risks, dependencies, delivery challenges, "
    "and recommended mitigation actions."
)

risk_query = st.text_input(
    "What risks should the agent investigate?",
    value=(
        "Identify project risks, schedule risks, dependency gaps, "
        "delivery challenges, and mitigation actions."
    ),
    key="risk_detection_query"
)

if st.button("Analyze Project Risks", key="analyze_project_risks"):

    if not risk_query.strip():
        st.warning("Please enter a risk analysis request.")

    else:
        try:
            with st.spinner("Analyzing project risks..."):

                agent = RiskDetectionAgent()
                result = agent.detect_risks(risk_query)

            if result.get("status") == "success":

                data = result.get("risk_analysis", {})

                st.success("Risk analysis completed.")

                st.caption(
                    f"Retrieved source chunks: "
                    f"{result.get('source_chunks', 0)}"
                )

                forecast = data.get("delivery_forecast", {})

                st.markdown("### Delivery Forecast")

                st.write(
                    "**Status:**",
                    forecast.get(
                        "status",
                        "Insufficient Information"
                    )
                )

                st.write(
                    "**Reason:**",
                    forecast.get(
                        "reason",
                        "No forecast explanation provided."
                    )
                )

                st.markdown("### Identified Risks")

                risks = data.get("risks", [])

                if risks:

                    for index, risk in enumerate(risks, start=1):

                        with st.expander(
                            f"{index}. {risk.get('risk', 'Unnamed risk')}"
                        ):

                            st.write(
                                "**Category:**",
                                risk.get("category", "Unknown")
                            )

                            st.write(
                                "**Likelihood:**",
                                risk.get("likelihood", "Unknown")
                            )

                            st.write(
                                "**Impact:**",
                                risk.get("impact", "Unknown")
                            )

                            st.write(
                                "**Severity:**",
                                risk.get("severity", "Unknown")
                            )

                            st.write(
                                "**Evidence:**",
                                risk.get("evidence", "Not provided")
                            )

                            st.write(
                                "**Suggested mitigation:**",
                                risk.get("mitigation", "Not provided")
                            )

                else:
                    st.info("No supported risks were identified.")

                st.markdown("### Dependency Gaps")

                dependencies = data.get("dependency_gaps", [])

                if dependencies:

                    for dependency in dependencies:

                        st.markdown(
                            f"**{dependency.get('dependency', 'Unnamed dependency')}**"
                        )

                        st.write(
                            "Evidence:",
                            dependency.get("evidence", "Not provided")
                        )

                        st.write(
                            "Recommended action:",
                            dependency.get(
                                "recommended_action",
                                "Not provided"
                            )
                        )

                        st.divider()

                else:
                    st.info("No dependency gaps were identified.")

                st.markdown("### Missing Information")

                missing = data.get("missing_information", [])

                if missing:
                    for item in missing:
                        st.markdown(f"- {item}")
                else:
                    st.write("No missing information reported.")

            elif result.get("status") == "no_context":
                st.warning(result.get("message", "No project context found."))

            else:
                st.error("The agent returned an invalid response.")

                if result.get("raw_response"):
                    st.code(result["raw_response"])

        except Exception as error:
            st.error(f"Risk analysis failed: {error}")


# ---------------- BLOCKER AND ACTION AGENT ----------------

st.divider()

st.subheader("Blocker and Action Item Identification Agent")

st.write(
    "Identify project blockers, action items, unresolved dependencies, "
    "owners, deadlines, and missing information."
)

blocker_query = st.text_input(
    "What blockers or actions should the agent investigate?",
    value=(
        "Identify project blockers, action items, unresolved dependencies, "
        "responsible owners, deadlines, and missing information."
    ),
    key="blocker_action_query"
)

if st.button(
    "Identify Blockers and Actions",
    key="identify_blockers_button"
):

    if not blocker_query.strip():
        st.warning("Please enter a request.")

    else:
        try:
            with st.spinner("Analyzing blockers and action items..."):

                agent = BlockerActionAgent()
                result = agent.identify_blockers(blocker_query)

            if result.get("status") == "success":

                data = result.get("blocker_analysis", {})

                st.success("Blocker analysis completed.")

                st.caption(
                    f"Retrieved source chunks: "
                    f"{result.get('source_chunks', 0)}"
                )

                st.markdown("### Blockers")

                blockers = data.get("blockers", [])

                if blockers:

                    for index, item in enumerate(blockers, start=1):

                        with st.expander(
                            f"{index}. {item.get('blocker', 'Unnamed blocker')}"
                        ):

                            st.write(
                                "**Evidence:**",
                                item.get("evidence", "Not provided")
                            )

                            st.write(
                                "**Impact:**",
                                item.get("impact", "Unknown")
                            )

                            st.write(
                                "**Affected deliverable:**",
                                item.get("affected_deliverable", "Unknown")
                            )

                            st.write(
                                "**Suggested resolution:**",
                                item.get("suggested_resolution", "Not provided")
                            )

                else:
                    st.info("No documented blockers were identified.")

                st.markdown("### Action Items")

                actions = data.get("action_items", [])

                if actions:

                    for index, item in enumerate(actions, start=1):

                        st.markdown(
                            f"**{index}. {item.get('action', 'Unnamed action')}**"
                        )

                        st.write("Owner:", item.get("owner", "Not specified"))
                        st.write("Deadline:", item.get("deadline", "Not specified"))
                        st.write("Priority:", item.get("priority", "Unknown"))

                        st.caption(
                            f"Evidence: {item.get('evidence', 'Not provided')}"
                        )

                        st.divider()

                else:
                    st.info("No action items were identified.")

                st.markdown("### Unresolved Dependencies")

                dependencies = data.get("unresolved_dependencies", [])

                if dependencies:

                    for item in dependencies:

                        st.markdown(
                            f"**{item.get('dependency', 'Unnamed dependency')}**"
                        )

                        st.write(
                            "Evidence:",
                            item.get("evidence", "Not provided")
                        )

                        st.write(
                            "Next step:",
                            item.get("next_step", "Not provided")
                        )

                else:
                    st.info("No unresolved dependencies were identified.")

                st.markdown("### Missing Information")

                missing = data.get("missing_information", [])

                if missing:

                    for item in missing:
                        st.markdown(f"- {item}")

                else:
                    st.write("No missing information reported.")

            elif result.get("status") == "no_context":
                st.warning(result.get("message", "No project context found."))

            else:
                st.error("The agent returned an invalid response.")

                if result.get("raw_response"):
                    st.code(result["raw_response"])

        except Exception as error:
            st.error(f"Blocker analysis failed: {error}")


# ---------------- MILESTONE 3: DOCUMENTATION GENERATION ----------------

st.divider()

st.subheader("Documentation Generation Agent")

st.write(
    "Generate user stories, risk registers, and structured action items "
    "from indexed project documents."
)

documentation_query = st.text_area(
    "Enter your documentation request",
    value=(
        "Generate user stories with acceptance criteria, "
        "a project risk register with mitigation strategies, "
        "and structured action items with owners, deadlines, "
        "and priorities from the project documents."
    ),
    height=120,
    key="documentation_query"
)

if st.button(
    "Generate Project Documentation",
    type="primary",
    key="generate_documentation_button"
):

    if not documentation_query.strip():
        st.warning("Please enter a documentation request.")

    else:
        try:
            with st.spinner(
                "Generating structured project documentation..."
            ):

                agent = DocumentationAgent()

                result = agent.generate_documentation(
                    query=documentation_query,
                    top_k=8
                )

            if result.get("status") == "success":

                documentation = result.get("documentation", {})

                st.success("Project documentation generated successfully.")

                st.caption(
                    f"Retrieved source chunks: "
                    f"{result.get('source_chunks', 0)}"
                )

                # User Stories
                st.markdown("### 1. User Stories")

                user_stories = documentation.get("user_stories", [])

                if user_stories:

                    for index, story in enumerate(user_stories, start=1):

                        with st.expander(
                            f"{index}. {story.get('title', 'Untitled story')}",
                            expanded=True
                        ):

                            st.write(
                                "**User Story:**",
                                story.get("story", "Not specified")
                            )

                            st.write("**Acceptance Criteria:**")

                            criteria = story.get("acceptance_criteria", [])

                            if criteria:
                                for criterion in criteria:
                                    st.markdown(f"- {criterion}")
                            else:
                                st.info("No acceptance criteria provided.")

                            st.caption(
                                f"Evidence: {story.get('evidence', 'Not provided')}"
                            )

                else:
                    st.info("No user stories were generated.")

                # Risk Register
                st.markdown("### 2. Project Risk Register")

                risks = documentation.get("risk_register", [])

                if risks:

                    for index, risk in enumerate(risks, start=1):

                        with st.expander(
                            f"{index}. {risk.get('risk', 'Unnamed risk')}",
                            expanded=True
                        ):

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(
                                    "**Likelihood:**",
                                    risk.get("likelihood", "Unknown")
                                )

                            with col2:
                                st.write(
                                    "**Impact:**",
                                    risk.get("impact", "Unknown")
                                )

                            st.write(
                                "**Mitigation:**",
                                risk.get("mitigation", "Not specified")
                            )

                            st.caption(
                                f"Evidence: {risk.get('evidence', 'Not provided')}"
                            )

                else:
                    st.info("No risks were generated.")

                # Action Items
                st.markdown("### 3. Structured Action Items")

                action_items = documentation.get("action_items", [])

                if action_items:

                    for index, item in enumerate(action_items, start=1):

                        with st.expander(
                            f"{index}. {item.get('action', 'Unnamed action')}",
                            expanded=True
                        ):

                            st.write(
                                "**Owner:**",
                                item.get("owner", "Not specified")
                            )

                            st.write(
                                "**Deadline:**",
                                item.get("deadline", "Not specified")
                            )

                            st.write(
                                "**Priority:**",
                                item.get("priority", "Unknown")
                            )

                            st.caption(
                                f"Evidence: {item.get('evidence', 'Not provided')}"
                            )

                else:
                    st.info("No action items were generated.")

                # Missing Information
                st.markdown("### 4. Missing Information")

                missing = documentation.get("missing_information", [])

                if missing:

                    for item in missing:
                        st.markdown(f"- {item}")

                else:
                    st.write("No missing information reported.")

                # Download JSON
                st.download_button(
                    label="Download Documentation as JSON",
                    data=__import__("json").dumps(
                        documentation,
                        indent=4,
                        ensure_ascii=False
                    ),
                    file_name="project_documentation.json",
                    mime="application/json"
                )

            elif result.get("status") == "no_context":

                st.warning(
                    result.get(
                        "message",
                        "No indexed project documents were found."
                    )
                )

            else:

                st.error(
                    "The Documentation Agent returned an invalid response."
                )

                if result.get("raw_response"):
                    st.code(result["raw_response"])

        except Exception as error:
            st.error(f"Documentation generation failed: {error}")


# ---------------- MILESTONE 3: PROJECT HEALTH DASHBOARD ----------------

st.divider()

st.subheader("Project Health Scoring Dashboard")

st.write(
    "Evaluate overall project health using scope clarity, "
    "timeline health, and documented blockers."
)

if st.button(
    "Calculate Project Health",
    type="primary",
    key="calculate_project_health"
):

    try:
        with st.spinner(
            "Analyzing project scope, timeline, and blockers..."
        ):

            agent = ProjectHealthAgent()
            result = agent.calculate_project_health()

        if result.get("status") == "success":

            overall_score = result.get("overall_score", 0)
            health_status = result.get("health_status", "Unknown")

            scores = result.get("dimension_scores", {})

            st.success("Project health calculation completed.")

            # Overall score
            st.markdown("### Overall Project Health")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Overall Health Score",
                    f"{overall_score}/100"
                )

            with col2:
                st.metric(
                    "Health Status",
                    health_status
                )

            st.progress(overall_score / 100)

            st.divider()

            # Dimension-wise scores
            st.markdown("### Dimension-wise Health Breakdown")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Scope Clarity",
                    f"{scores.get('scope_clarity', 0)}/100"
                )

                st.progress(
                    scores.get("scope_clarity", 0) / 100
                )

            with col2:
                st.metric(
                    "Timeline Health",
                    f"{scores.get('timeline_health', 0)}/100"
                )

                st.progress(
                    scores.get("timeline_health", 0) / 100
                )

            with col3:
                st.metric(
                    "Blocker Health",
                    f"{scores.get('blocker_health', 0)}/100"
                )

                st.progress(
                    scores.get("blocker_health", 0) / 100
                )

            st.divider()

            # Additional project information
            st.markdown("### Project Overview")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Identified Risks",
                    result.get("identified_risk_count", 0)
                )

            with col2:
                st.metric(
                    "Documented Blockers",
                    result.get("blocker_count", 0)
                )

            # Delivery forecast
            forecast = result.get("delivery_forecast", {})

            st.markdown("### Delivery Forecast")

            st.write(
                "**Status:**",
                forecast.get(
                    "status",
                    "Insufficient Information"
                )
            )

            st.write(
                "**Reason:**",
                forecast.get(
                    "reason",
                    "No forecast explanation provided."
                )
            )

            # Missing information
            st.markdown("### Missing Information")

            missing = result.get("missing_information", [])

            if missing:

               for item in missing:
                     if isinstance(item, dict):
                            st.markdown(
                                f"- **{item.get('description', 'Not specified')}**"
                            )

                            if item.get("evidence"):
                                st.caption(f"Evidence: {item['evidence']}")
                     else:
                             st.markdown(f"- {item}")

            else:
                st.success("No missing information reported.")

        elif result.get("status") == "incomplete":

            st.warning(result.get("message"))

            st.write("Agent execution status:")

            st.json({
                "Scope Agent": result.get("scope_status"),
                "Risk Agent": result.get("risk_status"),
                "Blocker Agent": result.get("blocker_status")
            })

        else:
            st.error("Unable to calculate project health.")

    except Exception as error:
        st.error(f"Project health calculation failed: {error}")
        
# ---------------- MILESTONE 3: CONVERSATIONAL PROJECT ASSISTANT ----------------

st.divider()

st.subheader("Conversational Project Intelligence Assistant")

st.write(
    "Ask questions about your project. The AI assistant retrieves "
    "relevant information from indexed documents and generates "
    "evidence-based answers."
)

# Initialize conversation history
if "project_chat_history" not in st.session_state:
    st.session_state.project_chat_history = []

# Clear chat
col1, col2 = st.columns([5, 1])

with col2:
    if st.button("Clear Chat", key="clear_project_chat"):
        st.session_state.project_chat_history = []
        st.rerun()

# Display previous conversation
for chat in st.session_state.project_chat_history:

    with st.chat_message(chat["role"]):

        st.markdown(chat["content"])

        if chat["role"] == "assistant" and chat.get("sources"):

            with st.expander("View Source Documents"):

                for source in chat["sources"]:

                    st.write(
                        f"📄 {source['filename']} "
                        f"(Chunk {source['chunk']})"
                    )

# Suggested questions
st.markdown("### Suggested Questions")

suggested_questions = [
    "What are the main project deliverables?",
    "Who is responsible for registration and login?",
    "What are the biggest project risks?",
    "Are there any documented project blockers?"
]

for index, question in enumerate(suggested_questions):

    if st.button(
        question,
        key=f"suggested_project_question_{index}"
    ):

        st.session_state.selected_project_question = question
        st.rerun()

# Question input
selected_question = st.session_state.pop(
    "selected_project_question",
    ""
)

question = st.chat_input(
    "Ask anything about your project..."
)

if selected_question:
    question = selected_question

if question:

    st.session_state.project_chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing project documents..."):

            try:

                agent = ProjectIntelligenceAgent()

                result = agent.ask_question(
                    question=question,
                    top_k=5
                )

                if result.get("status") == "success":

                    answer = result.get("answer", "")

                    sources = result.get("sources", [])

                    st.markdown(answer)

                    st.caption(
                        f"Retrieved source chunks: "
                        f"{result.get('source_chunks', 0)}"
                    )

                    if sources:

                        with st.expander("View Source Documents"):

                            for source in sources:

                                st.write(
                                    f"📄 {source['filename']} "
                                    f"(Chunk {source['chunk']})"
                                )

                    st.session_state.project_chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                elif result.get("status") == "no_context":

                    message = result.get(
                        "message",
                        "No relevant project information found."
                    )

                    st.warning(message)

                    st.session_state.project_chat_history.append({
                        "role": "assistant",
                        "content": message,
                        "sources": []
                    })

                else:

                    message = result.get(
                        "message",
                        "The assistant could not generate an answer."
                    )

                    st.error(message)

                    st.session_state.project_chat_history.append({
                        "role": "assistant",
                        "content": message,
                        "sources": []
                    })

            except Exception as error:

                st.error(
                    f"Conversational assistant failed: {error}"
                )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI Project Intelligence & Risk Advisor | "
    "Milestone 3 - Documentation Generation"
)
