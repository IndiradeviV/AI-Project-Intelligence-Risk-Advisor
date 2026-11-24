import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import json

from ingestion.document_loader import extract_text
from ingestion.chunker import split_text
from embeddings.embedding_model import EmbeddingModel
from database.chroma_store import ChromaStore

from agents.scope_extraction_agent import ScopeExtractionAgent
from agents.risk_detection_agent import RiskDetectionAgent
from agents.blocker_action_agent import BlockerActionAgent
from agents.documentation_agent import DocumentationAgent
from agents.project_health_agent import ProjectHealthAgent
from agents.project_intelligence_agent import ProjectIntelligenceAgent


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ID = "ai_project_intelligence_risk_advisor"

st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# SESSION STATE
# =====================================================

SESSION_DEFAULTS = {
    "health_result": None,
    "scope_result": None,
    "risk_result": None,
    "blocker_result": None,
    "documentation_result": None,
    "chat_history": []
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>
    :root {
        --cream: #F5F3EA;
        --sidebar: #263B32;
        --green: #54775B;
        --green-dark: #3D5E46;
        --sage: #78947C;
        --charcoal: #39434A;
        --gold: #C49A57;
    }

    .stApp {
        background-color: var(--cream);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--sidebar);
    }

    section[data-testid="stSidebar"] * {
        color: #F5F3EA;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    section[data-testid="stSidebar"] label {
        font-size: 15px !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        gap: 9px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        font-size: 15px !important;
    }

    h1, h2, h3 {
        color: #263B32 !important;
    }

    p, label, .stCaption {
        color: #39434A;
    }

    .stButton > button,
    .stDownloadButton > button {
        background-color: #54775B !important;
        color: #FFFFFF !important;
        border: 1px solid #54775B !important;
        border-radius: 6px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }

    .stButton > button p,
    .stDownloadButton > button p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #3D5E46 !important;
        border-color: #3D5E46 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E2D9;
        border-radius: 8px;
        padding: 14px;
    }

    div[data-testid="stProgressBar"] > div > div {
        background-color: #54775B;
    }

    /* PROJECT ASSISTANT: DIFFERENTIATE QUESTIONS AND ANSWERS */
    div[class*="st-key-chat-user-"] {
        background-color: #E8EEF8 !important;
        border: 1px solid #C9D7EA !important;
        border-left: 5px solid #54749A !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin: 12px 0 !important;
    }

    div[class*="st-key-chat-assistant-"] {
        background-color: #E8F0E8 !important;
        border: 1px solid #C5D6C6 !important;
        border-left: 5px solid #54775B !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin: 12px 0 !important;
    }

    div[class*="st-key-chat-user-"] p,
    div[class*="st-key-chat-assistant-"] p {
        color: #263B32 !important;
    }


    /* COLOURED RESULT CARDS */
    div[class*="st-key-item-card-"] {
        background-color: #E8F0E8 !important;
        border: 1px solid #C5D6C6 !important;
        border-left: 5px solid #54775B !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        margin: 10px 0 16px 0 !important;
    }

    div[class*="st-key-item-card-"] [data-testid="stMarkdownContainer"] p {
        color: #263B32 !important;
        line-height: 1.65 !important;
    }

    div[class*="st-key-item-card-"] h3 {
        color: #263B32 !important;
        margin-top: 0.2rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# SHARED FUNCTIONS
# =====================================================

@st.cache_resource
def get_components():
    embedder = EmbeddingModel()
    store = ChromaStore(persist_directory="chroma_db")
    return embedder, store


def show_page_header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()


def show_error(error, title="Something went wrong"):
    st.error(f"{title}: {error}")


def get_database_count():
    try:
        _, store = get_components()
        return store.count_chunks(), "Connected"
    except Exception:
        return 0, "Unavailable"


def display_items(items, empty_message="No information available."):

    if not items:
        st.info(empty_message)
        return

    for index, item in enumerate(items, start=1):

        # Each result gets its own softly coloured card.
        with st.container(
            border=True,
            key=f"item-card-{id(items)}-{index}"
        ):

            if isinstance(item, dict):

                # User story format
                if item.get("title") or item.get("story"):

                    title = item.get("title", f"Item {index}")
                    st.markdown(f"### {index}. {title}")

                    if item.get("story"):
                        st.markdown("**Story**")
                        st.write(item["story"])

                    criteria = item.get("acceptance_criteria", [])

                    if criteria:
                        st.markdown("**Acceptance Criteria**")

                        for criterion_index, criterion in enumerate(
                            criteria, start=1
                        ):
                            if isinstance(criterion, dict):
                                criterion_text = (
                                    criterion.get("description")
                                    or criterion.get("criterion")
                                    or criterion.get("recommendation")
                                    or str(criterion)
                                )
                            else:
                                criterion_text = str(criterion)

                            st.markdown(
                                f"{criterion_index}. {criterion_text}"
                            )

                    if item.get("evidence"):
                        st.markdown("**Evidence**")
                        st.write(item["evidence"])

                    # Show any additional fields without exposing raw JSON.
                    known_fields = {
                        "title", "story", "acceptance_criteria", "evidence"
                    }

                    for key, value in item.items():
                        if key in known_fields or value in (None, "", [], {}):
                            continue

                        label = key.replace("_", " ").title()
                        st.markdown(f"**{label}**")
                        if isinstance(value, list):
                            for entry in value:
                                st.markdown(f"- {entry}")
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                st.write(
                                    f"**{sub_key.replace('_', ' ').title()}:** "
                                    f"{sub_value}"
                                )
                        else:
                            st.write(value)

                else:
                    # Generic structured result: display readable labels,
                    # never the raw dictionary representation.
                    heading = (
                        item.get("description")
                        or item.get("item")
                        or item.get("name")
                        or item.get("goal")
                        or item.get("deliverable")
                        or item.get("action")
                        or item.get("risk")
                        or item.get("blocker")
                        or item.get("milestone")
                        or item.get("responsibility")
                    )

                    if heading:
                        st.markdown(f"**{index}. {heading}**")
                    else:
                        st.markdown(f"**Item {index}**")

                    for key, value in item.items():

                        if key in {
                            "description", "item", "name", "goal",
                            "deliverable", "action", "risk", "blocker",
                            "milestone", "responsibility"
                        }:
                            continue

                        if value is None or value == "" or value == []:
                            continue

                        label = key.replace("_", " ").title()

                        if isinstance(value, list):
                            st.markdown(f"**{label}**")
                            for entry in value:
                                if isinstance(entry, dict):
                                    for sub_key, sub_value in entry.items():
                                        st.write(
                                            f"**{sub_key.replace('_', ' ').title()}:** "
                                            f"{sub_value}"
                                        )
                                else:
                                    st.markdown(f"- {entry}")

                        elif isinstance(value, dict):
                            st.markdown(f"**{label}**")
                            for sub_key, sub_value in value.items():
                                st.write(
                                    f"**{sub_key.replace('_', ' ').title()}:** "
                                    f"{sub_value}"
                                )

                        else:
                            st.markdown(f"**{label}:** {value}")

            else:
                st.markdown(f"**{index}.** {item}")


def json_download_button(label, data, filename, key):
    st.download_button(
        label=label,
        data=json.dumps(data, indent=4, ensure_ascii=False, default=str),
        file_name=filename,
        mime="application/json",
        key=key
    )


def display_health_report(result):
    """Display the project health result in a readable dashboard format."""

    score = result.get("overall_score", 0)
    status = result.get("health_status", "Unknown")

    st.subheader("Overall Project Health")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Health Score", f"{score}/100")

    with col2:
        if status == "Healthy":
            st.success(f"Status: {status}")
        elif status == "Moderate":
            st.warning(f"Status: {status}")
        else:
            st.error(f"Status: {status}")

    with col3:
        st.metric(
            "Identified Risks",
            result.get("identified_risk_count", 0)
        )

    try:
        st.progress(max(0.0, min(1.0, float(score) / 100)))
    except (ValueError, TypeError):
        pass

    st.divider()
    st.subheader("Health Breakdown")

    scores = result.get("dimension_scores", {})

    dimensions = [
        ("Scope Clarity", "scope_clarity"),
        ("Timeline Health", "timeline_health"),
        ("Blocker Health", "blocker_health")
    ]

    for label, key in dimensions:
        value = scores.get(key, 0)
        col1, col2 = st.columns([1, 4])

        with col1:
            st.write(f"**{label}**")

        with col2:
            try:
                st.progress(max(0.0, min(1.0, float(value) / 100)))
            except (ValueError, TypeError):
                st.write("Score unavailable")

        st.caption(f"{value}/100")

    st.divider()
    st.subheader("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documented Blockers", result.get("blocker_count", 0))

    with col2:
        st.metric("Identified Risks", result.get("identified_risk_count", 0))

    st.divider()
    st.subheader("Delivery Forecast")

    forecast = result.get("delivery_forecast", {})

    if isinstance(forecast, dict):
        forecast_status = forecast.get("status", "Unknown")

        if forecast_status == "On Track":
            st.success(forecast_status)
        elif forecast_status == "At Risk":
            st.error(forecast_status)
        else:
            st.warning(forecast_status)

        st.write(forecast.get("reason", "No explanation available."))
    elif forecast:
        st.write(forecast)
    else:
        st.info("No delivery forecast available.")

    st.divider()
    st.subheader("Information to Be Addressed")

    missing = result.get("missing_information", [])

    if missing:
        for index, item in enumerate(missing, start=1):
            if isinstance(item, dict):
                description = item.get(
                    "item",
                    item.get("description", "Not specified")
                )
                reason = item.get("reason", "")

                with st.expander(f"{index}. {description}"):
                    if reason:
                        st.write(reason)
            else:
                st.write(f"{index}. {item}")
    else:
        st.success("No missing information reported.")


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:
    st.markdown("## Project Advisor")
    st.caption("AI Project Intelligence")

    st.markdown("---")
    st.markdown("**WORKSPACE**")

    pages = [
        "Dashboard",
        "Documents",
        "Scope Analysis",
        "Risk Analysis",
        "Blockers & Actions",
        "Documentation",
        "Project Health",
        "Project Assistant",
        "Reports & Insights"
    ]

    selected_page = st.radio(
        "Navigate",
        pages,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**DATABASE STATUS**")

    chunk_count, database_status = get_database_count()

    if database_status == "Connected":
        st.success("Connected")
    else:
        st.error("Unavailable")

    st.caption(f"{chunk_count} indexed chunks")

    st.markdown("---")
    st.caption("Python · Streamlit · ChromaDB · Groq")


# =====================================================
# DASHBOARD PAGE
# =====================================================

if selected_page == "Dashboard":

    show_page_header(
        "Project Insights Dashboard",
        "Monitor project health, risks, action items and project scope."
    )

    st.write(
        "Generate a consolidated view from the project documents already "
        "indexed in ChromaDB. Existing documents are not modified."
    )

    if st.button("Generate / Refresh Project Insights", type="primary"):
        analysis_tasks = [
            (
                "Scope analysis",
                lambda: ScopeExtractionAgent().extract_scope(
                    "Extract the project goals, deliverables, milestones, "
                    "timelines, responsibilities and missing information."
                ),
                "scope_result"
            ),
            (
                "Risk analysis",
                lambda: RiskDetectionAgent().detect_risks(
                    "Identify project risks, schedule risks, dependency gaps, "
                    "delivery challenges and mitigation actions."
                ),
                "risk_result"
            ),
            (
                "Blocker and action analysis",
                lambda: BlockerActionAgent().identify_blockers(
                    "Identify project blockers, action items, unresolved "
                    "dependencies, responsible owners, deadlines and missing information."
                ),
                "blocker_result"
            ),
            (
                "Documentation generation",
                lambda: DocumentationAgent().generate_documentation(
                    query=(
                        "Generate project documentation containing user stories, "
                        "risk register, action items and missing information."
                    ),
                    top_k=8
                ),
                "documentation_result"
            ),
           (
    "Project health assessment",
    lambda: ProjectHealthAgent().calculate_project_health(
        st.session_state.scope_result,
        st.session_state.risk_result,
        st.session_state.blocker_result
    ),
    "health_result"
)
        ]

        completed = []
        failed = []
        progress = st.progress(0)
        status_text = st.empty()

        for task_index, (task_name, task_function, state_key) in enumerate(analysis_tasks):
            status_text.info(f"Running {task_name}...")
            try:
                task_result = task_function()
                if isinstance(task_result, dict) and task_result.get("status") == "success":
                    st.session_state[state_key] = task_result
                    completed.append(task_name)
                else:
                    failed.append(task_name)
            except Exception as error:
                failed.append(f"{task_name}: {error}")

            progress.progress((task_index + 1) / len(analysis_tasks))

        status_text.empty()
        if completed:
            st.success("Insights generated: " + ", ".join(completed))
        if failed:
            st.warning("Some analyses could not be completed: " + "; ".join(failed))
        if completed:
            st.rerun()

    # Read the latest results already generated by the existing agents.
    health = st.session_state.health_result or {}
    scope = st.session_state.scope_result or {}
    risk = st.session_state.risk_result or {}
    blocker = st.session_state.blocker_result or {}
    documentation = st.session_state.documentation_result or {}

    health_data = health if health.get("status") == "success" else {}
    scope_data = scope.get("extracted_data", {}) if scope.get("status") == "success" else {}
    risk_data = risk.get("risk_analysis", {}) if risk.get("status") == "success" else {}
    blocker_data = blocker.get("blocker_analysis", {}) if blocker.get("status") == "success" else {}
    documentation_data = documentation.get("documentation", {}) if documentation.get("status") == "success" else {}

    # Dashboard values
    health_score = health_data.get("overall_score", "—")
    health_status = health_data.get("health_status", "Not Assessed")
    risks = risk_data.get("risks", []) or []
    blockers = blocker_data.get("blockers", []) or []
    actions = blocker_data.get("action_items", []) or documentation_data.get("action_items", []) or []
    goals = scope_data.get("project_goals", []) or []
    deliverables = scope_data.get("deliverables", []) or []
    milestones = scope_data.get("milestones", []) or []
    total_chunks, db_status = get_database_count()

    st.subheader("Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Project Health", f"{health_score}/100" if health_score != "—" else "—")
    with col2:
        st.metric("Identified Risks", len(risks))
    with col3:
        st.metric("Action Items", len(actions))
    with col4:
        st.metric("Deliverables", len(deliverables))

    
    st.divider()
    st.subheader("Project Health Summary")

    if health_data:
        try:
            numeric_score = max(0.0, min(100.0, float(health_score)))
            st.progress(numeric_score / 100)
        except (TypeError, ValueError):
            st.info(f"Health score: {health_score}")

        st.caption(f"Health status: {health_status}")

        dimensions = health_data.get("dimension_scores", {})

        if isinstance(dimensions, dict) and dimensions:

            # Display individual health scores
            dimension_cols = st.columns(len(dimensions))

            for column, (dimension, value) in zip(
                dimension_cols, dimensions.items()
            ):
                column.metric(
                    dimension.replace("_", " ").title(),
                    value
                )

            # ==========================================
            # PROJECT HEALTH RADAR CHART
            # ==========================================

            st.divider()
            st.subheader("Project Health Radar Chart")

            categories = [
                "Scope Clarity",
                "Timeline Health",
                "Blocker Health"
            ]

            values = [
                float(dimensions.get("scope_clarity", 0)),
                float(dimensions.get("timeline_health", 0)),
                float(dimensions.get("blocker_health", 0))
            ]

            # Close the radar chart
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig = go.Figure()

            fig.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    name="Project Health",
                    line=dict(
                        color="#54775B",
                        width=3
                    ),
                    fillcolor="rgba(84, 119, 91, 0.25)"
                )
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[0, 20, 40, 60, 80, 100]
                    )
                ),
                showlegend=False,
                height=450,
                paper_bgcolor="#F5F3EA",
                font=dict(color="#263B32"),
                margin=dict(
                    l=50,
                    r=50,
                    t=40,
                    b=40
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        else:
            st.info("Health dimension scores are not available.")

    else:
        st.info(
            "Project health is not available yet. "
            "Run the assessment from the Project Health page."
        )


    st.divider()
    st.subheader("Project Risk Summary")
    if risks:
        risk_rows = []
        for item in risks:
            if isinstance(item, dict):
                risk_rows.append({
                    "Risk": item.get("risk", "Not specified"),
                    "Category": item.get("category", "Unknown"),
                    "Severity": item.get("severity", "Unknown"),
                    "Likelihood": item.get("likelihood", "Unknown"),
                    "Impact": item.get("impact", "Unknown")
                })
            else:
                risk_rows.append({"Risk": str(item), "Category": "Unknown", "Severity": "Unknown", "Likelihood": "Unknown", "Impact": "Unknown"})
        st.dataframe(risk_rows, use_container_width=True, hide_index=True)
        forecast = risk_data.get("delivery_forecast", {})
        if isinstance(forecast, dict) and forecast:
            st.markdown("**Delivery Forecast**")
            st.write("**Status:**", forecast.get("status", "Not specified"))
            st.write("**Reason:**", forecast.get("reason", "Not provided"))
        elif forecast:
            st.write("**Delivery Forecast:**", forecast)
    else:
        st.info("No risk results are available. Run Risk Analysis to display identified risks.")

    st.divider()
    st.subheader("Project Action Items")
    if actions:
        display_items(actions)
    else:
        st.info("No action items are available. Run Blockers & Actions or Documentation.")

    st.divider()
    st.subheader("Project Scope Summary")
    scope_col1, scope_col2 = st.columns(2)
    with scope_col1:
        st.markdown("**Project Goals**")
        display_items(goals, "Project goals are not available yet.")
    with scope_col2:
        st.markdown("**Deliverables**")
        display_items(deliverables, "Project deliverables are not available yet.")

    st.markdown("**Project Milestones**")
    display_items(milestones, "Project milestones are not available yet.")

    st.divider()
    st.subheader("Workspace Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**ChromaDB Connection**")
        if db_status == "Connected":
            st.success(f"Connected · {total_chunks} indexed chunks")
        else:
            st.error("Database unavailable")
    with status_col2:
        st.markdown("**Project Workspace**")
        st.success("Ready")

    st.caption("Project Insights Dashboard | Powered by the existing project analysis agents.")

# =====================================================
# DOCUMENTS PAGE
# =====================================================

elif selected_page == "Documents":

    show_page_header(
        "Documents",
        "Upload project files, index their contents and search stored information."
    )

    total_chunks, db_status = get_database_count()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stored Chunks", total_chunks)

    with col2:
        st.metric("Database Status", db_status)

    st.divider()
    st.subheader("Incremental Document Upload")
    st.write(
        "Upload new meeting notes, progress updates or revised project files. "
        "New document versions are added to the existing ChromaDB collection; "
        "previously indexed documents are preserved."
    )

    uploaded_files = st.file_uploader(
        "Choose project documents",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
        key="incremental_document_uploader"
    )

    if st.button("Add Documents to Existing Knowledge Base", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one document.")
        else:
            try:
                embedder, store = get_components()
                total_before = store.count_chunks()
                newly_added_total = 0
                unchanged_total = 0

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()
                    document_id = store.create_document_id(
                        uploaded_file.name,
                        file_bytes
                    )
                    existing_chunks = store.get_document_chunk_count(document_id)

                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if existing_chunks:
                            st.info(
                                f"{uploaded_file.name} is already indexed "
                                f"({existing_chunks} chunks). Checking and refreshing its existing version."
                            )

                        extracted_text = extract_text(uploaded_file.name, file_bytes)
                        chunks = split_text(
                            extracted_text,
                            chunk_size=350,
                            chunk_overlap=50
                        )

                        if not chunks:
                            st.warning(
                                f"No text was extracted from {uploaded_file.name}. "
                                "No database records were changed."
                            )
                            continue

                        embeddings = embedder.embed_texts(chunks)
                        store.add_chunks(
                            filename=uploaded_file.name,
                            file_bytes=file_bytes,
                            chunks=chunks,
                            embeddings=embeddings,
                            project_id=PROJECT_ID
                        )

                    if existing_chunks:
                        unchanged_total += len(chunks)
                        st.success(
                            f"{uploaded_file.name}: existing version refreshed; "
                            f"{len(chunks)} chunks retained with the same IDs."
                        )
                    else:
                        newly_added_total += len(chunks)
                        st.success(
                            f"{uploaded_file.name}: added {len(chunks)} chunks as a new document version."
                        )

                total_after = store.count_chunks()
                st.success(
                    f"Upload completed. New document chunks processed: {newly_added_total}. "
                    f"Previously indexed chunks refreshed: {unchanged_total}. "
                    f"Total chunks now stored: {total_after} (before: {total_before})."
                )
                st.caption(
                    "Existing document versions are preserved. Uploading the exact same file "
                    "again reuses its stable chunk IDs and does not create duplicate records."
                )
                st.rerun()

            except Exception as error:
                show_error(error, "Incremental document upload failed")

    st.divider()
    st.subheader("Search Indexed Documents")
    st.write(
        "Enter a question to retrieve relevant information from your indexed documents."
    )

    search_query = st.text_input(
        "Search query",
        value="What are the main project deliverables?"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=3
    )

    if st.button("Search Documents", type="primary"):

        if not search_query.strip():
            st.warning("Enter a search query.")

        else:
            try:
                embedder, store = get_components()

                query_embedding = embedder.embed_query(search_query)

                results = store.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    project_id=PROJECT_ID
                )

                if results:
                    st.subheader("Search Results")

                    for index, item in enumerate(results, start=1):

                        with st.expander(f"Result {index}"):

                            if isinstance(item, dict):
                                st.write(
                                    item.get(
                                        "text",
                                        item.get("content", str(item))
                                    )
                                )
                                if item.get("filename"):
                                    st.caption(
                                        f"File: {item['filename']}"
                                    )
                                if item.get("distance") is not None:
                                    st.caption(
                                        f"Distance: {item['distance']}"
                                    )
                            else:
                                st.write(item)

                else:
                    st.info("No matching documents found.")

            except Exception as error:
                show_error(error, "Document search failed")


# =====================================================
# SCOPE ANALYSIS PAGE
# =====================================================

elif selected_page == "Scope Analysis":

    show_page_header(
        "Scope Analysis",
        "Extract project goals, deliverables, milestones and responsibilities."
    )

    st.subheader("Scope Extraction Agent")

    scope_query = st.text_area(
        "Analysis Request",
        value=(
            "Extract the project goals, deliverables, milestones, "
            "timelines, responsibilities and missing information."
        ),
        height=100,
        key="scope_query"
    )

    if st.button("Extract Project Scope", type="primary"):

        if not scope_query.strip():
            st.warning("Please enter an analysis request.")

        else:
            try:
                with st.spinner("Analyzing project documents..."):
                    result = ScopeExtractionAgent().extract_scope(scope_query)

                st.session_state.scope_result = result

            except Exception as error:
                show_error(error, "Scope extraction failed")

    result = st.session_state.scope_result

    if result:

        if result.get("status") == "success":

            data = result.get("extracted_data", {})
            st.success("Scope extraction completed.")

            if result.get("source_chunks") is not None:
                st.caption(f"Source chunks: {result.get('source_chunks')}")

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
                    display_items(data.get(key, []))

        elif result.get("status") == "no_context":
            st.warning(result.get("message", "No project context found."))

        else:
            st.error("Scope extraction returned an invalid response.")
            if result.get("raw_response"):
                st.code(result["raw_response"])


# =====================================================
# RISK ANALYSIS PAGE
# =====================================================

elif selected_page == "Risk Analysis":

    show_page_header(
        "Risk Analysis",
        "Identify project risks, dependency gaps and delivery challenges."
    )

    st.subheader("Risk Detection and Delivery Forecasting")

    risk_query = st.text_area(
        "Analysis Request",
        value=(
            "Identify project risks, schedule risks, dependency gaps, "
            "delivery challenges and mitigation actions."
        ),
        height=100,
        key="risk_query"
    )

    if st.button("Analyze Project Risks", type="primary"):

        if not risk_query.strip():
            st.warning("Please enter a risk analysis request.")

        else:
            try:
                with st.spinner("Analyzing project risks..."):
                    result = RiskDetectionAgent().detect_risks(risk_query)

                st.session_state.risk_result = result

            except Exception as error:
                show_error(error, "Risk analysis failed")

    result = st.session_state.risk_result

    if result:

        if result.get("status") == "success":

            data = result.get("risk_analysis", {})
            st.success("Risk analysis completed.")

            forecast = data.get("delivery_forecast", {})

            st.subheader("Delivery Forecast")

            if isinstance(forecast, dict):
                for key, value in forecast.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            elif forecast:
                st.write(forecast)
            else:
                st.info("No delivery forecast available.")

            st.divider()
            st.subheader("Identified Risks")

            risks = data.get("risks", [])

            if risks:
                risk_rows = []

                for risk in risks:
                    if isinstance(risk, dict):
                        risk_rows.append({
                            "Risk": risk.get("risk", "Unnamed"),
                            "Category": risk.get("category", "Unknown"),
                            "Likelihood": risk.get("likelihood", "Unknown"),
                            "Impact": risk.get("impact", "Unknown"),
                            "Severity": risk.get("severity", "Unknown")
                        })

                if risk_rows:
                    st.dataframe(
                        risk_rows,
                        use_container_width=True,
                        hide_index=True
                    )

                for index, risk in enumerate(risks, start=1):
                    with st.expander(
                        f"Risk {index}: "
                        f"{risk.get('risk', 'Unnamed risk') if isinstance(risk, dict) else risk}"
                    ):
                        if isinstance(risk, dict):
                            st.write("**Evidence:**", risk.get("evidence", "Not provided"))
                            st.write("**Mitigation:**", risk.get("mitigation", "Not provided"))
            else:
                st.info("No supported risks were identified.")

            st.divider()
            st.subheader("Dependency Gaps")
            display_items(data.get("dependency_gaps", []))

            st.divider()
            st.subheader("Missing Information")
            display_items(data.get("missing_information", []))

        elif result.get("status") == "no_context":
            st.warning(result.get("message", "No project context found."))

        else:
            st.error("Risk analysis returned an invalid response.")
            if result.get("raw_response"):
                st.code(result["raw_response"])


# =====================================================
# BLOCKERS AND ACTIONS PAGE
# =====================================================

elif selected_page == "Blockers & Actions":

    show_page_header(
        "Blockers & Actions",
        "Identify unresolved issues, action items and dependencies."
    )

    st.subheader("Blocker and Action Item Identification Agent")

    blocker_query = st.text_area(
        "Analysis Request",
        value=(
            "Identify project blockers, action items, unresolved "
            "dependencies, responsible owners, deadlines and "
            "missing information."
        ),
        height=100,
        key="blocker_query"
    )

    if st.button("Identify Blockers and Actions", type="primary"):

        if not blocker_query.strip():
            st.warning("Please enter an analysis request.")

        else:
            try:
                with st.spinner("Analyzing blockers and action items..."):
                    result = BlockerActionAgent().identify_blockers(blocker_query)

                st.session_state.blocker_result = result

            except Exception as error:
                show_error(error, "Blocker analysis failed")

    result = st.session_state.blocker_result

    if result:

        if result.get("status") == "success":

            data = result.get("blocker_analysis", {})
            st.success("Blocker analysis completed.")

            blockers = data.get("blockers", [])
            actions = data.get("action_items", [])
            dependencies = data.get("unresolved_dependencies", [])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Blockers", len(blockers))

            with col2:
                st.metric("Action Items", len(actions))

            with col3:
                st.metric("Dependencies", len(dependencies))

            st.divider()
            st.subheader("Identified Blockers")

            if blockers:
                for index, item in enumerate(blockers, start=1):
                    if isinstance(item, dict):
                        heading = item.get("blocker", "Unnamed blocker")
                    else:
                        heading = str(item)

                    with st.expander(f"{index}. {heading}"):
                        if isinstance(item, dict):
                            st.write("**Evidence:**", item.get("evidence", "Not provided"))
                            st.write("**Impact:**", item.get("impact", "Unknown"))
                            st.write(
                                "**Affected Deliverable:**",
                                item.get("affected_deliverable", "Unknown")
                            )
                            st.write(
                                "**Suggested Resolution:**",
                                item.get("suggested_resolution", "Not provided")
                            )
            else:
                st.info("No documented blockers were identified.")

            st.divider()
            st.subheader("Action Items")

            if actions:
                action_rows = []

                for item in actions:
                    if isinstance(item, dict):
                        action_rows.append({
                            "Action": item.get("action", "Unnamed action"),
                            "Owner": item.get("owner", "Not specified"),
                            "Deadline": item.get("deadline", "Not specified"),
                            "Priority": item.get("priority", "Unknown")
                        })

                if action_rows:
                    st.dataframe(
                        action_rows,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    display_items(actions)
            else:
                st.info("No action items were identified.")

            st.divider()
            st.subheader("Unresolved Dependencies")
            display_items(dependencies)

            st.divider()
            st.subheader("Missing Information")
            display_items(data.get("missing_information", []))

        elif result.get("status") == "no_context":
            st.warning(result.get("message", "No project context found."))

        else:
            st.error("Blocker analysis returned an invalid response.")
            if result.get("raw_response"):
                st.code(result["raw_response"])


# =====================================================
# DOCUMENTATION PAGE
# =====================================================

elif selected_page == "Documentation":

    show_page_header(
        "Documentation",
        "Generate structured project documentation from indexed documents."
    )

    st.subheader("Documentation Generation Agent")

    documentation_query = st.text_area(
        "Documentation Request",
        value=(
            "Generate project documentation containing user stories, "
            "risk register, action items and missing information."
        ),
        height=100,
        key="documentation_query"
    )

    if st.button("Generate Documentation", type="primary"):

        if not documentation_query.strip():
            st.warning("Please enter a documentation request.")

        else:
            try:
                with st.spinner("Generating project documentation..."):
                    result = DocumentationAgent().generate_documentation(
                        query=documentation_query,
                        top_k=8
                    )

                st.session_state.documentation_result = result

            except Exception as error:
                show_error(error, "Documentation generation failed")

    result = st.session_state.documentation_result

    if result:

        if result.get("status") == "success":

            data = result.get("documentation", {})
            st.success("Documentation generated successfully.")

            st.subheader("User Stories")
            display_items(data.get("user_stories", []))

            st.divider()
            st.subheader("Risk Register")
            display_items(data.get("risk_register", []))

            st.divider()
            st.subheader("Documented Action Items")
            display_items(data.get("action_items", []))

            st.divider()
            st.subheader("Missing Information")
            display_items(data.get("missing_information", []))

            json_download_button(
                "Download Documentation",
                result,
                "project_documentation.json",
                "download_documentation"
            )

        else:
            st.warning(
                result.get(
                    "message",
                    "Documentation could not be generated completely."
                )
            )
            if result.get("raw_response"):
                st.code(result["raw_response"])



# =====================================================
# PROJECT HEALTH PAGE
# =====================================================

elif selected_page == "Project Health":

    show_page_header(
        "Project Health",
        "Evaluate project progress using scope, timeline and blocker indicators."
    )

    st.subheader("Project Health Assessment")

    st.write(
        "Calculate overall project health using scope clarity, "
        "timeline health and documented blockers."
    )

    if st.button("Run Health Assessment", type="primary"):

        try:
            with st.spinner("Calculating project health..."):

                # Get the previously generated agent results
                scope_result = st.session_state.scope_result
                risk_result = st.session_state.risk_result
                blocker_result = st.session_state.blocker_result

                # Check whether all three agents have completed successfully
                if not scope_result or scope_result.get("status") != "success":
                    st.warning(
                        "Please run Scope Analysis before calculating project health."
                    )

                elif not risk_result or risk_result.get("status") != "success":
                    st.warning(
                        "Please run Risk Analysis before calculating project health."
                    )

                elif not blocker_result or blocker_result.get("status") != "success":
                    st.warning(
                        "Please run Blockers & Actions before calculating project health."
                    )

                else:
                    # Calculate health using existing agent results
                    agent = ProjectHealthAgent()

                    result = agent.calculate_project_health(
                        scope_result,
                        risk_result,
                        blocker_result
                    )

                    st.session_state.health_result = result

        except Exception as error:
            st.error(f"Health assessment failed: {error}")

    # Display the previously calculated health result
    result = st.session_state.health_result

    if result:

        if result.get("status") == "success":

            st.success("Project health calculation completed.")

            display_health_report(result)

            st.download_button(
                "Download Health Report",
                data=json.dumps(
                    result,
                    indent=4,
                    ensure_ascii=False,
                    default=str
                ),
                file_name="project_health_report.json",
                mime="application/json"
            )

        elif result.get("status") == "incomplete":

            st.warning(
                result.get(
                    "message",
                    "Health assessment is incomplete."
                )
            )

            st.write("Agent execution status:")

            st.json({
                "Scope Agent": result.get("scope_status"),
                "Risk Agent": result.get("risk_status"),
                "Blocker Agent": result.get("blocker_status")
            })

        else:

            st.error("Unable to calculate project health.")

            if result.get("raw_response"):
                st.code(result["raw_response"])



# =====================================================
# PROJECT ASSISTANT PAGE
# =====================================================

elif selected_page == "Project Assistant":

    show_page_header(
        "Project Assistant",
        "Ask questions about your indexed project documents."
    )

    st.subheader("Project Conversation")
    st.caption(
        "Questions are shown in blue cards and AI responses in sage-green cards."
    )

    # Display previous conversation in separate coloured cards.
    for index, message in enumerate(st.session_state.chat_history):

        if message["role"] == "user":

            with st.container(
                border=True,
                key=f"chat-user-{index}"
            ):
                st.markdown("**YOUR QUESTION**")
                st.markdown(message["content"])

        else:

            with st.container(
                border=True,
                key=f"chat-assistant-{index}"
            ):
                st.markdown("**PROJECT ASSISTANT**")
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("View Sources"):
                        display_items(message["sources"])

    st.divider()

    question = st.chat_input(
        "Ask something about your project..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.container(
            border=True,
            key=f"chat-user-{len(st.session_state.chat_history) - 1}"
        ):
            st.markdown("**YOUR QUESTION**")
            st.markdown(question)

        try:

            with st.spinner("Searching project knowledge..."):

                response = ProjectIntelligenceAgent().ask_question(question)

            if isinstance(response, dict):

                answer = response.get(
                    "answer",
                    response.get(
                        "response",
                        response.get("message", str(response))
                    )
                )

                sources = response.get("sources", [])

            else:

                answer = str(response)
                sources = []

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": str(answer),
                "sources": sources
            })

            with st.container(
                border=True,
                key=f"chat-assistant-{len(st.session_state.chat_history) - 1}"
            ):
                st.markdown("**PROJECT ASSISTANT**")
                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        display_items(sources)

        except Exception as error:

            st.error(f"Assistant error: {error}")


# =====================================================
# REPORTS AND INSIGHTS PAGE
# =====================================================

elif selected_page == "Reports & Insights":

    show_page_header(
        "Reports & Insights",
        "View and download readable results generated by project agents."
    )

    st.subheader("Project Analysis Reports")

    report_sections = {
        "Scope Analysis": st.session_state.scope_result,
        "Risk Analysis": st.session_state.risk_result,
        "Blockers & Actions": st.session_state.blocker_result,
        "Documentation": st.session_state.documentation_result,
        "Project Health": st.session_state.health_result
    }

    available_reports = {
        name: report
        for name, report in report_sections.items()
        if report is not None
    }

    if available_reports:

        st.success(f"{len(available_reports)} report(s) available.")

        for name, report in available_reports.items():

            safe_name = (
                name.lower()
                .replace(" ", "_")
                .replace("&", "and")
            )

            with st.expander(name, expanded=(name == "Project Health")):

                if name == "Project Health":

                    display_health_report(report)

                elif name == "Risk Analysis":

                    data = report.get("risk_analysis", {})

                    st.subheader("Risk Summary")
                    risks = data.get("risks", [])
                    st.metric("Total Identified Risks", len(risks))

                    if risks:
                        risk_rows = []

                        for item in risks:
                            if isinstance(item, dict):
                                risk_rows.append({
                                    "Risk": item.get("risk", "Unknown"),
                                    "Category": item.get("category", "Unknown"),
                                    "Severity": item.get("severity", "Unknown"),
                                    "Likelihood": item.get("likelihood", "Unknown"),
                                    "Impact": item.get("impact", "Unknown")
                                })
                            else:
                                risk_rows.append({
                                    "Risk": str(item),
                                    "Category": "Not specified",
                                    "Severity": "Not specified",
                                    "Likelihood": "Not specified",
                                    "Impact": "Not specified"
                                })

                        st.dataframe(
                            risk_rows,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No risks were identified.")

                    st.subheader("Delivery Forecast")
                    forecast = data.get("delivery_forecast", {})

                    if isinstance(forecast, dict):
                        st.write("**Status:**", forecast.get("status", "Unknown"))
                        st.write("**Reason:**", forecast.get("reason", "Not available"))
                    elif forecast:
                        st.write(forecast)
                    else:
                        st.info("No delivery forecast available.")

                    st.subheader("Dependency Gaps")
                    display_items(data.get("dependency_gaps", []))

                    st.subheader("Missing Information")
                    display_items(data.get("missing_information", []))

                elif name == "Scope Analysis":

                    data = report.get("extracted_data", {})

                    for title, key in [
                        ("Project Goals", "project_goals"),
                        ("Deliverables", "deliverables"),
                        ("Milestones", "milestones"),
                        ("Timelines", "timelines"),
                        ("Responsibilities", "responsibilities"),
                        ("Missing Information", "missing_information")
                    ]:
                        with st.expander(title):
                            display_items(data.get(key, []))

                elif name == "Blockers & Actions":

                    data = report.get("blocker_analysis", {})

                    st.subheader("Blockers")
                    display_items(data.get("blockers", []))

                    st.subheader("Action Items")
                    display_items(data.get("action_items", []))

                    st.subheader("Unresolved Dependencies")
                    display_items(data.get("unresolved_dependencies", []))

                    st.subheader("Missing Information")
                    display_items(data.get("missing_information", []))

                elif name == "Documentation":

                    data = report.get("documentation", {})

                    st.subheader("User Stories")
                    display_items(data.get("user_stories", []))

                    st.subheader("Risk Register")
                    display_items(data.get("risk_register", []))

                    st.subheader("Action Items")
                    display_items(data.get("action_items", []))

                    st.subheader("Missing Information")
                    display_items(data.get("missing_information", []))

                json_download_button(
                    f"Download {name} (JSON)",
                    report,
                    f"{safe_name}.json",
                    f"download_{safe_name}"
                )

        st.divider()
        st.subheader("Complete Project Report")

        combined_report = {
            "project_id": PROJECT_ID,
            "generated_reports": available_reports
        }

        json_download_button(
            "Download Complete Project Report (JSON)",
            combined_report,
            "project_analysis_report.json",
            "download_complete_report"
        )

    else:
        st.info(
            "No reports are available yet. Run the analysis agents "
            "from the sidebar pages to generate reports."
        )


# =====================================================
# APPLICATION FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Project Intelligence & Risk Advisor | "
    "Project analysis powered by document retrieval and AI agents."
)
