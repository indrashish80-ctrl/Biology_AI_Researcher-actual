"""
app.py
------
ChatGPT-Style Autonomous Biology Research Web Platform.
Runs multi-source scientific retrieval, claim verification, 
hypothesis formulation, and adversarial review without code exposure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

# Virtual retrieval bridge: ensures loose files on GitHub import without errors
_root = os.path.dirname(os.path.abspath(__file__))
if "retrieval" not in sys.modules and not os.path.exists(os.path.join(_root, "retrieval", "__init__.py")):
    _pkg = types.ModuleType("retrieval")
    _pkg.__path__ = [_root]
    _pkg.__file__ = os.path.join(_root, "__init__.py")
    sys.modules["retrieval"] = _pkg

import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="BioSynthetica AI — Biology Research Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling: ChatGPT Clean Aesthetic with Modern Typography
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 900px;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }

    /* Message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.6rem 1rem;
    }

    /* Reasoning card styling (ChatGPT o1 / DeepSeek R1 style) */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.02);
        margin-bottom: 0.8rem;
    }

    /* Badges & Metrics */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
    }
    .badge-claim { background: #1e3a8a; color: #93c5fd; }
    .badge-hypo { background: #064e3b; color: #6ee7b7; }
    .badge-flag { background: #7f1d1d; color: #fca5a5; }

    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Core Engine Imports
# -----------------------------------------------------------------------------
from ai_router import AIRouter
from research_engine import ResearchEngine


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your **Autonomous Biology Research Assistant**.\n\n"
                "I search primary literature across **PubMed, PMC, and Europe PMC**, "
                "extract **quote-grounded empirical claims**, detect cross-study contradictions, and "
                "subject all findings to an **Adversarial Critic**.\n\n"
                "To get started, enter your **API key(s)** in the left sidebar and enter any biology research question."
            ),
        }
    ]

if "last_dossier" not in st.session_state:
    st.session_state.last_dossier = None

if "last_report_md" not in st.session_state:
    st.session_state.last_report_md = None


# -----------------------------------------------------------------------------
# Sidebar: User Credentials & Execution Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 **BioSynthetica Engine**")
    st.caption("Controlled, Source-Restricted Biological Intelligence")
    st.markdown("---")

    st.subheader("🔑 AI Provider Credentials")
    st.caption("Your keys are retained in your browser session memory only.")

    user_groq = st.text_input(
        "Groq API Key",
        type="password",
        value="",  # never pre-fill from the server .env (that leaked the owner's keys)
        help="Fastest inference (Llama-3.1 8B)",
    )
    user_gemini = st.text_input(
        "Gemini API Key",
        type="password",
        value="",  # never pre-fill from the server .env (that leaked the owner's keys)
        help="Google Gemini Flash models",
    )
    user_nvidia = st.text_input(
        "NVIDIA NIM Key",
        type="password",
        value="",  # never pre-fill from the server .env (that leaked the owner's keys)
        help="NVIDIA hosted models",
    )

    # Keys are passed to a per-session AIRouter (see below) and NEVER copied into
    # os.environ, which is shared by every visitor of the same Streamlit server.
    user_keys = {"groq": user_groq, "gemini": user_gemini, "nvidia": user_nvidia}

    st.markdown("---")
    st.subheader("⚙️ Investigation Parameters")
    
    investigation_mode = st.radio(
        "Research Protocol",
        ["Dynamic ReAct Agent (Interactive)", "Deep Research (Pipeline Synthesis)"],
        index=0,
    )
    
    max_papers = st.slider("Max Papers to Ingest", min_value=2, max_value=8, value=4)
    enable_geo = st.checkbox("Empirical NCBI GEO Loop Closure", value=True)

    st.markdown("---")
    st.subheader("🧪 Experimental Dataset (Optional)")
    st.caption("Upload experimental data to test replication and evaluate hypotheses.")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    group_col = None
    outcome_col = None
    if uploaded_file is not None:
        try:
            import pandas as pd
            if uploaded_file.name.endswith(".csv"):
                df_preview = pd.read_csv(uploaded_file)
            else:
                df_preview = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded: `{uploaded_file.name}` ({len(df_preview)} rows)")
            cols = list(df_preview.columns)
            group_col = st.selectbox("Treatment/Grouping Column", cols, index=0)
            outcome_col = st.selectbox("Measured Outcome Column", cols, index=min(1, len(cols) - 1))
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation cleared. Ready for a new biological question."}
        ]
        st.session_state.last_dossier = None
        st.session_state.last_report_md = None
        st.rerun()


# -----------------------------------------------------------------------------
# Main Chat Display
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🧬 Biological Research AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Grounded literature extraction, hypothesis formulation, and adversarial review.</div>',
    unsafe_allow_html=True,
)

# Render past chat conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display rich artifacts attached to assistant messages
        if "artifacts" in message:
            arts = message["artifacts"]
            if arts.get("report"):
                with st.expander("📄 View Full Markdown Research Report", expanded=False):
                    st.markdown(arts["report"])
                    st.download_button(
                        label="⬇️ Download RESEARCH_REPORT.md",
                        data=arts["report"],
                        file_name="RESEARCH_REPORT.md",
                        mime="text/markdown",
                        key=f"dl_rep_{id(message)}",
                    )
            if arts.get("dossier"):
                with st.expander("📊 View Machine-Readable Knowledge Dossier (JSON)", expanded=False):
                    st.json(arts["dossier"])


# -----------------------------------------------------------------------------
# Chat Input & Research Execution
# -----------------------------------------------------------------------------
prompt = st.chat_input("E.g., How does metformin inhibit cell proliferation and induce apoptosis in breast cancer?")

if prompt:
    # 1. Verification of keys
    if not (user_groq or user_gemini or user_nvidia):
        st.warning("⚠️ Please provide at least one API key in the left sidebar to begin.")
        st.stop()

    # 2. Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Assistant Reasoning & Synthesis Loop
    with st.chat_message("assistant"):
        thought_container = st.status("🔬 Initiating Biological Investigation...", expanded=True)
        temp_data_path = None

        try:
            # Handle uploaded dataset file saving
            if uploaded_file is not None:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    uploaded_file.seek(0)
                    tmp.write(uploaded_file.read())
                    temp_data_path = tmp.name

            # Initialize Engine with Custom Router
            thought_container.write("⚡ Initializing AI Router and database session...")
            router = AIRouter(keys=user_keys)
            engine = ResearchEngine(router=router, use_ai=True)

            if "ReAct Agent" in investigation_mode:
                # Mode A: Autonomous Tool-Calling Agent
                thought_container.write("🤖 Deploying Autonomous Research Agent (PubMed, Europe PMC, NCBI)...")
                agent_res = engine.run_agent(prompt, max_steps=8)

                thought_container.update(label="✅ Autonomous Agent Finished Investigation!", state="complete", expanded=False)
                
                # Fetch output report
                report_path = Path("RESEARCH_REPORT.md")
                report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else "No report generated."
                
                fin = agent_res.get("final_synthesis", {})
                answer_md = (
                    f"### Investigation Complete\n\n"
                    f"The autonomous agent evaluated literature across independent studies and completed an "
                    f"**adversarial review** on the hypotheses.\n\n"
                    f"**Executive Synthesis:**\n"
                    f"{fin.get('executive_summary', 'Synthesis finalized.')}\n\n"
                    f"**Key Molecular Mechanism:**\n"
                    f"`{fin.get('key_mechanism', 'Mechanism reviewed.')}`\n\n"
                    f"Review the full report and adversarial audit flags below."
                )

                st.markdown(answer_md)
                with st.expander("📄 View Full Markdown Research Report", expanded=True):
                    st.markdown(report_md)
                    st.download_button(
                        label="⬇️ Download RESEARCH_REPORT.md",
                        data=report_md,
                        file_name="RESEARCH_REPORT.md",
                        mime="text/markdown",
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_md,
                    "artifacts": {"report": report_md},
                })

            else:
                # Mode B: Deep Research Pipeline
                thought_container.write("📚 Harvesting literature & extracting quote-grounded claims...")
                dossier = engine.investigate(
                    question=prompt,
                    experimental_dataset=temp_data_path,
                    exp_group_col=group_col,
                    exp_outcome_col=outcome_col,
                    max_papers=max_papers,
                    enable_geo=enable_geo,
                    output_report="RESEARCH_REPORT.md",
                )

                thought_container.update(label="✅ Deep Research Pipeline Completed!", state="complete", expanded=False)

                report_path = Path("RESEARCH_REPORT.md")
                report_md = report_path.read_text(encoding="utf-8") if report_path.exists() else "No report generated."

                answer_md = (
                    f"### Deep Research Dossier Generated\n\n"
                    f"- **Papers Evaluated:** {dossier.get('papers_analyzed', 0)}\n"
                    f"- **Contradictions Analyzed:** {dossier.get('contradiction_summary', {}).get('conflicting_pairs', 0)} conflicting pairs\n"
                    f"- **Falsifiable Hypotheses:** {len(dossier.get('hypotheses', []))}\n"
                    f"- **Adversarial Audit Flags:** {len(dossier.get('audit_flags', []))}\n\n"
                    f"Review the complete audit report and download the dossier below."
                )

                st.markdown(answer_md)
                with st.expander("📄 View Full Markdown Research Report", expanded=True):
                    st.markdown(report_md)
                    st.download_button(
                        label="⬇️ Download RESEARCH_REPORT.md",
                        data=report_md,
                        file_name="RESEARCH_REPORT.md",
                        mime="text/markdown",
                    )

                with st.expander("📊 View Machine-Readable Knowledge Dossier (JSON)", expanded=False):
                    st.json(dossier)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_md,
                    "artifacts": {"report": report_md, "dossier": dossier},
                })

        except Exception as exc:
            thought_container.update(label="❌ Error during investigation", state="error")
            st.error(f"Investigation Error: {exc}")
        finally:
            if temp_data_path and os.path.exists(temp_data_path):
                try:
                    os.unlink(temp_data_path)
                except Exception:
                    pass

# --------------------------------------------------------------------------- running this file directly
# app.py is the OLD Streamlit interface, kept as an extra command-line tool. `python app.py` runs the module
# without Streamlit, which prints a wall of "missing ScriptRunContext" warnings and starts nothing - so say
# plainly what to run instead of leaving the reader with warnings and no program.
if __name__ == "__main__":
    import sys as _sys
    if not any("streamlit" in str(a).lower() for a in _sys.argv):
        print("This file is the older Streamlit interface, not the BioSynthetica web app.\n"
              "To start BioSynthetica:        python main.py        (or run.bat / run.sh)\n"
              "Then open:                     http://127.0.0.1:8000\n"
              "To run this Streamlit page instead:   streamlit run app.py")
