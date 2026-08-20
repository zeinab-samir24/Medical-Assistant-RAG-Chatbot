import os
import base64
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "chroma_db_500_75")
COLLECTION_NAME = "medical_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 4

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_FILES = ["logo_orange.jpeg", "logo_instant.jpeg", "logo_creativa.jpeg"]

# =====================================================
# PAGE CONFIG — Wide Layout
# =====================================================

st.set_page_config(
    page_title="First Aid Assistant",
    page_icon="⛑️",
    layout="wide",
)

# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    [data-testid*="Bottom"],
    .main,
    .block-container,
    .stApp {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    /* Fast & Dynamic Animated Header Bar */
    @keyframes moveGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes floatParticle1 {
        0% { transform: translateY(0px) translateX(0px) rotate(0deg) scale(1); opacity: 0.3; }
        50% { transform: translateY(-20px) translateX(15px) rotate(180deg) scale(1.2); opacity: 0.8; }
        100% { transform: translateY(0px) translateX(0px) rotate(360deg) scale(1); opacity: 0.3; }
    }

    @keyframes floatParticle2 {
        0% { transform: translateY(0px) translateX(0px) rotate(0deg) scale(1); opacity: 0.2; }
        50% { transform: translateY(25px) translateX(-20px) rotate(-180deg) scale(1.3); opacity: 0.7; }
        100% { transform: translateY(0px) translateX(0px) rotate(-360deg) scale(1); opacity: 0.2; }
    }

    @keyframes pulseGlow {
        0% { opacity: 0.15; transform: scale(0.9); }
        50% { opacity: 0.35; transform: scale(1.1); }
        100% { opacity: 0.15; transform: scale(0.9); }
    }

    .header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 2.2rem 2.8rem;
        min-height: 120px;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        background: linear-gradient(-45deg, #00a896, #0284c7, #2563eb, #0d9488, #059669);
        background-size: 300% 300%;
        animation: moveGradient 4s ease infinite;
        box-shadow: 0 8px 24px rgba(0, 168, 150, 0.22);
    }

    /* Glowing dynamic center aura */
    .header-bar::before {
        content: '';
        position: absolute;
        top: -50%;
        left: 20%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        animation: pulseGlow 3s ease-in-out infinite;
        pointer-events: none;
    }

    .header-bg-shape1 {
        position: absolute;
        top: 10px;
        right: 22%;
        width: 75px;
        height: 75px;
        background: rgba(255, 255, 255, 0.22);
        border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
        animation: floatParticle1 3.5s ease-in-out infinite;
        pointer-events: none;
    }

    .header-bg-shape2 {
        position: absolute;
        bottom: 5px;
        left: 35%;
        width: 55px;
        height: 55px;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 50%;
        animation: floatParticle2 2.8s ease-in-out infinite;
        pointer-events: none;
    }

    .logo-row {
        display: flex;
        align-items: center;
        gap: 14px;
        z-index: 2;
    }

    .logo-row img {
        height: 48px;
        border-radius: 10px;
        background: #ffffff;
        padding: 5px 12px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    .app-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.3rem;
        color: #ffffff;
        margin-bottom: 0.1rem;
        z-index: 2;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }

    .app-title .accent {
        color: #a7f3d0;
    }

    .panel-header {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #0f172a;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Quick Cards */
    .emergency-card {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .emergency-title { color: #be123c; font-weight: 600; font-size: 0.85rem; }
    .emergency-desc { color: #881337; font-size: 0.78rem; margin-top: 2px; }

    .info-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .info-title { color: #15803d; font-weight: 600; font-size: 0.85rem; }
    .info-desc { color: #166534; font-size: 0.78rem; margin-top: 2px; }

    /* Checklist Item Box */
    .checklist-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #334155;
    }

    .disclaimer-box {
        background: #fffbe3;
        border: 1px solid #fef08a;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.75rem;
        color: #854d0e;
        line-height: 1.4;
    }

    /* Welcome Card */
    .welcome-card {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        color: #0f766e;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    /* Chat Bubbles */
    [data-testid="stChatMessage"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatMessageAvatarUser"] { background: #00a896 !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background: #0f172a !important; }

    .pages-pill {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 3px 10px;
        border-radius: 999px;
        background: #ccfbf1;
        border: 1px solid #5eead4;
        font-size: 0.75rem;
        color: #0f766e;
        font-weight: 500;
    }

    div[data-testid="stButton"] button {
        background: #ffffff;
        border: 1px solid #00a896;
        color: #00a896;
        border-radius: 8px;
        font-size: 0.82rem;
        width: 100%;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stButton"] button:hover {
        background: #00a896;
        color: #ffffff;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# HEADER
# =====================================================

def load_logo_b64(filename):
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_tags = ""
for fname in LOGO_FILES:
    b64 = load_logo_b64(fname)
    if b64:
        logo_tags += f'<img src="data:image/jpeg;base64,{b64}" />'

st.markdown(
    f"""
    <div class="header-bar">
        <div class="header-bg-shape1"></div>
        <div class="header-bg-shape2"></div>
        <div class="app-title">⛑️ First Aid <span class="accent">Assistant</span></div>
        <div class="logo-row">{logo_tags}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# LOAD RESOURCES
# =====================================================

@st.cache_resource
def load_resources():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return model, collection

try:
    with st.spinner("Initializing system..."):
        model, collection = load_resources()
except Exception as e:
    st.error(f"Failed to load vector database/models: {e}")
    st.stop()

# =====================================================
# RAG LOGIC
# =====================================================

def query_rag_context(question, model, collection):
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for i in range(len(documents)):
        page = metadatas[i].get("page", "N/A")
        source = metadatas[i].get("source", "N/A")
        context_parts.append(f"\nSOURCE {i + 1}\nPage: {page}\nSource: {source}\n\n{documents[i]}\n")
        
    context = "\n".join(context_parts)
    pages = sorted(list(set(m.get("page", "N/A") for m in metadatas)))
    return context, pages

def stream_ollama_response(question, context):
    prompt = f"""
You are a medical information assistant.
Answer the user's question using ONLY the SOURCE CONTEXT.

RULES:
1. Use only information explicitly supported by the source.
2. Do not use outside medical knowledge.
3. Answer directly and concisely.
4. Do NOT provide page numbers in your answer text.

USER QUESTION: {question}
SOURCE CONTEXT: {context}
"""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']

# =====================================================
# LAYOUT: 3 COLUMNS
# =====================================================

left_col, center_col, right_col = st.columns([1.1, 2.3, 1.1])

# -----------------------------------------------------
# LEFT COLUMN: Protocols & Suggested Topics
# -----------------------------------------------------
with left_col:
    st.markdown('<div class="panel-header">🚨 Quick Protocols</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="emergency-card">
            <div class="emergency-title">CPR Rate (Adult)</div>
            <div class="emergency-desc">100–120 chest compressions per min. 2 inches deep.</div>
        </div>
        <div class="info-card">
            <div class="info-title">Severe Bleeding</div>
            <div class="info-desc">Apply firm, direct pressure. Hold continuous elevation.</div>
        </div>
        <div class="emergency-card">
            <div class="emergency-title">Choking (Adult)</div>
            <div class="emergency-desc">Give 5 back blows followed by 5 abdominal thrusts.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel-header" style="margin-top: 1rem;">💡 Common Questions</div>', unsafe_allow_html=True)
    
    suggested_query = None
    if st.button("How to treat second-degree burns?"):
        suggested_query = "How to treat second-degree burns?"
    if st.button("What are the steps for adult CPR?"):
        suggested_query = "What are the steps for adult CPR?"
    if st.button("How to treat a suspected fracture?"):
        suggested_query = "How to treat a suspected fracture?"
    if st.button("Signs of heat stroke vs heat exhaustion?"):
        suggested_query = "Signs of heat stroke vs heat exhaustion?"

# -----------------------------------------------------
# CENTER COLUMN: Primary Chat Interface
# -----------------------------------------------------
with center_col:
    if "history" not in st.session_state:
        st.session_state.history = []

    header_left, header_right = st.columns([4, 1])
    with header_right:
        if st.button("Clear Chat"):
            st.session_state.history = []
            st.rerun()

    if not st.session_state.history:
        st.markdown(
            """
            <div class="welcome-card">
                👋 <strong>Welcome!</strong> Ask any question regarding first aid or CPR.
                Responses are directly verified against reference guide pages.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Chat History
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "pages" in msg and msg["pages"]:
                formatted_pages = ", ".join(map(str, msg["pages"]))
                st.markdown(f'<span class="pages-pill">📄 Pages referenced: {formatted_pages}</span>', unsafe_allow_html=True)

    # Input handling (either from chat box or left sidebar suggestion buttons)
    user_input = st.chat_input("Ask a first aid question...")
    question = user_input or suggested_query

    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                context, pages = query_rag_context(question, model, collection)
                response_text = st.write_stream(stream_ollama_response(question, context))
                
                formatted_pages = ", ".join(map(str, pages))
                st.markdown(f'<span class="pages-pill">📄 Pages referenced: {formatted_pages}</span>', unsafe_allow_html=True)
                
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": response_text, 
                    "pages": pages
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")

# -----------------------------------------------------
# RIGHT COLUMN: Emergency Response Checklist & Disclaimer
# -----------------------------------------------------
with right_col:
    st.markdown('<div class="panel-header">📋 Emergency Checklist</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
         <div class="checklist-box">
                            <strong> PS. I was created by: SAWSAN, ZAINAB, SARA, AHMED, AND YOUMNA</strong>
                            
                        
        <div class="checklist-box">
            <strong>1. Call Emergency Services</strong><br/>
            Dial 911 / 112 immediately if the situation is life-threatening.
        </div>
        <div class="checklist-box">
            <strong>2. Check Scene Safety</strong><br/>
            Ensure you and the victim are safe from traffic, fire, or hazard.
        </div>
        <div class="checklist-box">
            <strong>3. Check Responsiveness</strong><br/>
            Tap shoulders and shout. Check for breathing or movement.
        </div>
        <div class="checklist-box">
            <strong>4. Provide First Aid</strong><br/>
            Follow guide instructions until professional medics arrive.
        </div>
        <br/>
        <div class="disclaimer-box">
            ⚠️ <strong>Disclaimer:</strong> This assistant provides information strictly sourced from reference materials. It is not a substitute for professional medical advice or emergency care.
        </div>
        <br/>
       
        """,
        unsafe_allow_html=True,
    )