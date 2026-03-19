import streamlit as st
import requests
import time
import os

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Savvy",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── API Configuration ───────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8081")

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(160deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Custom header ── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #8892b0;
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: 0;
    }

    /* ── Divider ── */
    .subtle-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea44, transparent);
        margin: 0.5rem auto 1.5rem auto;
        max-width: 300px;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
        backdrop-filter: blur(10px);
    }
    .stChatMessage p {
        line-height: 1.75 !important;
        letter-spacing: 0.01em;
        margin-bottom: 0.6rem !important;
    }

    /* ── User message accent ── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Chat input ── */
    .stChatInput {
        border-color: rgba(102, 126, 234, 0.3) !important;
    }
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 14px !important;
    }
    .stChatInput textarea {
        color: #e6e6e6 !important;
    }

    /* ── Source badges ── */
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 6px;
        letter-spacing: 0.3px;
    }
    .badge-vector {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .badge-web {
        background: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .badge-youtube {
        background: rgba(237, 100, 100, 0.15);
        color: #ed6464;
        border: 1px solid rgba(237, 100, 100, 0.3);
    }

    /* ── Welcome card ── */
    .welcome-card {
        text-align: center;
        padding: 3rem 2rem;
        margin: 2rem auto;
        max-width: 500px;
    }
    .welcome-card .emoji {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .welcome-card h3 {
        color: #ccd6f6;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .welcome-card p {
        color: #8892b0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Suggestion chips ── */
    .suggestion-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-top: 1.5rem;
    }
    .suggestion-chip {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: #a8b2d1;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        cursor: default;
        transition: all 0.2s ease;
    }

    /* ── Spinner styling ── */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🎬 Movie Savvy</h1>
    <p>Your AI-powered entertainment expert</p>
</div>
<div class="subtle-divider"></div>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helper: Query the API ───────────────────────────────────────────────
def query_api(user_query: str) -> dict:
    """Send a query to the FastAPI backend and return the result."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": user_query},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"response": "⚠️ Cannot reach the API server. Make sure the backend is running.", "content_sources_used": {}}
    except requests.exceptions.Timeout:
        return {"response": "⏳ The request timed out. Please try a simpler query.", "content_sources_used": {}}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}", "content_sources_used": {}}


# ── Helper: Render source badges ────────────────────────────────────────
def render_source_badges(sources: dict) -> str:
    """Generate HTML for source badges."""
    if not sources:
        return ""
    badges = []
    if sources.get("vector_store"):
        badges.append('<span class="source-badge badge-vector">📚 Knowledge Base</span>')
    if sources.get("web_search"):
        badges.append('<span class="source-badge badge-web">🌐 Web Search</span>')
    if sources.get("youtube"):
        badges.append('<span class="source-badge badge-youtube">▶ YouTube</span>')
    return "".join(badges)


# ── Helper: Format response text ────────────────────────────────────────
def format_response(text: str) -> str:
    """Break a dense response into readable paragraphs."""
    if not text:
        return text
    import re
    # Split on sentence endings followed by uppercase letter (new thought)
    formatted = re.sub(r'(?<=[.!?])\s{1,2}(?=[A-Z])', '\n\n', text)
    return formatted.strip()


# ── Welcome state ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="emoji">🍿</div>
        <h3>What are you watching today?</h3>
        <p>Ask me anything about movies, anime, manga, K-dramas, Bollywood, or Hollywood.
           I'll search across my knowledge base, the web, and YouTube to find the best answer.</p>
        <div class="suggestion-row">
            <span class="suggestion-chip">Explain One Piece chapter 1153</span>
            <span class="suggestion-chip">Best anime like Berserk</span>
            <span class="suggestion-chip">MCU Phase 6 movies</span>
            <span class="suggestion-chip">Top K-dramas of 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Display chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎬"):
        st.markdown(format_response(msg["content"]))
        if msg["role"] == "assistant" and msg.get("sources"):
            badge_html = render_source_badges(msg["sources"])
            if badge_html:
                st.markdown(f"<div style='margin-top:8px'>{badge_html}</div>", unsafe_allow_html=True)


# ── Chat input ──────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about movies, anime, K-dramas..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant", avatar="🎬"):
        with st.spinner("Thinking..."):
            result = query_api(user_input)

        response_text = result.get("response", "Sorry, I couldn't generate a response.")
        sources = result.get("content_sources_used", {})

        st.markdown(format_response(response_text))

        badge_html = render_source_badges(sources)
        if badge_html:
            st.markdown(f"<div style='margin-top:8px'>{badge_html}</div>", unsafe_allow_html=True)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
