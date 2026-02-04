import streamlit as st

# import config, os

# os.environ['GEMINI_API_KEY'] = config.GEMINI_API_KEY


from pro_implementation.answer import answer_question


st.set_page_config(
    page_title="Insurellm Expert Assistant",
    layout="wide",
    page_icon="🏢"
)

with st.sidebar:
    st.markdown("### 👨‍💻 Project Info")
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <a href="https://github.com/RohanTemgire/LLM-apps" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/GitHub-View_Source-181717?style=for-the-badge&logo=github&logoColor=white" alt="View on GitHub"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        "This chatbot uses a **RAG (Retrieval-Augmented Generation)** workflow to answer questions based on the Insurellm knowledge base."
    )

def format_context(context_docs):
    """Formats the retrieved documents into HTML for display."""
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    if not context_docs:
        return result + "<p>No context retrieved.</p>"
        
    for doc in context_docs:
        result += "<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
        source = doc.metadata.get('source', 'Unknown Source')
        result += f"<span style='color: #ff7800; font-weight: bold;'>Source: {source}</span><br><br>"
        result += f"<div style='font-size: 14px; color: #333;'>{doc.page_content}</div>"
        result += "</div>"
    return result


if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_context_html" not in st.session_state:
    st.session_state.retrieved_context_html = "*Retrieved context will appear here*"


st.title("🏢 Insurellm Expert Assistant")
st.markdown("Ask me anything about Insurellm!")


col1, col2 = st.columns([1, 1], gap="medium")

# --- Left Column: Chat Interface ---
with col1:
    st.subheader("💬 Conversation")
    
    # Create a container for chat history to keep it organized
    chat_container = st.container(height=600, border=True)
    
    with chat_container:
        # Display existing chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# --- Right Column: Retrieved Context ---
with col2:
    st.subheader("📚 Retrieved Context")
    
    context_container = st.container(height=600, border=True)
    
    with context_container:
        st.markdown(st.session_state.retrieved_context_html, unsafe_allow_html=True)

# --- Chat Logic (Input Handling) ---
# Note: st.chat_input is always fixed at the bottom of the screen or container
if prompt := st.chat_input("Ask anything about Insurellm..."):
    
    # 1. Update UI with User Message immediately
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # 2. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Process the logic (Call Backend)
    # Prepare history for the backend (excluding the current prompt we just added)
    prior_history = st.session_state.messages[:-1]
    
    with st.spinner("Thinking..."):
        answer, context_docs = answer_question(prompt, prior_history)
    
    # 4. Update Context in Session State
    st.session_state.retrieved_context_html = format_context(context_docs)
    
    # 5. Update UI with Assistant Response
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(answer)
            
    # 6. Add Assistant Message to History
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # 7. Rerun to ensure the Context Column (Right side) updates immediately
    st.rerun()