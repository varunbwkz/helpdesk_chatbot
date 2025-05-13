"""
Freshdesk Helpdesk Chatbot
A Streamlit-based chatbot for answering questions about Freshdesk articles.
Run with: streamlit run helpdesk_chatbot.py
"""
import streamlit as st
import logging
from chatbot_rag_helper import FreshdeskRAG
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("chatbot.log")]
)
logger = logging.getLogger("chatbot")

# Custom CSS for better styling
def apply_custom_css():
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e6f3ff;
    }
    .chat-header {
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #555;
    }
    h3 {
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    .source-link {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #f1f3f4;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    .main-content {
        max-width: 900px;
        margin: 0 auto;
    }
    /* Make sure markdown renders well */
    .stMarkdown ul, .stMarkdown ol {
        padding-left: 2rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown blockquote {
        border-left: 2px solid #ccc;
        padding-left: 1rem;
        color: #555;
    }
    .stMarkdown code {
        padding: 0.2rem 0.4rem;
        background-color: #f8f8f8;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Freshdesk Help Assistant",
    page_icon="🤖",
    layout="wide"
)

# Apply custom CSS
apply_custom_css()

# Create main content container
main_container = st.container()
with main_container:
    st.title("🤖 Freshdesk Help Assistant")
    st.caption("Powered by OpenAI & ChromaDB | Using Freshdesk articles as knowledge source")

# Initialize RAG helper (cached to prevent reinitialization)
@st.cache_resource
def get_rag_helper():
    """Initialize and return the RAG helper (cached)"""
    try:
        logger.info("Initializing RAG helper")
        return FreshdeskRAG()
    except Exception as e:
        logger.error(f"Failed to initialize RAG helper: {e}", exc_info=True)
        return None

rag_helper = get_rag_helper()

# Sidebar with options and stats
with st.sidebar:
    st.header("🛠️ Options & Info")
    
    # Chat history control
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your Freshdesk Help Assistant. Ask me anything about uploading, downloading, or managing your assets."}]
        st.success("Chat history cleared!")
        st.rerun()
        
    st.divider()
    
    # Knowledge base stats
    st.header("📚 Knowledge Base")
    kb_loaded = st.session_state.get("kb_loaded", False)
    
    # Visual indicator of knowledge base status
    if kb_loaded:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Loading... ⏳")
    
    if kb_loaded and rag_helper and hasattr(rag_helper, 'collection'):
        try:
            article_count = len(rag_helper.parse_url_file())
            chunk_count = rag_helper.collection.count()
                
        except Exception as e:
            logger.error(f"Error getting KB info: {e}")
            st.error("Error loading stats")
    
    # Knowledge base refresh button
    if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
        if rag_helper:
            with st.spinner("Reloading knowledge base... This may take a moment."):
                try:
                    success = rag_helper.load_knowledge_base(force_reload=True)
                    if success:
                        st.session_state.kb_loaded = True
                        st.success("Knowledge base refreshed!")
                        st.rerun()
                    else:
                        st.error("Failed to refresh knowledge base. Check logs.")
                except Exception as e:
                    logger.error(f"Error during refresh: {e}", exc_info=True)
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("System not initialized correctly.")
            
    # Add helpful info section
    st.divider()
    st.header("💡 Sample Questions")
    st.markdown("""
    - How do I upload assets?
    - What's the process for downloading files?
    - How can I share assets with colleagues?
    - Can I move assets between folders?
    - How do I tag my assets?
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your Freshdesk Help Assistant. Ask me anything about uploading, downloading, or managing your assets."}]

# Initialize knowledge base
if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False

# Load knowledge base if not already loaded
if not st.session_state.kb_loaded and rag_helper:
    with st.spinner("Initializing knowledge base... Please wait."):
        try:
            success = rag_helper.load_knowledge_base(force_reload=False)
            if success:
                st.session_state.kb_loaded = True
                logger.info("Knowledge base loaded successfully")
            else:
                st.error("Failed to initialize knowledge base. Check logs.")
        except Exception as e:
            st.error(f"Error initializing knowledge base: {str(e)}")
            logger.error(f"Knowledge base initialization error: {e}", exc_info=True)
elif not rag_helper:
    st.error("Failed to initialize. Check your OpenAI API key and configuration.")

# Custom chat message container
def display_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-header">You</div>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-header">Assistant</div>
            {content}
        </div>
        """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about downloading, uploading, sharing assets etc."):
    # Check if system is ready
    if not st.session_state.kb_loaded:
        st.warning("Knowledge base not ready. Please wait before asking questions.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Show typing animation
                typing_text = "Thinking"
                for _ in range(3):
                    for dots in [".", "..", "..."]:
                        time.sleep(0.2)
                        message_placeholder.markdown(f"{typing_text}{dots}")
                
                # Get response from RAG system
                response = rag_helper.get_response(prompt)
                message_placeholder.markdown(response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"### ⚠️ Error\n\nI encountered a problem while generating a response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                message_placeholder.markdown(error_msg)