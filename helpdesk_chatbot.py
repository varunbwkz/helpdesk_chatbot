"""
My awesome Freshdesk Helpdesk Chatbot! 🤖
This is the main app that users interact with.
It uses our RAG helper to answer questions about Freshdesk articles.
Run it with: streamlit run helpdesk_chatbot.py
"""
import streamlit as st
import chatbot_rag_helper as rag_helper  # Import our helper module with all the RAG stuff
import time
import traceback

# --- Setting up the Streamlit page ---
st.set_page_config(page_title="Freshdesk Help Assistant", layout="wide")
st.title("🤖 Freshdesk Help Assistant")
st.caption(f"Powered by OpenAI & ChromaDB | Using source articles from Freshdesk")

# Configuring logging so we can track what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Sidebar with options and info ---
with st.sidebar:
    st.header("Options & Info")

    # Button to clear chat history
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the Freshdesk articles provided."}]
        st.success("Chat history cleared!")
        st.rerun()

    st.divider()

    # Show some stats about our knowledge base
    st.header("Knowledge Base Info")
    kb_loaded = st.session_state.get("vector_store_loaded", False)  # Check if we've loaded the KB
    if kb_loaded and rag_helper.collection:
        try:
            article_count = len(rag_helper.parse_url_file())  # Count how many articles we have
            chunk_count = rag_helper.collection.count()  # Count chunks in the DB
        except Exception as e:
            logging.error(f"Error getting KB info: {e}")
            article_count = "Error"
            chunk_count = "Error"
    else:
        article_count = "N/A"
        chunk_count = "N/A"

    st.write(f"Source Articles Parsed: {article_count}")
    st.write(f"Chunks in DB: {chunk_count}")

    # Button to refresh the knowledge base
    if st.button("🔄 Refresh Knowledge-Base"):
        # Make sure everything is ready before trying to reload
        if rag_helper.collection and rag_helper.openai_client:
            with st.spinner("Reloading knowledge base... This may take significant time and API calls."):
                success = rag_helper.load_and_embed_data(force_reload=True)
                if success:
                     st.session_state.vector_store_loaded = True
                     st.success("Knowledge base reloaded successfully!")
                     st.rerun()
                else:
                     # If something goes wrong
                     st.error("Failed to reload knowledge base. Check logs.")
        else:
            st.warning("Cannot reload - system not initialized correctly.")


# --- Set up the chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the Freshdesk articles provided."}]

# --- Load the vector store when the app starts ---
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False

# Try to load the vector store if it's not already loaded
if not st.session_state.vector_store_loaded:
    logging.info("Attempting to initialize/load vector store...")
    if rag_helper.collection and rag_helper.openai_client:
        with st.spinner("Initializing knowledge base... Please wait."):
             # Load existing data without forcing a reload
             success = rag_helper.load_and_embed_data(force_reload=False)
             if success:
                 st.session_state.vector_store_loaded = True
                 logging.info("Knowledge base initialization/load successful.")
             else:
                  st.error("Failed to initialize knowledge base during startup. Check logs.")
                  logging.error("Knowledge base initialization failed during startup.")
    else:
         # Something went wrong with initialization
         st.error("Core components (ChromaDB/OpenAI Client) failed to initialize. Cannot load knowledge base.")
         logging.error("Core components not available for knowledge base loading.")


# --- The function that generates answers to user questions ---
def get_rag_response(query):
    # Make sure we have what we need before trying to answer
    if not rag_helper.collection or not rag_helper.openai_client:
         logging.warning("Attempted RAG query but collection or OpenAI client is not available.")
         return "Sorry, the knowledge base is not available right now."

    try:
        logging.info(f"RAG Query Received: '{query}'")
        # 1. Find relevant chunks of text from our knowledge base
        results = rag_helper.collection.query(
            query_texts=[query],
            n_results=3,  # Get 3 most relevant chunks
            include=['documents', 'metadatas']
        )
        context_chunks = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        # If we couldn't find anything relevant
        if not context_chunks:
            logging.info("No relevant documents found in ChromaDB for query.")
            return "I couldn't find information related to your question in the available articles. Could you please try rephrasing or asking about a different topic covered in the help guides (like uploading, downloading, sharing)?"

        # Extract source information so we can link to the original articles
        sources_info = set()
        if metadatas:
             for meta in metadatas:
                  if isinstance(meta, dict):
                     url = meta.get('source_url')
                     name = meta.get('article_name')
                     if url and name:
                         sources_info.add((name, url))
                  else:
                       logging.warning(f"Encountered non-dict metadata item: {meta}")

        # 2. Join all the chunks into one string for context
        context_string = "\n\n---\n\n".join(context_chunks)
        logging.info(f"Found {len(context_chunks)} relevant chunks. Using as context.")
        if sources_info:
             logging.info(f"Source articles identified: {', '.join([f'{n} ({u})' for n, u in sources_info])}")

        # 3. Create a prompt for the AI model
        # This tells the AI how to answer the question
        system_prompt = """You are a helpful assistant answering questions based *only* on the provided context from Freshdesk articles about a Digital Asset Management system. User queries may sometimes be vague or use different terminology than the articles.

        **Your goal is to provide the most helpful answer possible *using information found within the context*.**

        Follow these steps:
        1.  Carefully read the user's question and the provided context.
        2.  **Determine Relevance:** Assess if the context contains information directly related to the user's question, even if the wording differs. For example, if the user asks about "storing files," context about "uploading files" is highly relevant.
        3.  **Synthesize Answer:**
            *   If the context directly answers the query, provide a concise answer based **only** on the information in the context.
            *   If the context doesn't *directly* answer but contains **relevant information** (like the 'storing' vs 'uploading' example), synthesize an answer that explains the relevant process described in the context. State clearly how this relates to the user's query if possible (e.g., "To store files in the system, you need to upload them. Here's how the articles describe the upload process: ...").
            *   If the context is truly **irrelevant** or insufficient to answer the user's query, state clearly that the available articles do not seem to cover that specific topic. Do NOT apologize or ask the user for context. Suggest they rephrase or ask about known topics if appropriate.
        4.  **Be Factual:** Base your entire answer **strictly** on the provided context. Do not add information, steps, or assumptions not present in the text.
        5.  **Ignore Metadata:** Do not mention source URLs or article names in your response body; this will be added separately. Focus only on the textual answer.
        """

        user_prompt = f"""Based *only* on the following context, please answer the user's question according to the steps outlined in your instructions.

        Context from Freshdesk articles:
        --- START CONTEXT ---
        {context_string}
        --- END CONTEXT ---

        User's Question: {query}

        Answer:"""  # The "Answer:" helps prime the model to start generating

        logging.info("Sending refined request to OpenAI API for response generation...")
        # Using GPT-4o-mini which is pretty good but not too expensive
        completion = rag_helper.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=350  # Limit response length
        )
        response = completion.choices[0].message.content.strip()
        logging.info("Received response from OpenAI.")

        # Check if the response is unhelpful
        if not response or "cannot answer based on the provided context" in response.lower() or "provide the context" in response.lower():
             logging.warning(f"LLM gave a potentially unhelpful refusal/generic response for query: '{query}'. Context chunks might have been insufficient or truly irrelevant.")

        # Add source links to the response
        if sources_info:
             source_links_markdown = []
             for name, url in sorted(list(sources_info)):
                  source_links_markdown.append(f"[{name}]({url})")
             # Add sources at the end of the response
             if response:
                 response += f"\n\n---\nSource(s): {', '.join(source_links_markdown)}"
             else:  # Handle empty response case
                 response = f"Source(s): {', '.join(source_links_markdown)}"

             logging.info("Appended sources to response.")

        # Final check for empty response
        if not response:
            logging.error(f"Empty response generated after processing query: '{query}'.")
            return "I encountered an issue processing your request. Please try again."

        return response

    except Exception as e:
        # Log the error for debugging
        logging.error(f"Error during RAG query for '{query}': {e}", exc_info=True)
        return f"An error occurred while processing your request. Please check the logs or try again later."

# --- Display the chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle user input ---
if prompt := st.chat_input("Ask about downloading, uploading, sharing assets etc."):
    # Make sure the system is ready before processing questions
    if not st.session_state.vector_store_loaded:
         st.warning("Knowledge base not ready. Please wait or check errors before asking questions.")
    else:
        # Add the user's message to the chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            # Get the response from our RAG system
            assistant_response = get_rag_response(prompt)
            message_placeholder.markdown(assistant_response)

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})