# Freshdesk Helpdesk Chatbot

A smart, AI-powered chatbot that answers questions about Freshdesk articles using RAG (Retrieval Augmented Generation).

## 📚 Table of Contents

- [What This Does](#what-this-does)
- [How It Works (Detailed Flow)](#how-it-works-detailed-flow)
- [Setup Guide](#setup-guide)
- [File Structure](#file-structure)
- [Customization Options](#customization-options)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## 🤔 What This Does

This chatbot helps users find information from Freshdesk help articles without having to search through documentation. Users can ask questions in natural language, and the chatbot provides accurate answers based on the actual content of your Freshdesk articles, with links to the original sources.

**Key Features:**
- 🔍 Intelligent search across all your help articles
- 💬 Natural language understanding of questions
- 📝 Answers based only on your actual documentation
- 🔗 Source citations with links to original articles
- 🔄 Easy knowledge base refreshing when articles change

## 🔄 How It Works (Detailed Flow)

This section breaks down exactly how the chatbot works, step by step, so you can understand what's happening behind the scenes.

### 1. Initial Setup Phase

When you first run the application:

1. **Load Environment Variables**
   - The app reads your `.env` file to get your OpenAI API key
   - This key is needed to access OpenAI's embedding and chat models

2. **Initialize Clients**
   - **OpenAI Client**: Sets up the connection to OpenAI's API
   - **ChromaDB Client**: Creates or opens a database for storing embeddings
   - **ChromaDB Collection**: Creates or opens a collection named "freshdesk_articles"

3. **Check for Existing Data**
   - The app checks if there's already data in the ChromaDB collection
   - If data exists and you're not forcing a reload, it skips the loading process
   - If no data exists, it proceeds to the data loading phase

### 2. Data Loading Phase

This happens either on first run or when you click "Refresh Knowledge Base":

1. **Read URL File**
   - Parses your `knowledge_urls.txt` file to get article names and URLs
   - Format in the file should be: `Article Name | URL`
   - Lines starting with `#` are treated as comments

2. **Web Scraping**
   - For each URL in your file:
     - Makes an HTTP request to fetch the article
     - Uses BeautifulSoup to parse the HTML
     - Extracts the main content using the CSS selector `div.article-body`
     - Cleans up the text to remove unnecessary spaces and formatting

3. **Text Chunking**
   - Splits each article into smaller chunks (about 1000 characters each)
   - Creates overlap between chunks (about 150 characters) to maintain context
   - This is necessary because AI models have token limits

4. **Creating Embeddings**
   - For each chunk of text:
     - Sends the text to OpenAI's embedding model (`text-embedding-3-small`)
     - Gets back a vector (array of numbers) that represents the semantic meaning of the text
     - This vector allows for semantic search later (finding similar meanings, not just keywords)

5. **Storing in ChromaDB**
   - For each chunk and its embedding:
     - Saves the text, embedding, and metadata (source URL, article name) to ChromaDB
     - Organizes them in batches for efficiency
     - ChromaDB handles the complex vector operations needed for similarity search

6. **Update Application State**
   - Sets `vector_store_loaded` to `True` in the Streamlit session state
   - Updates the sidebar with the number of articles and chunks loaded

### 3. User Interaction Phase

When a user asks a question:

1. **Input Processing**
   - Captures the user's question from the chat input
   - Adds the question to the chat history and displays it

2. **Retrieval (Finding Relevant Content)**
   - Converts the user's question into an embedding using the same model
   - Queries ChromaDB to find the most similar chunks to the question
   - Gets back the top 3 most relevant chunks, along with their metadata
   - This is the "Retrieval" part of RAG (Retrieval Augmented Generation)

3. **Prompt Construction**
   - Creates a carefully designed system prompt that tells the AI how to answer
   - Combines the system prompt, retrieved context chunks, and user's question
   - This forms a complete prompt to send to OpenAI
   - This is the "Augmentation" part of RAG

4. **Response Generation**
   - Sends the prompt to OpenAI's model (gpt-4o-mini)
   - The model generates a response based only on the provided context
   - This ensures answers are factual and based on your content
   - This is the "Generation" part of RAG

5. **Source Attribution**
   - Adds links to the original articles at the end of the response
   - Format: "Source(s): [Article Name](URL)"
   - This helps users find more information if needed

6. **Display and Storage**
   - Shows the response in the chat interface
   - Adds the response to the chat history for context in future questions

### 4. Knowledge Base Refreshing

When you click "Refresh Knowledge Base":

1. **Clear Existing Data**
   - Deletes the existing collection from ChromaDB
   - Creates a new collection with the same name

2. **Reload Data**
   - Repeats the entire Data Loading Phase described above
   - This allows you to update the knowledge base when articles change

## 🛠️ Setup Guide

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Freshdesk articles accessible via public URLs

### Installation Steps

1. **Clone this repository**
   ```bash
   git clone <repository-url>
   cd freshdesk-helpdesk-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file with your OpenAI API key**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Create a `knowledge_urls.txt` file with your Freshdesk articles**
   ```
   # Format: Article Name | URL
   Downloading Assets | https://brandworkz.freshdesk.com/support/solutions/articles/80000881124-downloading-assets
   Uploading Assets | https://brandworkz.freshdesk.com/support/solutions/articles/80000881255-uploading-assets
   # Add more URLs here
   ```

6. **Run the application**
   ```bash
   streamlit run helpdesk_chatbot.py
   ```

7. **Wait for the initial knowledge base to load**
   - The first time you run the app, it will scrape and embed all articles
   - This may take a few minutes depending on how many articles you have

## 📁 File Structure

- `helpdesk_chatbot.py` - The main Streamlit application
- `chatbot_rag_helper.py` - Backend functionality for RAG
- `knowledge_urls.txt` - List of article names and URLs
- `.env` - Environment variables (API key)
- `freshdesk_chroma_db/` - Directory where ChromaDB stores data
- `requirements.txt` - Required Python packages

## ⚙️ Customization Options

You can customize several aspects of the chatbot:

### In `chatbot_rag_helper.py`:

- **EMBEDDING_MODEL**: Change the OpenAI embedding model
  ```python
  EMBEDDING_MODEL = "text-embedding-3-small"  # Default
  ```

- **CHUNK_SIZE**: Adjust the size of text chunks
  ```python
  CHUNK_SIZE = 1000  # Characters per chunk
  ```

- **CHUNK_OVERLAP**: Adjust the overlap between chunks
  ```python
  CHUNK_OVERLAP = 150  # Characters overlap
  ```

- **FRESHDESK_CONTENT_SELECTOR**: Change the CSS selector if your Freshdesk layout is different
  ```python
  FRESHDESK_CONTENT_SELECTOR = "div.article-body"
  ```

### In `helpdesk_chatbot.py`:

- **AI Model**: Change the OpenAI model used for generating responses
  ```python
  model="gpt-4o-mini"  # You can use "gpt-4o" for better quality but higher cost
  ```

- **Number of Results**: Adjust how many chunks are retrieved
  ```python
  n_results=3  # Increase for more comprehensive answers
  ```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **"OpenAI API key not found"**
   - Make sure you've created a `.env` file with your API key
   - Check that the key is correctly formatted

2. **"Failed to initialize knowledge base"**
   - Check your internet connection
   - Verify that your Freshdesk URLs are accessible
   - Look at logs for specific errors

3. **No relevant answers**
   - Try refreshing the knowledge base
   - Check that your articles contain the information being asked about
   - Try rephrasing the question

4. **ChromaDB errors**
   - Delete the `freshdesk_chroma_db` directory and let it rebuild
   - Make sure you have sufficient disk space

5. **Memory errors**
   - Reduce the number of articles or increase your system's memory
   - Adjust the chunk size to be smaller

## ❓ FAQ

**Q: How much does it cost to run this chatbot?**
A: The cost depends on your usage. The chatbot uses OpenAI's API, which charges based on tokens processed. Using the smaller embedding model and gpt-4o-mini helps keep costs reasonable.

**Q: Can it answer questions about topics not in the articles?**
A: No, the chatbot is designed to only answer based on the content in your Freshdesk articles. This ensures factual responses but means it can't answer questions outside that scope.

**Q: How often should I refresh the knowledge base?**
A: Refresh whenever you add new articles or update existing ones. The refresh process re-scrapes all articles to capture any changes.

**Q: Can I use this for other documentation besides Freshdesk?**
A: Yes, but you'd need to modify the scraping logic in `fetch_and_extract_text()` to work with the HTML structure of your documentation.

**Q: How can I make the answers more detailed?**
A: You can increase the `n_results` parameter to retrieve more context chunks, or adjust the system prompt to encourage more detailed responses.

**Q: Does this work offline?**
A: No, this requires an internet connection to access the OpenAI API.

---

## 📊 Technical Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Freshdesk      │     │  ChromaDB       │     │  OpenAI API     │
│  Articles       │     │  Vector Store   │     │                 │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                   chatbot_rag_helper.py                          │
│                                                                  │
│  - Scrapes articles                                              │
│  - Creates embeddings                                            │
│  - Stores in ChromaDB                                            │
│  - Retrieves relevant context                                    │
│                                                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                   helpdesk_chatbot.py                            │
│                                                                  │
│  - Provides the user interface                                   │
│  - Handles user queries                                          │
│  - Manages chat history                                          │
│  - Displays responses                                            │
│                                                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│                   User Interface (Streamlit)                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```