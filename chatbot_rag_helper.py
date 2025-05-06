# """
# The brains behind my helpdesk chatbot!
# This is where all the magic happens for finding and remembering stuff from our Freshdesk articles.
# """
# import os
# import requests
# import time
# from bs4 import BeautifulSoup  # BeautifulSoup is awesome for scraping HTML
# from openai import OpenAI
# import chromadb  # My vector database for storing embeddings
# from chromadb.utils import embedding_functions
# from dotenv import load_dotenv

# # --- Setting things up ---
# load_dotenv()  # Grabs our API key from the .env file - super handy!
# API_KEY = os.getenv("OPENAI_API_KEY")
# # I'm using the smaller embedding model to save some cash 💰
# EMBEDDING_MODEL = "text-embedding-3-small"
# # Where we'll save our database stuff
# CHROMA_PERSIST_DIR = "freshdesk_chroma_db"
# # Name for our collection in the database
# CHROMA_COLLECTION_NAME = "freshdesk_articles"

# # This is the CSS selector that finds the main content in Freshdesk articles
# # Had to look at the page source to figure this one out!
# FRESHDESK_CONTENT_SELECTOR = "div.article-body"  # Works on most Freshdesk sites

# # How we split up the text into chunks - played around with these values until they worked well
# CHUNK_SIZE = 1000  # Characters per chunk
# CHUNK_OVERLAP = 150  # Overlap between chunks so we don't lose context


# # --- Setting up OpenAI ---
# openai_client = None
# if API_KEY:
#     try:
#         openai_client = OpenAI(api_key=API_KEY)
#         print("OpenAI client initialized for RAG helper.")
#         # This lets us use OpenAI to create embeddings (vector representations of text)
#         openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#             api_key=API_KEY,
#             model_name=EMBEDDING_MODEL
#         )
#     except Exception as e:
#         print(f"Error initializing OpenAI client or EF: {e}")
#         openai_client = None
#         openai_ef = None
# else:
#     print("Warning: OPENAI_API_KEY not found. RAG helper setup may fail.")
#     openai_ef = None

# # --- Setting up ChromaDB ---
# try:
#     # This creates/opens our vector database
#     chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
#     # Now we create a collection in the database (or open an existing one)
#     if openai_ef:
#          collection = chroma_client.get_or_create_collection(
#              name=CHROMA_COLLECTION_NAME,
#              embedding_function=openai_ef
#          )
#          print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded/created in '{CHROMA_PERSIST_DIR}'.")
#     else:
#         collection = None
#         print("ChromaDB collection could not be initialized due to missing OpenAI Embedding Function.")

# except Exception as e:
#     print(f"Error initializing ChromaDB client: {e}")
#     chroma_client = None
#     collection = None

# # --- The actual functions that do all the work ---

# def parse_url_file(filepath="knowledge_urls.txt"):
#     """Reads our URL file and returns a list of (name, url) tuples.
#     The format in the file should be: Article Name | URL"""
#     urls_to_process = []
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 # Skip empty lines and comments
#                 if not line or line.startswith('#'):
#                     continue
#                 parts = line.split('|', 1)  # Split on the first | only
#                 if len(parts) == 2:
#                     name = parts[0].strip()
#                     url = parts[1].strip()
#                     if name and url:
#                         urls_to_process.append((name, url))
#                     else:
#                          print(f"Warning: Skipping invalid line: '{line}'")
#                 else:
#                      print(f"Warning: Skipping line with incorrect format: '{line}'")
#         print(f"Found {len(urls_to_process)} URLs to process from '{filepath}'.")
#         return urls_to_process
#     except FileNotFoundError:
#         print(f"Error: URL file not found at '{filepath}'")
#         return []
#     except Exception as e:
#          print(f"Error reading URL file '{filepath}': {e}")
#          return []

# def fetch_and_extract_text(url):
#     """Fetches content from a URL and extracts the main text.
#     This is where we pull the content from Freshdesk articles."""
#     print(f"Fetching: {url}...")
#     try:
#         # Pretending to be a browser so websites don't block us...
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
#         response = requests.get(url, headers=headers, timeout=15)  # 15 sec timeout should be plenty
#         response.raise_for_status()  # This raises an error if we get a 404 or other bad response

#         # Parse the HTML with BeautifulSoup
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Find the main content container 
#         content_div = soup.select_one(FRESHDESK_CONTENT_SELECTOR)

#         if content_div:
#             # Get text and clean it up a bit
#             text = content_div.get_text(separator='\n', strip=True)
#             print(f"Successfully extracted ~{len(text)} chars from {url}")
#             return text
#         else:
#             print(f"Warning: Content selector '{FRESHDESK_CONTENT_SELECTOR}' not found at {url}. Trying body.")
#             # Plan B - just grab everything in the body tag (messy but better than nothing)
#             body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
#             if body_text:
#                  print("Using fallback body text (may include headers/footers).")
#                  return body_text
#             else:
#                  print(f"Error: Could not find content selector or body text at {url}")
#                  return None
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching URL {url}: {e}")
#         return None
#     except Exception as e:
#         print(f"Error processing URL {url}: {e}")
#         return None

# def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
#     """Splits text into overlapping chunks.
#     We need to do this because the AI models have token limits."""
#     if not text:
#         return []
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - chunk_overlap  # Move forward but with some overlap
#         if start >= len(text):  # Stop if we've reached the end
#              break
#         # Just a safety check - if our math somehow goes wrong
#         if start < 0: start = end  # Reset if calculation yields negative index
#     return chunks


# def load_and_embed_data(force_reload=False):
#     """The main function that loads URLs, processes content, and stores embeddings in ChromaDB.
#     This is where the real work happens!"""
#     global collection  # Need to use the global var if we recreate it
#     if not chroma_client or not collection or not openai_ef:
#         print("Error: ChromaDB or OpenAI Embedding Function not initialized. Cannot load data.")
#         return False

#     # Check if we've already loaded data
#     existing_count = collection.count()
#     print(f"Existing documents in collection: {existing_count}")

#     if existing_count > 0 and not force_reload:
#         print("Collection already contains data. Skipping loading process. Use force_reload=True to override.")
#         return True

#     # If we're force reloading, we need to clear the existing collection
#     if force_reload and existing_count > 0:
#          print("Force reload requested. Clearing existing collection...")
#          try:
#              chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
#              print(f"Old collection '{CHROMA_COLLECTION_NAME}' deleted.")
#              collection = chroma_client.get_or_create_collection(
#                  name=CHROMA_COLLECTION_NAME,
#                  embedding_function=openai_ef
#              )
#              print("New collection created.")
#          except Exception as e:
#              print(f"Error clearing collection: {e}. Aborting reload.")
#              return False

#     # Load articles from our URL file
#     urls_data = parse_url_file()
#     if not urls_data:
#         return False

#     # This is where we'll store all our processed text chunks
#     all_chunks = []
#     all_metadatas = []
#     all_ids = []
#     chunk_counter = 0

#     # Process each article
#     for name, url in urls_data:
#         article_text = fetch_and_extract_text(url)
#         if article_text:
#             chunks = split_text_into_chunks(article_text)
#             for i, chunk in enumerate(chunks):
#                 chunk_id = f"{url}_chunk_{i}"
#                 all_chunks.append(chunk)
#                 all_metadatas.append({"source_url": url, "article_name": name, "chunk_index": i})
#                 all_ids.append(chunk_id)
#                 chunk_counter += 1
#         else:
#             print(f"Skipping article due to extraction failure: {name} ({url})")
#         time.sleep(0.2)  # Small delay to be nice to the servers

#     # Make sure we actually got some text to work with
#     if not all_chunks:
#         print("No text chunks extracted from any URL. Nothing to embed.")
#         return False

#     print(f"\nGenerated {len(all_chunks)} text chunks from {len(urls_data)} articles.")
#     print("Adding chunks to ChromaDB (this may take time depending on the number of chunks)...")

#     try:
#         # Process in batches for better performance
#         batch_size = 100  # Adjust if needed
#         for i in range(0, len(all_chunks), batch_size):
#             batch_chunks = all_chunks[i:i + batch_size]
#             batch_metadatas = all_metadatas[i:i + batch_size]
#             batch_ids = all_ids[i:i + batch_size]

#             print(f"Adding batch {i//batch_size + 1} ({len(batch_ids)} items)...")
#             collection.add(
#                 documents=batch_chunks,
#                 metadatas=batch_metadatas,
#                 ids=batch_ids
#             )
#             print(f"Batch {i//batch_size + 1} added.")
#         print("\nAll data successfully added to ChromaDB collection.")
#         return True
#     except Exception as e:
#         print(f"\nError adding data to ChromaDB: {e}")
#         return False


# # This runs when we execute this file directly (for testing)
# if __name__ == "__main__":
#     print("Running RAG Helper setup...")
#     # Set force_reload=True to re-process all URLs even if we already have data
#     # Set force_reload=False to only load if the collection is empty (saves time and API calls)
#     success = load_and_embed_data(force_reload=False)
#     if success:
#         print("\nData loading process complete.")
#         # Uncomment to test a query
#         # results = collection.query(query_texts=["How do I upload?"], n_results=2)
#         # print("\nTest Query Results:")
#         # print(results)
#     else:
#         print("\nData loading process failed.")






"""
RAG Helper for Freshdesk Helpdesk Chatbot
Handles knowledge base creation, embedding, and retrieval from Freshdesk articles.
"""
import os
import logging
import time
from typing import List, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("rag_helper.log")]
)
logger = logging.getLogger("rag_helper")

class FreshdeskRAG:
    """Helper class for RAG operations with Freshdesk articles"""
    
    def __init__(
        self,
        persist_dir: str = "freshdesk_chroma_db",
        collection_name: str = "freshdesk_articles",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        content_selector: str = "div.article-body"
    ):
        """Initialize the RAG helper with configuration parameters"""
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("No OpenAI API key found in environment")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Store configuration
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.content_selector = content_selector
        
        # Initialize components
        self._setup_openai()
        self._setup_chromadb()
    
    def _setup_openai(self) -> None:
        """Initialize OpenAI client and embedding function"""
        try:
            self.openai_client = OpenAI(api_key=self.api_key)
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.embedding_model
            )
            logger.info(f"OpenAI initialized with model {self.embedding_model}")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            raise RuntimeError(f"OpenAI setup error: {e}")
    
    def _setup_chromadb(self) -> None:
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.openai_ef
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise RuntimeError(f"ChromaDB setup error: {e}")
    
    def parse_url_file(self, filepath: str = "knowledge_urls.txt") -> List[Tuple[str, str]]:
        """Parse URL file and return list of (name, url) tuples"""
        urls = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        name, url = parts[0].strip(), parts[1].strip()
                        if name and url:
                            urls.append((name, url))
                        else:
                            logger.warning(f"Invalid line: '{line}'")
                    else:
                        logger.warning(f"Incorrect format: '{line}'")
            
            logger.info(f"Found {len(urls)} URLs in '{filepath}'")
            return urls
        except FileNotFoundError:
            logger.error(f"URL file not found: '{filepath}'")
            return []
        except Exception as e:
            logger.error(f"Error reading URL file: {e}")
            return []
    
    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract text from a Freshdesk article URL"""
        logger.info(f"Fetching: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.select_one(self.content_selector)
            
            if content_div:
                text = content_div.get_text(separator='\n', strip=True)
                logger.info(f"Extracted {len(text)} chars from {url}")
                return text
            else:
                # Fallback to body content
                body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
                if body_text:
                    logger.info(f"Using fallback body text for {url}")
                    return body_text
                
                logger.error(f"No content found at {url}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for processing"""
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    def load_knowledge_base(self, force_reload: bool = False) -> bool:
        """Load articles into the vector database"""
        # Check if data already exists
        existing_count = self.collection.count()
        logger.info(f"Existing documents: {existing_count}")
        
        if existing_count > 0 and not force_reload:
            logger.info("Knowledge base already loaded. Use force_reload=True to refresh.")
            return True
            
        # Clear existing collection if force reloading
        if force_reload and existing_count > 0:
            logger.info("Clearing existing collection...")
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.openai_ef
                )
            except Exception as e:
                logger.error(f"Failed to clear collection: {e}")
                return False
                
        # Load articles from URL file
        urls_data = self.parse_url_file()
        if not urls_data:
            logger.error("No valid URLs found")
            return False
            
        # Process articles in batches
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for name, url in urls_data:
            article_text = self.fetch_article_content(url)
            if article_text:
                chunks = self.split_into_chunks(article_text)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{url}_chunk_{i}"
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source_url": url,
                        "article_name": name,
                        "chunk_index": i
                    })
                    all_ids.append(chunk_id)
            else:
                logger.warning(f"Skipping article: {name} ({url})")
                
            time.sleep(0.2)  # Be nice to the server
            
        if not all_chunks:
            logger.error("No text extracted from any URL")
            return False
            
        logger.info(f"Generated {len(all_chunks)} chunks from {len(urls_data)} articles")
        
        # Add chunks to database in batches
        try:
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                end_idx = min(i + batch_size, len(all_chunks))
                logger.info(f"Adding batch {i//batch_size + 1} ({end_idx - i} items)...")
                
                self.collection.add(
                    documents=all_chunks[i:end_idx],
                    metadatas=all_metadatas[i:end_idx],
                    ids=all_ids[i:end_idx]
                )
                
            logger.info("All data added to vector database")
            return True
        except Exception as e:
            logger.error(f"Error adding data to database: {e}")
            return False
    
    def get_response(self, query: str, n_results: int = 3) -> str:
        """Generate a formatted response to a user query using RAG"""
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Find relevant chunks
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            
            chunks = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            # No relevant information found
            if not chunks:
                logger.info("No relevant documents found")
                return "I couldn't find information related to your question in the available articles. Could you try rephrasing or asking about a topic covered in our help guides?"
                
            # Extract source information
            sources = []
            for meta in metadatas:
                if isinstance(meta, dict):
                    url = meta.get('source_url')
                    name = meta.get('article_name')
                    if url and name and (name, url) not in [(s[0], s[1]) for s in sources]:
                        sources.append((name, url))
                        
            # Create context for the LLM
            context = "\n\n---\n\n".join(chunks)
            
            # Create prompt for the AI
            system_prompt = """You are a helpful assistant answering questions based *only* on the provided context from Freshdesk articles about a Digital Asset Management system. 

    **Your goal is to provide the most helpful answer possible *using information found within the context*.**

    Follow these steps:
    1. Carefully read the user's question and the provided context.
    2. **Determine Relevance:** Assess if the context contains information directly related to the user's question, even if the wording differs.
    3. **Synthesize Answer:** Provide a concise answer based only on the information in the context.
    4. **Be Factual:** Do not add information, steps, or assumptions not present in the text.
    5. **Ignore Metadata:** Do not mention source URLs or article names in your response body; this will be added separately.

    Format your response with clear sections using markdown formatting:
    - Use **bold** for steps, key actions, or important notes
    - Use bullet points for lists of options or features
    - Structure longer answers with subheadings
    - Keep content concise and scannable
    """

            user_prompt = f"""Based only on the following context, please answer the user's question.

    Context from Freshdesk articles:
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    User's Question: {query}

    Answer:"""

            # Generate response with OpenAI
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Format the response with a nice visual layout
            formatted_response = f"### 📝 Answer\n\n{response}\n\n"
            
            # Add formatted source attribution
            if sources:
                formatted_response += "### 📚 Sources\n\n"
                for i, (name, url) in enumerate(sources, 1):
                    formatted_response += f"{i}. [{name}]({url})\n"
                    
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"### ⚠️ Error\n\nAn error occurred while processing your request. Please try again."