"""
The brains behind my helpdesk chatbot!
This is where all the magic happens for finding and remembering stuff from our Freshdesk articles.
"""
import os
import requests
import time
from bs4 import BeautifulSoup  # BeautifulSoup is awesome for scraping HTML
from openai import OpenAI
import chromadb  # My vector database for storing embeddings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# --- Setting things up ---
load_dotenv()  # Grabs our API key from the .env file - super handy!
API_KEY = os.getenv("OPENAI_API_KEY")
# I'm using the smaller embedding model to save some cash 💰
EMBEDDING_MODEL = "text-embedding-3-small"
# Where we'll save our database stuff
CHROMA_PERSIST_DIR = "freshdesk_chroma_db"
# Name for our collection in the database
CHROMA_COLLECTION_NAME = "freshdesk_articles"

# This is the CSS selector that finds the main content in Freshdesk articles
# Had to look at the page source to figure this one out!
FRESHDESK_CONTENT_SELECTOR = "div.article-body"  # Works on most Freshdesk sites

# How we split up the text into chunks - played around with these values until they worked well
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks so we don't lose context


# --- Setting up OpenAI ---
openai_client = None
if API_KEY:
    try:
        openai_client = OpenAI(api_key=API_KEY)
        print("OpenAI client initialized for RAG helper.")
        # This lets us use OpenAI to create embeddings (vector representations of text)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=API_KEY,
            model_name=EMBEDDING_MODEL
        )
    except Exception as e:
        print(f"Error initializing OpenAI client or EF: {e}")
        openai_client = None
        openai_ef = None
else:
    print("Warning: OPENAI_API_KEY not found. RAG helper setup may fail.")
    openai_ef = None

# --- Setting up ChromaDB ---
try:
    # This creates/opens our vector database
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # Now we create a collection in the database (or open an existing one)
    if openai_ef:
         collection = chroma_client.get_or_create_collection(
             name=CHROMA_COLLECTION_NAME,
             embedding_function=openai_ef
         )
         print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' loaded/created in '{CHROMA_PERSIST_DIR}'.")
    else:
        collection = None
        print("ChromaDB collection could not be initialized due to missing OpenAI Embedding Function.")

except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    chroma_client = None
    collection = None

# --- The actual functions that do all the work ---

def parse_url_file(filepath="knowledge_urls.txt"):
    """Reads our URL file and returns a list of (name, url) tuples.
    The format in the file should be: Article Name | URL
    I use # for comments in the file."""
    urls_to_process = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                parts = line.split('|', 1)  # Split on the first | only
                if len(parts) == 2:
                    name = parts[0].strip()
                    url = parts[1].strip()
                    if name and url:
                        urls_to_process.append((name, url))
                    else:
                         print(f"Warning: Skipping invalid line: '{line}'")
                else:
                     print(f"Warning: Skipping line with incorrect format: '{line}'")
        print(f"Found {len(urls_to_process)} URLs to process from '{filepath}'.")
        return urls_to_process
    except FileNotFoundError:
        print(f"Error: URL file not found at '{filepath}'")
        return []
    except Exception as e:
         print(f"Error reading URL file '{filepath}': {e}")
         return []

def fetch_and_extract_text(url):
    """Fetches content from a URL and extracts the main text.
    This is where we pull the content from Freshdesk articles."""
    print(f"Fetching: {url}...")
    try:
        # Pretending to be a browser so websites don't block us 🕵️
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)  # 15 sec timeout should be plenty
        response.raise_for_status()  # This raises an error if we get a 404 or other bad response

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content container 
        content_div = soup.select_one(FRESHDESK_CONTENT_SELECTOR)

        if content_div:
            # Get text and clean it up a bit
            text = content_div.get_text(separator='\n', strip=True)
            print(f"Successfully extracted ~{len(text)} chars from {url}")
            return text
        else:
            print(f"Warning: Content selector '{FRESHDESK_CONTENT_SELECTOR}' not found at {url}. Trying body.")
            # Plan B - just grab everything in the body tag (messy but better than nothing)
            body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            if body_text:
                 print("Using fallback body text (may include headers/footers).")
                 return body_text
            else:
                 print(f"Error: Could not find content selector or body text at {url}")
                 return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks.
    We need to do this because the AI models have token limits."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap  # Move forward but with some overlap
        if start >= len(text):  # Stop if we've reached the end
             break
        # Just a safety check - if our math somehow goes wrong
        if start < 0: start = end  # Reset if calculation yields negative index
    return chunks


def load_and_embed_data(force_reload=False):
    """The main function that loads URLs, processes content, and stores embeddings in ChromaDB.
    This is where the real work happens!"""
    global collection  # Need to use the global var if we recreate it
    if not chroma_client or not collection or not openai_ef:
        print("Error: ChromaDB or OpenAI Embedding Function not initialized. Cannot load data.")
        return False

    # Check if we've already loaded data
    existing_count = collection.count()
    print(f"Existing documents in collection: {existing_count}")

    if existing_count > 0 and not force_reload:
        print("Collection already contains data. Skipping loading process. Use force_reload=True to override.")
        return True

    # If we're force reloading, we need to clear the existing collection
    if force_reload and existing_count > 0:
         print("Force reload requested. Clearing existing collection...")
         try:
             chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
             print(f"Old collection '{CHROMA_COLLECTION_NAME}' deleted.")
             collection = chroma_client.get_or_create_collection(
                 name=CHROMA_COLLECTION_NAME,
                 embedding_function=openai_ef
             )
             print("New collection created.")
         except Exception as e:
             print(f"Error clearing collection: {e}. Aborting reload.")
             return False

    # Load articles from our URL file
    urls_data = parse_url_file()
    if not urls_data:
        return False

    # This is where we'll store all our processed text chunks
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_counter = 0

    # Process each article
    for name, url in urls_data:
        article_text = fetch_and_extract_text(url)
        if article_text:
            chunks = split_text_into_chunks(article_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{url}_chunk_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({"source_url": url, "article_name": name, "chunk_index": i})
                all_ids.append(chunk_id)
                chunk_counter += 1
        else:
            print(f"Skipping article due to extraction failure: {name} ({url})")
        time.sleep(0.2)  # Small delay to be nice to the servers

    # Make sure we actually got some text to work with
    if not all_chunks:
        print("No text chunks extracted from any URL. Nothing to embed.")
        return False

    print(f"\nGenerated {len(all_chunks)} text chunks from {len(urls_data)} articles.")
    print("Adding chunks to ChromaDB (this may take time depending on the number of chunks)...")

    try:
        # Process in batches for better performance
        batch_size = 100  # Adjust if needed
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            print(f"Adding batch {i//batch_size + 1} ({len(batch_ids)} items)...")
            collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Batch {i//batch_size + 1} added.")
        print("\nAll data successfully added to ChromaDB collection.")
        return True
    except Exception as e:
        print(f"\nError adding data to ChromaDB: {e}")
        return False


# This runs when we execute this file directly (for testing)
if __name__ == "__main__":
    print("Running RAG Helper setup...")
    # Set force_reload=True to re-process all URLs even if we already have data
    # Set force_reload=False to only load if the collection is empty (saves time and API calls)
    success = load_and_embed_data(force_reload=False)
    if success:
        print("\nData loading process complete.")
        # Uncomment to test a query
        # results = collection.query(query_texts=["How do I upload?"], n_results=2)
        # print("\nTest Query Results:")
        # print(results)
    else:
        print("\nData loading process failed.")