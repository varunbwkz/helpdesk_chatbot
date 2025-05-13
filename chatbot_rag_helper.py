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
        """Init the RAG helper with config parameters"""
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
            formatted_response = f"\n\n{response}\n\n"
            
            # Add formatted source attribution
            if sources:
                formatted_response += "### 📚 Sources\n\n"
                for i, (name, url) in enumerate(sources, 1):
                    formatted_response += f"{i}. [{name}]({url})\n"
                    
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"### ⚠️ Error\n\nAn error occurred while processing your request. Please try again."