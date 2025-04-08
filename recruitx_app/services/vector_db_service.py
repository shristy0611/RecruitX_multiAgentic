import chromadb
import logging
import os
from typing import Optional, List, Dict, Any

# Import ChromaDB utility and our settings
import chromadb.utils.embedding_functions as embedding_functions
from recruitx_app.core.config import settings 

logger = logging.getLogger(__name__)

# Determine the path for persistent storage
# Place it within the project root, maybe in a .vector_store directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, ".vector_store")

class VectorDBService:
    """
    Service for interacting with the ChromaDB vector store.
    Handles client initialization, collection management, embedding, and CRUD operations.
    """
    _instance = None
    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[chromadb.Collection] = None
    _embedding_function: Optional[embedding_functions.GoogleGenerativeAiEmbeddingFunction] = None

    COLLECTION_NAME = "recruitx_documents"

    def __new__(cls):
        # Singleton pattern to ensure only one client instance
        if cls._instance is None:
            cls._instance = super(VectorDBService, cls).__new__(cls)
            cls._instance._initialize_client()
            cls._instance._get_embedding_function() # Initialize embedding function on creation
        return cls._instance

    def _initialize_client(self):
        """Initializes the persistent ChromaDB client."""
        if self._client is None:
            try:
                if not os.path.exists(PERSIST_DIRECTORY):
                    os.makedirs(PERSIST_DIRECTORY)
                    logger.info(f"Created persistence directory: {PERSIST_DIRECTORY}")
                    
                logger.info(f"Initializing ChromaDB persistent client at: {PERSIST_DIRECTORY}")
                self._client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
                logger.info("ChromaDB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
                self._client = None # Ensure client is None if init fails

    def _get_embedding_function(self):
        """Initializes and returns the Google Generative AI embedding function."""
        if self._embedding_function is None:
            try:
                logger.info(f"Initializing Google Generative AI embedding function with model: {settings.GEMINI_EMBEDDING_MODEL}")
                # Using API key rotation for safety, although Chroma might cache the function instance
                self._embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                    api_key=settings.get_next_api_key(), 
                    model_name=settings.GEMINI_EMBEDDING_MODEL
                )
                logger.info("Google Generative AI embedding function initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Google Generative AI embedding function: {e}", exc_info=True)
                self._embedding_function = None
        return self._embedding_function

    def get_collection(self) -> Optional[chromadb.Collection]:
        """
        Gets or creates the default ChromaDB collection with the embedding function.
        
        Returns:
            The ChromaDB Collection object or None if the client failed to initialize.
        """
        if self._client is None:
            logger.error("ChromaDB client is not initialized. Cannot get collection.")
            return None
            
        embedding_func = self._get_embedding_function()
        if embedding_func is None:
             logger.error("Embedding function failed to initialize. Cannot get collection.")
             return None
             
        if self._collection is None:
            try:
                logger.info(f"Getting or creating ChromaDB collection: {self.COLLECTION_NAME}")
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=embedding_func # Assign the embedding function
                )
                logger.info(f"Collection '{self.COLLECTION_NAME}' ready.")
            except Exception as e:
                logger.error(f"Failed to get or create collection '{self.COLLECTION_NAME}': {e}", exc_info=True)
                self._collection = None
                
        return self._collection

    # --- NEW Method: Generate Embeddings --- 
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generates embeddings for a list of texts using the configured function."""
        embedding_func = self._get_embedding_function()
        if not embedding_func:
            logger.error("Cannot generate embeddings, embedding function not available.")
            return None
            
        try:
            # The embedding function itself is usually callable like this
            embeddings = embedding_func(texts)
            if embeddings:
                 logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return None

    # --- Add document and query methods --- 

    async def add_document_chunks(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """
        Adds document chunks (text, metadata, ids) to the collection.
        Embeddings are generated automatically by the configured embedding function.
        """
        collection = self.get_collection()
        if not collection:
            logger.error("Cannot add documents, collection not available.")
            return False
            
        if collection.embedding_function is None:
             logger.error("Cannot add documents, embedding function is not configured for the collection.")
             return False

        try:
            # ChromaDB's add method handles embedding generation via the collection's function
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added/updated {len(ids)} chunks to collection '{self.COLLECTION_NAME}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to collection: {e}", exc_info=True)
            return False

    async def query_collection(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Queries the collection for relevant document chunks using the configured embedding function.
        """
        collection = self.get_collection()
        if not collection:
            logger.error("Cannot query collection, collection not available.")
            return None

        if collection.embedding_function is None:
             logger.error("Cannot query collection, embedding function is not configured.")
             return None
            
        try:
            # ChromaDB's query method handles embedding the query_texts automatically
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where, # Optional filter
                include=['metadatas', 'documents', 'distances'] # Include useful info
            )
            # Log query results concisely
            num_results = len(results.get('ids', [[]])[0]) if results and results.get('ids') else 0
            query_preview = query_texts[0][:70] + "..." if query_texts else "N/A"
            logger.info(f"Query '{query_preview}' returned {num_results} results.")
            return results
        except Exception as e:
            logger.error(f"Failed to query collection: {e}", exc_info=True)
            return None

# Instantiate the service so it can be imported
vector_db_service = VectorDBService() 