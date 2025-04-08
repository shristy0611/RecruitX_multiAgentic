import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from recruitx_app.services.vector_db_service import VectorDBService, PERSIST_DIRECTORY
from chromadb.api.models.Collection import Collection
from chromadb.api.types import GetResult, QueryResult
from chromadb.errors import ChromaError
from chromadb.utils import embedding_functions

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Test data
TEST_DOCUMENTS = [
    "This is a test job description for a Python developer position.",
    "This is a test CV for a candidate with Python experience."
]

TEST_METADATA = [
    {"doc_type": "job", "job_id": 1, "timestamp": "2025-04-01"},
    {"doc_type": "candidate", "candidate_id": 1, "timestamp": "2025-04-01"}
]

MOCK_QUERY_RESULTS = {
    'ids': [['doc1', 'doc2']],
    'documents': [TEST_DOCUMENTS],
    'metadatas': [TEST_METADATA],
    'distances': [[0.1, 0.2]]
}

EMPTY_QUERY_RESULTS = {
    'ids': [[]],
    'documents': [[]],
    'metadatas': [[]],
    'distances': [[]]
}

class TestVectorDBService:
    
    @pytest.fixture
    def mock_chromadb(self):
        """Create a mock ChromaDB client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Set up the client to return our mock collection
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Configure the collection's add method
        mock_collection.add.return_value = None
        
        # Configure the collection's query method
        mock_collection.query.return_value = MOCK_QUERY_RESULTS
        
        return mock_client
    
    @pytest.fixture
    def vector_db_service(self, mock_chromadb):
        """Create a VectorDBService with a mock ChromaDB client."""
        with patch('chromadb.PersistentClient', return_value=mock_chromadb):
            service = VectorDBService()
            service._client = mock_chromadb
            service._collection = mock_chromadb.get_or_create_collection()
            return service
    
    @patch('recruitx_app.services.vector_db_service.chromadb.PersistentClient')
    @patch('recruitx_app.services.vector_db_service.logger')
    @patch('recruitx_app.services.vector_db_service.os.path.exists')
    @patch('recruitx_app.services.vector_db_service.os.makedirs')
    def test_initialize_client(self, mock_makedirs, mock_exists, mock_logger, mock_chroma_client, vector_db_service):
        """Test successful client initialization when directory does not exist."""
        # Reset client to ensure initialization is tested
        vector_db_service._client = None
        mock_instance = MagicMock()
        mock_chroma_client.return_value = mock_instance
        mock_exists.return_value = False # Simulate directory does not exist

        vector_db_service._initialize_client()

        # Assertions
        mock_exists.assert_called_once_with(PERSIST_DIRECTORY)
        mock_makedirs.assert_called_once_with(PERSIST_DIRECTORY) # Check directory creation
        mock_logger.info.assert_any_call(f"Initializing ChromaDB persistent client at: {PERSIST_DIRECTORY}")
        mock_chroma_client.assert_called_once_with(path=PERSIST_DIRECTORY)
        mock_logger.info.assert_any_call("ChromaDB client initialized successfully.")
        assert vector_db_service._client == mock_instance
        assert vector_db_service._client is not None

    def test_initialize_client_failure(self, vector_db_service):
        """Test handling client initialization failure."""
        # Reset the client
        vector_db_service._client = None
        
        # Mock the client initialization to raise an exception
        with patch('chromadb.PersistentClient', side_effect=Exception("Test initialization error")):
            # Call the initialize method
            vector_db_service._initialize_client()
            
            # Verify that the client remains None
            assert vector_db_service._client is None
    
    def test_get_collection(self, vector_db_service, mock_chromadb):
        """Test getting or creating a collection."""
        # Reset the collection and reset the mock
        vector_db_service._collection = None
        mock_chromadb.get_or_create_collection.reset_mock()
        
        # Mock the embedding function
        with patch.object(vector_db_service, '_get_embedding_function', return_value=MagicMock()):
            # Call the method
            collection = vector_db_service.get_collection()
            
            # Verify the collection was retrieved
            assert collection is not None
            mock_chromadb.get_or_create_collection.assert_called_once()
    
    def test_get_collection_no_client(self, vector_db_service):
        """Test getting a collection when client is None."""
        # Reset the collection and client
        vector_db_service._collection = None
        vector_db_service._client = None
        
        # Call the method
        collection = vector_db_service.get_collection()
        
        # Verify the result
        assert collection is None
    
    def test_get_collection_no_embedding_function(self, vector_db_service):
        """Test getting a collection when embedding function is None."""
        # Reset the collection
        vector_db_service._collection = None
        
        # Mock embedding function to return None
        with patch.object(vector_db_service, '_get_embedding_function', return_value=None):
            # Call the method
            collection = vector_db_service.get_collection()
            
            # Verify the result
            assert collection is None
    
    def test_get_collection_exception(self, vector_db_service, mock_chromadb):
        """Test handling exception when getting collection."""
        # Reset the collection
        vector_db_service._collection = None
        
        # Setup mock to raise exception
        mock_chromadb.get_or_create_collection.side_effect = Exception("Test collection error")
        
        # Mock embedding function
        with patch.object(vector_db_service, '_get_embedding_function', return_value=MagicMock()):
            # Call the method
            collection = vector_db_service.get_collection()
            
            # Verify the result
            assert collection is None
    
    def test_get_embedding_function(self, vector_db_service):
        """Test getting the embedding function."""
        # Reset the embedding function
        vector_db_service._embedding_function = None
        
        # Direct patch of the module functions used in the method
        with patch('recruitx_app.services.vector_db_service.settings') as mock_settings, \
             patch('recruitx_app.services.vector_db_service.embedding_functions.GoogleGenerativeAiEmbeddingFunction', return_value=MagicMock()) as mock_embedding:
            
            # Configure the mock settings
            mock_settings.get_next_api_key.return_value = "test-api-key"
            mock_settings.GEMINI_EMBEDDING_MODEL = "test-embedding-model"
            
            # Call the method
            result = vector_db_service._get_embedding_function()
            
            # Verify the embedding function was created
            assert result is not None
            mock_settings.get_next_api_key.assert_called_once()
            mock_embedding.assert_called_once_with(
                api_key="test-api-key",
                model_name=mock_settings.GEMINI_EMBEDDING_MODEL
            )
    
    def test_get_embedding_function_exception(self, vector_db_service):
        """Test handling exception when getting embedding function."""
        # Reset the embedding function
        vector_db_service._embedding_function = None
        
        # Direct patch of the module functions used in the method
        with patch('recruitx_app.services.vector_db_service.settings') as mock_settings, \
             patch('recruitx_app.services.vector_db_service.embedding_functions.GoogleGenerativeAiEmbeddingFunction', 
                  side_effect=Exception("Test embedding error")):
            
            # Configure the mock settings
            mock_settings.get_next_api_key.return_value = "test-api-key"
            mock_settings.GEMINI_EMBEDDING_MODEL = "test-embedding-model"
            
            # Call the method
            result = vector_db_service._get_embedding_function()
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, vector_db_service):
        """Test generating embeddings from the embedding model."""
        embedding_function = MagicMock()
        embedding_function.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        with patch.object(vector_db_service, '_get_embedding_function', return_value=embedding_function):
            # Call the method
            result = await vector_db_service.generate_embeddings(["Test document 1", "Test document 2"])
            
            # Verify that the embedding function was called
            embedding_function.assert_called_once_with(["Test document 1", "Test document 2"])
            
            # Verify the result
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_no_function(self, vector_db_service):
        """Test generating embeddings when embedding function is None."""
        with patch.object(vector_db_service, '_get_embedding_function', return_value=None):
            # Call the method
            result = await vector_db_service.generate_embeddings(["Test document"])
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_exception(self, vector_db_service):
        """Test handling exception when generating embeddings."""
        embedding_function = MagicMock()
        embedding_function.side_effect = Exception("Test embedding error")
        
        with patch.object(vector_db_service, '_get_embedding_function', return_value=embedding_function):
            # Call the method
            result = await vector_db_service.generate_embeddings(["Test document"])
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_add_document_chunks(self, vector_db_service):
        """Test adding document chunks to the vector database."""
        # Mock the get_collection method
        with patch.object(vector_db_service, 'get_collection') as mock_get_collection:
            mock_get_collection.return_value = vector_db_service._collection
            
            # Call the method
            result = await vector_db_service.add_document_chunks(
                documents=TEST_DOCUMENTS,
                metadatas=TEST_METADATA,
                ids=["doc1", "doc2"]
            )
            
            # Verify that the collection's add method was called
            vector_db_service._collection.add.assert_called_once()
            call_args = vector_db_service._collection.add.call_args[1]
            assert call_args["documents"] == TEST_DOCUMENTS
            assert call_args["metadatas"] == TEST_METADATA
            assert call_args["ids"] == ["doc1", "doc2"]
            
            # Verify the result
            assert result is True
    
    @pytest.mark.asyncio
    async def test_add_document_chunks_no_collection(self, vector_db_service):
        """Test adding document chunks when collection is None."""
        with patch.object(vector_db_service, 'get_collection', return_value=None):
            # Call the method
            result = await vector_db_service.add_document_chunks(
                documents=TEST_DOCUMENTS,
                metadatas=TEST_METADATA,
                ids=["doc1", "doc2"]
            )
            
            # Verify the result
            assert result is False
    
    @pytest.mark.asyncio
    async def test_add_document_chunks_no_embedding_function(self, vector_db_service):
        """Test adding document chunks when embedding function is None."""
        # Create a collection without embedding function
        mock_collection = MagicMock()
        mock_collection.embedding_function = None
        
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            # Call the method
            result = await vector_db_service.add_document_chunks(
                documents=TEST_DOCUMENTS,
                metadatas=TEST_METADATA,
                ids=["doc1", "doc2"]
            )
            
            # Verify the result
            assert result is False
    
    @pytest.mark.asyncio
    async def test_add_document_chunks_exception(self, vector_db_service):
        """Test handling exception when adding document chunks."""
        # Mock collection with add method that raises exception
        mock_collection = MagicMock()
        mock_collection.add.side_effect = Exception("Test add error")
        
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            # Call the method
            result = await vector_db_service.add_document_chunks(
                documents=TEST_DOCUMENTS,
                metadatas=TEST_METADATA,
                ids=["doc1", "doc2"]
            )
            
            # Verify the result
            assert result is False
    
    @pytest.mark.asyncio
    async def test_query_collection(self, vector_db_service):
        """Test querying the vector database."""
        # Mock the get_collection method
        with patch.object(vector_db_service, 'get_collection') as mock_get_collection:
            mock_get_collection.return_value = vector_db_service._collection
            
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Python developer with 5 years experience"],
                n_results=2,
                where={"doc_type": "job"}
            )
            
            # Verify that the collection's query method was called
            vector_db_service._collection.query.assert_called_once()
            call_args = vector_db_service._collection.query.call_args[1]
            assert call_args["query_texts"] == ["Python developer with 5 years experience"]
            assert call_args["n_results"] == 2
            assert call_args["where"] == {"doc_type": "job"}
            
            # Verify the result
            assert result == MOCK_QUERY_RESULTS
    
    @pytest.mark.asyncio
    async def test_query_collection_no_collection(self, vector_db_service):
        """Test querying when collection is None."""
        with patch.object(vector_db_service, 'get_collection', return_value=None):
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Test query"]
            )
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_query_collection_no_embedding_function(self, vector_db_service):
        """Test querying when embedding function is None."""
        # Create a collection without embedding function
        mock_collection = MagicMock()
        mock_collection.embedding_function = None
        
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Test query"]
            )
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_query_collection_exception(self, vector_db_service):
        """Test handling exception when querying collection."""
        # Mock collection with query method that raises exception
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("Test query error")
        
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Test query"]
            )
            
            # Verify the result
            assert result is None
    
    @pytest.mark.asyncio
    async def test_query_collection_empty_results(self, vector_db_service):
        """Test querying when results are empty."""
        # Mock collection with query method that returns empty results
        mock_collection = MagicMock()
        mock_collection.query.return_value = EMPTY_QUERY_RESULTS
        
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            # Call the method
            result = await vector_db_service.query_collection(
                query_texts=["Test query"]
            )
            
            # Verify the result structure
            assert result == EMPTY_QUERY_RESULTS
            assert len(result['ids'][0]) == 0
            assert len(result['documents'][0]) == 0

    def test_singleton_pattern(self):
        """Test that the VectorDBService follows the singleton pattern."""
        # Create two instances of the service
        with patch('chromadb.PersistentClient', return_value=MagicMock()):
            service1 = VectorDBService()
            service2 = VectorDBService()
            
            # Verify they are the same instance
            assert service1 is service2

    @patch('recruitx_app.services.vector_db_service.logger.error')
    def test_initialize_client_logs_error(self, mock_logger_error, vector_db_service):
        """Test error logging during client initialization failure."""
        vector_db_service._client = None # Ensure client is reset
        with patch('chromadb.PersistentClient', side_effect=Exception("Init Error!")):
            vector_db_service._initialize_client()
            assert vector_db_service._client is None
            mock_logger_error.assert_called_once()
            assert "Failed to initialize ChromaDB client" in mock_logger_error.call_args[0][0]

    @patch('recruitx_app.services.vector_db_service.logger.error')
    def test_get_embedding_function_logs_error(self, mock_logger_error, vector_db_service):
        """Test error logging during embedding function initialization failure."""
        vector_db_service._embedding_function = None # Ensure it's reset
        with patch('recruitx_app.services.vector_db_service.embedding_functions.GoogleGenerativeAiEmbeddingFunction', side_effect=Exception("Embedding Init Error!")):
            result = vector_db_service._get_embedding_function()
            assert result is None
            mock_logger_error.assert_called_once()
            assert "Failed to initialize Google Generative AI embedding function" in mock_logger_error.call_args[0][0]

    @patch('recruitx_app.services.vector_db_service.logger.error')
    def test_get_collection_logs_error_no_client(self, mock_logger_error, vector_db_service):
        """Test error logging when getting collection with no client."""
        vector_db_service._client = None
        vector_db_service._collection = None
        result = vector_db_service.get_collection()
        assert result is None
        mock_logger_error.assert_called_with("ChromaDB client is not initialized. Cannot get collection.")

    @patch('recruitx_app.services.vector_db_service.logger.error')
    def test_get_collection_logs_error_no_embedding_func(self, mock_logger_error, vector_db_service):
        """Test error logging when getting collection with no embedding function."""
        vector_db_service._collection = None # Reset collection
        # Ensure client exists but embedding function fails
        vector_db_service._initialize_client() # Make sure client is not None
        with patch.object(vector_db_service, '_get_embedding_function', return_value=None):
            result = vector_db_service.get_collection()
            assert result is None
            mock_logger_error.assert_called_with("Embedding function failed to initialize. Cannot get collection.")

    @patch('recruitx_app.services.vector_db_service.logger.error')
    def test_get_collection_logs_error_on_exception(self, mock_logger_error, vector_db_service, mock_chromadb):
        """Test error logging when get_or_create_collection raises an exception."""
        vector_db_service._collection = None # Reset collection
        # Mock the call to raise an exception
        mock_chromadb.get_or_create_collection.side_effect = Exception("Collection Creation Error!")
        # Ensure embedding function exists
        with patch.object(vector_db_service, '_get_embedding_function', return_value=MagicMock()):
            result = vector_db_service.get_collection()
            assert result is None
            mock_logger_error.assert_called_once()
            assert "Failed to get or create collection" in mock_logger_error.call_args[0][0]

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_add_document_logs_error_no_collection(self, mock_logger_error, vector_db_service):
        """Test error logging when adding documents with no collection."""
        with patch.object(vector_db_service, 'get_collection', return_value=None):
            result = await vector_db_service.add_document_chunks([], [], [])
            assert result is False
            mock_logger_error.assert_called_with("Cannot add documents, collection not available.")

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_add_document_logs_error_no_embedding_func(self, mock_logger_error, vector_db_service):
        """Test error logging when adding documents with no embedding func in collection."""
        mock_collection = MagicMock()
        mock_collection.embedding_function = None
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            result = await vector_db_service.add_document_chunks([], [], [])
            assert result is False
            mock_logger_error.assert_called_with("Cannot add documents, embedding function is not configured for the collection.")

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_add_document_logs_error_on_exception(self, mock_logger_error, vector_db_service):
        """Test error logging when collection.add raises an exception."""
        mock_collection = MagicMock()
        mock_collection.add.side_effect = Exception("DB Add Error!")
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            result = await vector_db_service.add_document_chunks([], [], [])
            assert result is False
            mock_logger_error.assert_called_once()
            assert "Failed to add documents to collection" in mock_logger_error.call_args[0][0]

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_query_collection_logs_error_no_collection(self, mock_logger_error, vector_db_service):
        """Test error logging when querying with no collection."""
        with patch.object(vector_db_service, 'get_collection', return_value=None):
            result = await vector_db_service.query_collection(["test"])
            assert result is None
            mock_logger_error.assert_called_with("Cannot query collection, collection not available.")

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_query_collection_logs_error_no_embedding_func(self, mock_logger_error, vector_db_service):
        """Test error logging when querying with no embedding func in collection."""
        mock_collection = MagicMock()
        mock_collection.embedding_function = None
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            result = await vector_db_service.query_collection(["test"])
            assert result is None
            mock_logger_error.assert_called_with("Cannot query collection, embedding function is not configured.")

    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.error')
    async def test_query_collection_logs_error_on_exception(self, mock_logger_error, vector_db_service):
        """Test error logging when collection.query raises an exception."""
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("DB Query Error!")
        with patch.object(vector_db_service, 'get_collection', return_value=mock_collection):
            result = await vector_db_service.query_collection(["test"])
            assert result is None
            mock_logger_error.assert_called_once()
            assert "Failed to query collection" in mock_logger_error.call_args[0][0]

    # Test for logging successful query
    @pytest.mark.asyncio
    @patch('recruitx_app.services.vector_db_service.logger.info')
    async def test_query_collection_logs_success_info(self, mock_logger_info, vector_db_service):
        """Test info logging after a successful query."""
        with patch.object(vector_db_service, 'get_collection') as mock_get_collection:
            mock_get_collection.return_value = vector_db_service._collection
            # Ensure the mock collection returns the expected query result structure
            vector_db_service._collection.query.return_value = MOCK_QUERY_RESULTS

            await vector_db_service.query_collection(query_texts=["test query"])

            # Check if logger.info was called with the specific success message
            found_log = False
            for call in mock_logger_info.call_args_list:
                if "Query 'test query...' returned 2 results." in call[0][0]:
                    found_log = True
                    break
            assert found_log, "Successful query info log not found"

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('recruitx_app.services.vector_db_service.logger.info')
    def test_initialize_client_directory_exists(self, mock_logger_info, mock_makedirs, mock_exists, vector_db_service):
        """Test client initialization when the persistence directory already exists."""
        vector_db_service._client = None # Reset client
        mock_exists.return_value = True # Simulate directory existing

        with patch('chromadb.PersistentClient') as mock_chroma_client:
            vector_db_service._initialize_client()

            # Verify os.makedirs was NOT called
            mock_makedirs.assert_not_called()

            # Verify logger info messages
            assert mock_logger_info.call_count >= 2 # Init message + Success message
            assert "Initializing ChromaDB persistent client" in mock_logger_info.call_args_list[0][0][0]
            assert "ChromaDB client initialized successfully." in mock_logger_info.call_args_list[1][0][0]

            # Verify ChromaDB client was initialized
            mock_chroma_client.assert_called_once() 